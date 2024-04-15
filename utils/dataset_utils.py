import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import textwrap
from tqdm import tqdm
import random
from utils.poly_utils import is_clockwise, revert_direction, reorder_points, approximate_polygons, interpolate_polygons, polygons_to_string
from typing import Union


class DatasetUtils:
    @staticmethod
    def write_image(
        data: dict[str, Union[list[np.ndarray], np.ndarray]],
        save_path: str,
        key: str,
        file_name_prefix: str = None,
        append_to_exsiting_file: bool = True,
        jpeg_quality: int = 80,
        png_compression: int = 9,
    ) -> list[int]:
        # create directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if append_to_exsiting_file:
            # get number of existing files
            num_existing_files = len(os.listdir(save_path))
        else:
            num_existing_files = 0
            # delete all existing files
            for file_name in os.listdir(save_path):
                os.remove(os.path.join(save_path, file_name))
        # unify data type to dict[str, list[np.ndarray]]
        if isinstance(data[key], np.ndarray):
            data[key] = [data[key]]

        match key:
            case "colors":
                # write images with name like file_name_prefix_00000001.jpg
                for i, img in enumerate(data[key]):
                    img = img.astype(np.uint8)
                    file_name = f"{file_name_prefix}_{i+num_existing_files:08d}.jpg"
                    # convert rgb to bgr
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(os.path.join(save_path, file_name), bgr_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    if not success:
                        raise Exception(f"Failed to write image {file_name}")

            case "depth":
                # write images with name like file_name_prefix_00000001.png
                for i, img in enumerate(data[key]):
                    img = img.astype(np.float16)
                    img.dtype = np.uint16
                    file_name = f"{file_name_prefix}_{i+num_existing_files:08d}.png"
                    success = cv2.imwrite(os.path.join(save_path, file_name), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
                    if not success:
                        raise Exception(f"Failed to write image {file_name}")

            case "scene_id_segmaps":
                # write images with name like file_name_prefix_00000001.png
                for i, img in enumerate(data[key]):
                    img = img.astype(np.uint8)
                    file_name = f"{file_name_prefix}_{i+num_existing_files:08d}.png"
                    success = cv2.imwrite(os.path.join(save_path, file_name), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
                    if not success:
                        raise Exception(f"Failed to write image {file_name}")
            case _:
                raise ValueError(f"key {key} is not supported")

        return list(range(num_existing_files, num_existing_files + len(data[key])))

    @staticmethod
    def write_expressions(img_idxs: list[int], expressions: dict, save_path: str, file_name_prefix: str = None) -> int:
        # create directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # write expressions
        num_existing_files = len(os.listdir(save_path))
        file_name = f"{file_name_prefix}_{num_existing_files:08d}.json"
        with open(os.path.join(save_path, file_name), "w") as f:
            json.dump({"img_idxs": img_idxs, "expressions": expressions}, f)
        return num_existing_files

    @staticmethod
    def merge_expressions(save_path: str, temp_path: str, file_name: str = "expressions.json", delete_temp: bool = True) -> int:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # create empty json list if not exists
        if not os.path.exists(os.path.join(save_path, file_name)):
            with open(os.path.join(save_path, file_name), "w") as f:
                json.dump([], f)
        # add jsons from temp_path to the save_path json
        with open(os.path.join(save_path, file_name), "r") as f:
            expressions = json.load(f)
        for file_to_merge in os.listdir(temp_path):
            with open(os.path.join(temp_path, file_to_merge), "r") as f:
                expressions_to_merge = json.load(f)
            expressions.append(expressions_to_merge)
        # write expressions
        with open(os.path.join(save_path, file_name), "w") as f:
            json.dump(expressions, f)
        # delete temp_path
        if delete_temp:
            for file_to_delete in os.listdir(temp_path):
                os.remove(os.path.join(temp_path, file_to_delete))
            os.rmdir(temp_path)
        return len(expressions)

    @staticmethod
    def check_image_file_nums(base_path: str, folder_names: list[str]) -> bool:
        # check if all folders have the same number of files
        num_files = []
        for folder_name in folder_names:
            try:
                num_files.append(len(os.listdir(os.path.join(base_path, folder_name))))
            except FileNotFoundError:
                num_files.append(0)
        if len(set(num_files)) == 1:
            return True
        else:
            return False

    @staticmethod
    def visualize_image(
        rgb_img_path: str,
        segmap_path: str,
        expressions_json_path: str,
        display_bbox: bool = True,
        display_mask: bool = True,
        display_polygon: bool = True,
        show_image: bool = True,
        save_image: bool = False,
        save_path: str = None,
        expression_json_temp: list = None,
    ) -> None:
        # get img index from rgb_img_path like ./output/rgb/rgb_00000001.jpg
        img_idx = int(rgb_img_path.split("_")[-1].split(".")[0])
        # retrive expressions of img_idx
        if expression_json_temp is not None and expression_json_temp != []:
            for item in expression_json_temp:
                if img_idx in item["img_idxs"]:
                    expressions = item["expressions"]
                    break
        else:
            with open(expressions_json_path, "r") as f:
                data = json.load(f)
                for item in data:
                    if img_idx in item["img_idxs"]:
                        expressions = item["expressions"]
                        break
                # update expression_json_temp with data
                if expression_json_temp is not None:
                    expression_json_temp += data
        # read rgb image
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # read segmap
        segmap = cv2.imread(segmap_path, cv2.IMREAD_UNCHANGED)
        # clear image
        plt.clf()
        # add segmap to rgb image
        for i, expression in enumerate(expressions):
            # get bound box
            bound_box = DatasetUtils.get_bound_box(segmap, expression["obj"]["scene_id"])
            if bound_box is None:
                continue
            top_left, bottom_right = bound_box
            # randomize color for bound box and mask
            color = np.random.uniform((0.3, 0.3, 0.3), (0.8, 0.8, 0.8))
            # display bound box
            if display_bbox:
                plt.gca().add_patch(plt.Rectangle(top_left, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1], fill=False, color=color))
            # add mask with specified color
            if display_mask:
                mask = segmap == expression["obj"]["scene_id"]
                rgb_img[mask, :] = rgb_img[mask, :] * 0.6 + color * 0.4 * 255
            if display_polygon:
                polygons = DatasetUtils.get_polygons(segmap, expression["obj"]["scene_id"])
                for polygon in polygons:
                    # scatter
                    plt.scatter(polygon[:, 0, 0], polygon[:, 0, 1], s=5, color=color, alpha=0.5)
            # display text
            plt.text(
                top_left[0] + 1,
                top_left[1] + 1,
                i,
                color=color,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, linewidth=0, pad=1),
                va="top",
                ha="left",
            )
            # display sentence on the right of the img
            plt.text(
                rgb_img.shape[1] + 20,
                i * 80,
                textwrap.fill(f"{i}: {expression['expression']}", width=70),
                color=color,
                fontsize=10,
                va="top",
                ha="left",
            )
        plt.axis("off")
        plt.imshow(rgb_img)
        if save_image:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f"{img_idx:08d}.jpg"), bbox_inches="tight", pil_kwargs={"quality": 95})
        if show_image:
            plt.show()

    @staticmethod
    def get_bound_box(mask: np.ndarray, poi_idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
        # get bound box of a mask
        # mask: (H,W)
        # poi_idx: index of point of interest
        # return: (top_left,bottom_right)
        # get indices of non-zero elements
        poi_mask = mask == poi_idx
        non_zero_indices = np.nonzero(poi_mask)
        if non_zero_indices[0].size == 0:
            return None
        # get top left and bottom right points
        top_left = (np.min(non_zero_indices[1]), np.min(non_zero_indices[0]))
        bottom_right = (np.max(non_zero_indices[1]), np.max(non_zero_indices[0]))
        return (top_left, bottom_right)

    # @staticmethod
    # def GetMask(mask: np.ndarray, poi_idx: int) -> np.ndarray:
    #     # 255 for poi_idx, 0 for others
    #     return (mask == poi_idx).astype(np.uint8) * 255

    @staticmethod
    def get_polygons(mask: np.ndarray, poi_idx: int) -> list[np.ndarray]:
        polygons = cv2.findContours((mask == poi_idx).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
        return polygons

    @staticmethod
    def generate_tsv_file_for_REC(
        tsv_path: str, dataset_path: str, tsv_filename: str = None, expression_json_temp: list = None, shuffle: bool = True
    ) -> None:
        if expression_json_temp is not None and expression_json_temp != []:
            expressions = expression_json_temp
        else:
            with open(os.path.join(dataset_path, "expressions.json"), "r") as f:
                expressions = json.load(f)
        lines = []
        for i, item in enumerate(tqdm(expressions)):
            for img_idx in item["img_idxs"]:
                rgb_img_path = os.path.join(dataset_path, "rgb", f"rgb_{img_idx:08d}.jpg")
                mask_img_path = os.path.join(dataset_path, "mask", f"mask_{img_idx:08d}.png")
                mask = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
                for item_expressions in item["expressions"]:
                    bound_box = DatasetUtils.get_bound_box(mask, item_expressions["obj"]["scene_id"])
                    expression = item_expressions["expression"]
                    # * tsv format: index \t expression \t x1,y1,x2,y2 \t path_to_image
                    line = f"{len(lines)}\t{expression}\t{bound_box[0][0]},{bound_box[0][1]},{bound_box[1][0]},{bound_box[1][1]}\t{rgb_img_path}\n"
                    # replace "\" with "/"
                    line = line.replace("\\", "/")
                    lines.append(line)
        # shuffle lines
        if shuffle:
            random.shuffle(lines)
        # makedir if necessary and write tsv file
        if not os.path.exists(tsv_path):
            os.makedirs(tsv_path)
        if tsv_filename is None:
            tsv_filename = "train_shuffled.tsv"
        with open(os.path.join(tsv_path, tsv_filename), "w") as f:
            f.writelines(lines)

    @staticmethod
    def generate_tsv_file_for_RES(
        tsv_path: str, dataset_path: str, tsv_filename: str = None, expression_json_temp: list = None, shuffle: bool = True
    ) -> None:
        if expression_json_temp is not None and expression_json_temp != []:
            expressions = expression_json_temp
        else:
            with open(os.path.join(dataset_path, "expressions.json"), "r") as f:
                expressions = json.load(f)
        lines = []
        for i, item in enumerate(tqdm(expressions)):
            for img_idx in item["img_idxs"]:
                rgb_img_path = os.path.join(dataset_path, "rgb", f"rgb_{img_idx:08d}.jpg")
                mask_img_path = os.path.join(dataset_path, "mask", f"mask_{img_idx:08d}.png")
                mask = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
                for item_expressions in item["expressions"]:
                    bound_box = DatasetUtils.get_bound_box(mask, item_expressions["obj"]["scene_id"])
                    expression = item_expressions["expression"]
                    polygons = DatasetUtils.get_polygons(mask, item_expressions["obj"]["scene_id"])
                    # reshape to array [[x1,y1,x2,y2,...],...]
                    polygons = [poly.flatten().tolist() for poly in polygons]

                    polygons_processed = []
                    for polygon in polygons:
                        # make the polygon clockwise
                        if not is_clockwise(polygon):
                            polygon = revert_direction(polygon)

                        # reorder the polygon so that the first vertex is the one closest to image origin
                        polygon = reorder_points(polygon)
                        polygons_processed.append(polygon)

                    polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
                    polygons_interpolated = interpolate_polygons(polygons)
                    polygons = approximate_polygons(polygons)
                    pts_string = polygons_to_string(polygons)
                    pts_string_interpolated = polygons_to_string(polygons_interpolated)

                    # * tsv format: index \t path_to_image \t boundbox(x1,y1,x2,y2) \t expression \t polygon \t polygon_interpolated
                    line = (
                        f"{len(lines)}\t{rgb_img_path}\t{bound_box[0][0]},{bound_box[0][1]},{bound_box[1][0]},{bound_box[1][1]}"
                        + f"\t{expression}\t{pts_string}\t{pts_string_interpolated}\n"
                    )
                    # replace "\" with "/"
                    line = line.replace("\\", "/")
                    lines.append(line)
        # shuffle lines
        if shuffle:
            random.shuffle(lines)
        # makedir if necessary and write tsv file
        if not os.path.exists(tsv_path):
            os.makedirs(tsv_path)
        if tsv_filename is None:
            tsv_filename = "train_shuffled.tsv"
        with open(os.path.join(tsv_path, tsv_filename), "w") as f:
            f.writelines(lines)

    @staticmethod
    def rename_obj_files(obj_file: str, mtl_file: str, texture_file: str, new_name_prefix: str) -> None:
        """
        Rename the obj, mtl and texture files to a new name. Make sure three files are in the same folder.

        Example: `rename_obj_files('chair.obj','chair.mtl','chair.png','new_chair')` will rename the three files to 'new_chair.obj','new_chair.mtl','new_chair.png'.

        Args:
            obj_file (str): The path of the obj file (.obj).
            mtl_file (str): The path of the mtl file (.mtl).
            texture_file (str): The path of the texture file (.png or .jpg).
            new_name_prefix (str): The new name prefix of the three files.
        """
        import os

        png_or_jpg = texture_file.split(".")[-1]
        with open(obj_file, "r") as obj_file_read:
            obj_lines = obj_file_read.readlines()
            for i, line in enumerate(obj_lines):
                # search for line containing 'mtllib'
                if "mtllib" in line:
                    # change the mtl file name to new name
                    obj_lines[i] = line.replace(line.split()[1], new_name_prefix + ".mtl")
                    break
        with open(obj_file, "w") as obj_file_write:
            obj_file_write.writelines(obj_lines)
        with open(mtl_file, "r") as mtl_file_read:
            mtl_lines = mtl_file_read.readlines()
            for i, line in enumerate(mtl_lines):
                if "map_Kd" in line:
                    # change the png file name to new name without changing other content
                    mtl_lines[i] = line.replace(line.split()[1], new_name_prefix + "." + png_or_jpg)
                    break
        with open(mtl_file, "w") as mtl_file_write:
            mtl_file_write.writelines(mtl_lines)
        # rename all files
        file_folder = os.path.dirname(obj_file)
        os.rename(obj_file, os.path.join(file_folder, new_name_prefix + ".obj"))
        os.rename(mtl_file, os.path.join(file_folder, new_name_prefix + ".mtl"))
        os.rename(texture_file, os.path.join(file_folder, new_name_prefix + "." + png_or_jpg))
