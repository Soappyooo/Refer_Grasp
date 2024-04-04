import blenderproc as bproc
import numpy as np
import os
import sys
import time
import json
import bpy
from mathutils import Vector

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

sys.path.append(os.path.abspath("."))

from scene_graph_generation import SceneGraph
from utils.dataset_utils import DatasetUtils

obj_dir = "./models/ycb_models"  # 模型存放目录
background_dir = "."  # 背景blend文件存放目录
background_file = "background1.blend"  # 背景blend文件名
output_log = "./output_log.txt"
output_dir = "./output"
models_info_file = "./models/models_info.xlsx"
iterations = 40
images_per_iteration = 5  # 每个iteration生成的图片数量，不同视角
resolution_w = 512
resolution_h = 512
scene_graph_rows = 4
scene_graph_cols = 4


bproc.init()
f = open(output_log, "w")
f.write(f"[{time.asctime()}]: started\n")

# check if output_dir has files
for _, _, files in os.walk(output_dir):
    if files != []:
        f.write(f"[{time.asctime()}]: WARNING: output_dir ({output_dir}) is not empty\n")
        print(f"WARNING: output_dir ({output_dir}) is not empty, continue? (y/n)")
        if input() != "y":
            f.write(f"[{time.asctime()}]: stopped because output_dir ({output_dir}) is not empty\n")
            print("stopped because output_dir is not empty")
            f.close()
            sys.exit()
        else:
            break

s = SceneGraph((scene_graph_rows, scene_graph_cols))
s.LoadModelsInfo(models_info_file)
print("loading background...")
# load background into the scene
background_objs = bproc.loader.load_blend(os.path.join(background_dir, background_file))
for obj in background_objs:
    obj.set_cp("obj_id", None)
    obj.set_cp("scene_id", 255)

surface: bproc.types.MeshObject = bproc.filter.one_by_attr(background_objs, "name", "surface")
surface.hide(True)
walls: list[bproc.types.MeshObject] = bproc.filter.by_attr(background_objs, "name", "^plane(?!$).*$", regex=True)
for wall in walls:
    wall.enable_rigidbody(False, friction=1)
floor: bproc.types.MeshObject = bproc.filter.one_by_attr(background_objs, "name", "floor")
floor.enable_rigidbody(False, friction=1)

light = bproc.types.Light()
light.set_type("POINT")

bproc.renderer.enable_depth_output(activate_antialiasing=False)

active_objs: list[bproc.types.MeshObject] = []
print("loading objects...")

for file_name in os.listdir(obj_dir):
    if file_name.endswith(".obj"):
        obj = bproc.loader.load_obj(os.path.join(obj_dir, file_name), use_legacy_obj_import=True)[0]
        active_objs.append(obj)
        obj.set_cp("obj_id", file_name.split(".")[0])  # should be like obj_000001
    #######
    if len(active_objs) > 21:
        break
    #######

for i, obj in enumerate(active_objs):
    obj.set_origin(mode="CENTER_OF_MASS")
    obj.set_location([2, 2, 2])
    obj.hide(True)

f.write(f"[{time.asctime()}]: loaded objects\n")


def get_length_and_width_from_bbox(bbox: np.ndarray) -> tuple[float, float]:
    # get length(x) and width(y) from bbox(8x3 array)
    length = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
    width = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
    return length, width


def get_base_coordinate_from_bbox(bbox: np.ndarray) -> tuple[float, float, float]:
    return np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.min(bbox[:, 2])


def write_expressions_file(data: dict, file_name: str, append_to_existing_file: bool = True):
    if append_to_existing_file:
        with open(file_name, "r") as f:
            existing_data = json.load(f)
        data = {**existing_data, **data}


def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    # obj.set_location(bproc.sampler.upper_region(
    #     objects_to_sample_on=[surface],
    #     min_height=0.3,
    #     max_height=1,
    #     use_ray_trace_check=False
    # ))
    # obj.set_rotation_euler(np.random.uniform(
    #     [0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))
    global surface
    x0, y0, z0 = get_base_coordinate_from_bbox(surface.get_bound_box())
    l, w = get_length_and_width_from_bbox(surface.get_bound_box())
    i, j = obj.get_cp("coordinate")
    obj.set_location(
        [
            x0 + (2 * i + 1) * l / (2 * scene_graph_rows) + np.random.uniform(-w / (4 * scene_graph_rows), w / (4 * scene_graph_rows)),
            y0 + (2 * j + 1) * w / (2 * scene_graph_cols) + np.random.uniform(-l / (4 * scene_graph_cols), l / (4 * scene_graph_cols)),
            z0 + 1 + np.random.uniform(-0.5, 0.5),
        ]
    )

    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    # transfoem hsv(0-1) to rgb(0-1)
    h = hsv[0] * 6
    s = hsv[1]
    v = hsv[2]
    i = np.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if i == 0:
        return np.array([v, t, p])
    elif i == 1:
        return np.array([q, v, p])
    elif i == 2:
        return np.array([p, v, t])
    elif i == 3:
        return np.array([p, q, v])
    elif i == 4:
        return np.array([t, p, v])
    elif i == 5:
        return np.array([v, p, q])
    else:
        raise ValueError("Invalid h value")


def CheckOcclusion(objs_to_check: list[bproc.types.MeshObject], res_ratio: float, threshold: float = 0.4) -> bool:
    areas_with_occlusion = {}
    blender_objs_to_check = [obj.blender_obj for obj in objs_to_check]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = bpy.context.scene.objects["Camera"]
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=bpy.context.scene)
    camera_quaternion = camera.matrix_world.to_quaternion()
    camera_translation = camera.matrix_world.translation
    x_range = np.linspace(top_left[0], top_right[0], int(resolution_w * res_ratio))
    y_range = np.linspace(top_left[1], bottom_left[1], int(resolution_h * res_ratio))
    z_dir = top_left[2]
    pixel_vectors = []
    for x in x_range:
        line = []
        for y in y_range:
            pixel_vector = Vector((x, y, z_dir))
            pixel_vector.rotate(camera_quaternion)
            pixel_vector.normalize()
            line.append(pixel_vector)
        pixel_vectors.append(line)

    for obj_to_check in blender_objs_to_check:
        areas_with_occlusion[obj_to_check] = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
            if is_hit and hit_obj in blender_objs_to_check:
                areas_with_occlusion[hit_obj] += 1

    for obj_to_check in blender_objs_to_check:
        # hide other objs
        for obj in blender_objs_to_check:
            if obj != obj_to_check:
                obj.hide_set(True)
        area_without_occlusion = 0
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
                if is_hit and hit_obj == obj_to_check:
                    area_without_occlusion += 1
        # show other objs
        for obj in blender_objs_to_check:
            if obj != obj_to_check:
                obj.hide_set(False)
        if area_without_occlusion == 0 or areas_with_occlusion[obj_to_check] / area_without_occlusion < threshold:
            return False
    return True


def CheckInView(points: list[np.ndarray]) -> bool:
    for point in points:
        if not bproc.camera.is_point_inside_camera_frustum(point):
            return False
    return True


skips = 0
i = 0
while i < iterations:
    bproc.utility.reset_keyframes()
    f.write(f"[{time.asctime()}]: iteration {i} started\n")

    while True:
        if s.CreateScene(6) is True:
            break
    # s = SceneGraph.LoadScene("test_scene_graph")

    obj_names = [objectNode.obj_id for objectNode in s.objectNodes]

    selected_objs: list[bproc.types.MeshObject] = []
    for j, obj_name in enumerate(obj_names):
        selected_obj = bproc.filter.one_by_cp(active_objs, "obj_id", obj_name).duplicate()
        selected_obj.set_cp("coordinate", s.objectNodes[j].coordinate)
        selected_obj.set_cp("scene_id", j)
        selected_objs.append(selected_obj)

    for obj in selected_objs:
        obj.enable_rigidbody(True, friction=1)
        obj.hide(False)

    selected_objs = bproc.object.sample_poses_on_surface(
        selected_objs, surface, sample_pose, min_distance=0.001, max_distance=10, check_all_bb_corners_over_surface=False
    )

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)

    # define a light and set its location and energy level
    light.set_location(np.random.uniform([-0.5, -0.5, 0.5], [0.5, 0.5, 1.5]))
    light.set_energy(np.random.uniform(5, 100))
    light.set_color(hsv2rgb(np.random.uniform([0, 0, 1], [1, 0.7, 1])))

    # poi = bproc.object.compute_poi(selected_objs)
    poi = np.array(
        [
            np.mean(
                [
                    np.max(np.array([obj.get_bound_box()[:, 0] for obj in selected_objs])),
                    np.min(np.array([obj.get_bound_box()[:, 0] for obj in selected_objs])),
                ]
            ),
            np.mean(
                [
                    np.max(np.array([obj.get_bound_box()[:, 1] for obj in selected_objs])),
                    np.min(np.array([obj.get_bound_box()[:, 1] for obj in selected_objs])),
                ]
            ),
            np.mean(
                [
                    np.max(np.array([obj.get_bound_box()[:, 2] for obj in selected_objs])),
                    np.min(np.array([obj.get_bound_box()[:, 2] for obj in selected_objs])),
                ]
            ),
        ]
    )

    render_flag = True
    for j in range(images_per_iteration):
        retry = 20
        all_objs_in_view = False
        all_objs_not_occluded = False
        while retry:
            all_objs_in_view = False
            all_objs_not_occluded = False
            # Sample random camera location above objects
            location = np.random.uniform([0.5 + poi[0], -0.1 + poi[1], 0.2], [0.8 + poi[0], 0.1 + poi[1], 0.7])

            # check if all objs in view
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi / 12, np.pi / 12))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix, frame=j)
            # check if all objs in view
            all_objs_in_view = CheckInView([obj.get_location() for obj in selected_objs])
            if all_objs_in_view:
                # check if all objs not occluded
                all_objs_not_occluded = CheckOcclusion(selected_objs, 0.05)
            if all_objs_in_view and all_objs_not_occluded:
                break
            retry -= 1
        if retry == 0:
            render_flag = False
            if not all_objs_in_view:
                print("put all objs in view failed")
                f.write(f"[{time.asctime()}]: WARNING: put all objs in view failed for frame {j}. Skip render.\n")
                break
            elif not all_objs_not_occluded:
                print("check occlusion failed")
                f.write(f"[{time.asctime()}]: WARNING: check occlusion failed for frame {j}. Skip render.\n")
                break
            else:
                print("unknown error")
                f.write(f"[{time.asctime()}]: WARNING: unknown error. Skip render.\n")
                break

    if not render_flag:
        for obj in selected_objs:
            obj.delete()
        skips += 1
        print(f"no suitable camera pose found, skip iteration {i}")
        f.write(f"[{time.asctime()}]: no suitable camera pose found, skip iteration {i}\n")
        continue
    print(f"iteration {i} started rendering")
    f.write(f"[{time.asctime()}]: iteration {i} started rendering\n")

    scene_graph_idx = SceneGraph.WriteSceneGraphToFile(s, os.path.join(output_dir, "scene_graphs"), "scene_graph")
    f.write(f"[{time.asctime()}]: wrote scene graph: {scene_graph_idx}\n")

    bproc.camera.set_resolution(resolution_w, resolution_h)

    # activate normal and depth rendering
    # bproc.renderer.enable_normals_output()

    bproc.renderer.enable_segmentation_output(map_by=["scene_id"])
    # render the whole pipeline
    data = bproc.renderer.render()
    data["depth"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])
    f.write(f"[{time.asctime()}]: rendered {images_per_iteration} images\n")
    for obj in selected_objs:
        obj.delete()

    # write expressions and images
    depth_idxs = DatasetUtils.WriteImage(data, os.path.join(output_dir, "depth"), "depth", file_name_prefix="depth", append_to_exsiting_file=True)
    print(f"write depth images: {depth_idxs}")
    f.write(f"[{time.asctime()}]: wrote depth images: {depth_idxs}\n")
    colors_idxs = DatasetUtils.WriteImage(data, os.path.join(output_dir, "rgb"), "colors", file_name_prefix="rgb", append_to_exsiting_file=True)
    print(f"write rgb images: {colors_idxs}")
    f.write(f"[{time.asctime()}]: wrote rgb images: {colors_idxs}\n")
    segmaps_idxs = DatasetUtils.WriteImage(
        data, os.path.join(output_dir, "mask"), "scene_id_segmaps", file_name_prefix="mask", append_to_exsiting_file=True
    )
    print(f"write mask images: {segmaps_idxs}")
    f.write(f"[{time.asctime()}]: wrote mask images: {segmaps_idxs}\n")

    expressions = []
    for j, expression in enumerate(s.GetComplexReferringExpressions()):
        node = list(s.referringExpressionStructures.keys())[j]
        # get id(int) from obj_id(str) like obj_000001
        obj_id = int(node.obj_id.split("_")[1])
        expressions.append({"obj": {"id": obj_id, "scene_id": node.scene_id}, "expression": expression})

    expressions_idx = DatasetUtils.WriteExpressions(
        segmaps_idxs, expressions, save_path=os.path.join(output_dir, "temp"), file_name_prefix="expressions"
    )
    print(f"write expressions: {expressions_idx}")
    f.write(f"[{time.asctime()}]: wrote expressions json: {expressions_idx}\n")

    if not DatasetUtils.CheckImageFileNums(output_dir, ["depth", "rgb", "mask"]):
        f.write(f"[{time.asctime()}]: ERROR: image file numbers not match\n")
        raise Exception("image file numbers not match. see output_log.txt for details")

    i += 1

num_expressions = DatasetUtils.MergeExpressions(output_dir, os.path.join(output_dir, "temp"))
f.write(f"[{time.asctime()}]: merged expressions jsons\n")
if num_expressions != scene_graph_idx + 1:
    f.write(
        f"[{time.asctime()}]: ERROR: number of expressions jsons ({num_expressions}) and number of scene graphs ({scene_graph_idx+1}) not match\n"
    )
    raise Exception("number of expressions not match. see output_log.txt for details")
print(f"process finished with {iterations} successful iterations, {skips} skipped. see output_log.txt for details")
f.write(f"[{time.asctime()}]: process finished with {iterations} successful iterations, {skips} skipped\n")
f.close()
