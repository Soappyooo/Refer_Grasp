import blenderproc as bproc
import numpy as np
import os
import sys
import time
import bpy
from mathutils import Vector
import logging
import random

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# for debugging
sys.path.append(os.getcwd())
###

from scene_graph_generation import SceneGraph
from utils.dataset_utils import DatasetUtils
from utils.blender_utils import BlenderUtils

OBJ_DIR = "./models/ycb_models"  # obj files directory
BLENDER_SCENE_FILE_PATH = "./blender_files/background.blend"  # background scene file path
OUTPUT_DIR = "./output/temp"
MODELS_INFO_FILE_PATH = "./models/models_info_test.xlsx"
LOG_FILE_PATH = "./dataset_generation.log"
ITERATIONS = 5
IMAGES_PER_ITERATION = 5
RESOLUTION_WIDTH = 512
RESOLUTION_HEIGHT = 512
SCENE_GRAPH_ROWS = 4
SCENE_GRAPH_COLS = 4
SEED = 1713104701

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s - line %(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=LOG_FILE_PATH,
    filemode="w",
)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - [%(filename)s - line %(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(stream_handler)


# bproc.init()
# # f = open(output_log, "w")
# # f.write(f"[{time.asctime()}]: started\n")
# logging.info("started")

# # check if output_dir has files
# for _, _, files in os.walk(output_dir):
#     if files != []:
#         # f.write(f"[{time.asctime()}]: WARNING: output_dir ({output_dir}) is not empty\n")
#         logging.warning(f"output_dir ({output_dir}) is not empty")
#         print(f"WARNING: output_dir ({output_dir}) is not empty, continue? (y/n)")
#         if input() != "y":
#             # f.write(f"[{time.asctime()}]: stopped because output_dir ({output_dir}) is not empty\n")
#             logging.warning(f"stopped because output_dir ({output_dir}) is not empty")
#             print("stopped because output_dir is not empty")
#             sys.exit()
#         else:
#             break

# s = SceneGraph((scene_graph_rows, scene_graph_cols))
# s.LoadModelsInfo(models_info_file)
# print("loading background...")
# # load background into the scene
# background_objs = bproc.loader.load_blend(os.path.join(background_dir, background_file))
# for obj in background_objs:
#     obj.set_cp("obj_id", None)
#     obj.set_cp("scene_id", 255)
#     if type(obj) == bproc.types.MeshObject:
#         obj.disable_rigidbody()

# surface: bproc.types.MeshObject = bproc.filter.one_by_attr(background_objs, "name", "surface")
# surface.hide(True)
# surface.enable_rigidbody(True, friction=1)
# # walls: list[bproc.types.MeshObject] = bproc.filter.by_attr(background_objs, "name", "^plane(?!$).*$", regex=True)
# # for wall in walls:
# #     wall.enable_rigidbody(False, friction=1)
# # floor: bproc.types.MeshObject = bproc.filter.one_by_attr(background_objs, "name", "floor")
# # floor.enable_rigidbody(False, friction=1)

# light = bproc.types.Light()
# light.set_type("POINT")

# bproc.renderer.enable_depth_output(activate_antialiasing=False)
# bproc.renderer.set_max_amount_of_samples(16)

# active_objs: list[bproc.types.MeshObject] = []
# print("loading objects...")

# for file_name in os.listdir(obj_dir):
#     if file_name.endswith(".obj"):
#         obj = bproc.loader.load_obj(os.path.join(obj_dir, file_name), use_legacy_obj_import=True)[0]
#         active_objs.append(obj)
#         obj.set_cp("obj_id", file_name.split(".")[0])  # should be like obj_000001
#     #######
#     if len(active_objs) > 5:
#         break
#     #######

# for i, obj in enumerate(active_objs):
#     obj.set_origin(mode="CENTER_OF_MASS")
#     obj.set_location([2, 2, 2])
#     obj.hide(True)

# # f.write(f"[{time.asctime()}]: loaded objects\n")
# logging.info("loaded objects")


# def get_length_and_width_from_bbox(bbox: np.ndarray) -> tuple[float, float]:
#     # get length(x) and width(y) from bbox(8x3 array)
#     length = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
#     width = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
#     return length, width


# def get_base_coordinate_from_bbox(bbox: np.ndarray) -> tuple[float, float, float]:
#     return np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.min(bbox[:, 2])


# def write_expressions_file(data: dict, file_name: str, append_to_existing_file: bool = True):
#     if append_to_existing_file:
#         with open(file_name, "r") as f:
#             existing_data = json.load(f)
#         data = {**existing_data, **data}


def sample_pose(obj: bproc.types.MeshObject, surface: bproc.types.MeshObject):
    # x0, y0, z0 = get_base_coordinate_from_bbox(surface.get_bound_box())
    # l, w = get_length_and_width_from_bbox(surface.get_bound_box())
    # i, j = obj.get_cp("coordinate")
    # obj.set_location(
    #     [
    #         x0 + (2 * i + 1) * l / (2 * scene_graph_rows) + np.random.uniform(-w / (4 * scene_graph_rows), w / (4 * scene_graph_rows)),
    #         y0 + (2 * j + 1) * w / (2 * scene_graph_cols) + np.random.uniform(-l / (4 * scene_graph_cols), l / (4 * scene_graph_cols)),
    #         z0 + 1 + np.random.uniform(-0.5, 0.5),
    #     ]
    # )

    surface_bbox_cords = BlenderUtils.get_bound_box_local_coordinates(surface.blender_obj)
    x0, y0, z0 = surface_bbox_cords[0]
    l, w, _ = surface.blender_obj.dimensions
    obj_scene_graph_cord_x, obj_scene_graph_cord_y = obj.get_cp("coordinate")
    obj_location_random = surface.blender_obj.matrix_world @ Vector(
        [
            x0
            + (2 * obj_scene_graph_cord_x + 1) * l / (2 * SCENE_GRAPH_ROWS)
            + np.random.uniform(-w / (4 * SCENE_GRAPH_ROWS), w / (4 * SCENE_GRAPH_ROWS)),
            y0
            + (2 * obj_scene_graph_cord_y + 1) * w / (2 * SCENE_GRAPH_COLS)
            + np.random.uniform(-l / (4 * SCENE_GRAPH_COLS), l / (4 * SCENE_GRAPH_COLS)),
            z0 + 1,
        ]
    )
    obj.set_location(obj_location_random)
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))


# def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
#     # transfoem hsv(0-1) to rgb(0-1)
#     h = hsv[0] * 6
#     s = hsv[1]
#     v = hsv[2]
#     i = np.floor(h)
#     f = h - i
#     p = v * (1 - s)
#     q = v * (1 - s * f)
#     t = v * (1 - s * (1 - f))
#     if i == 0:
#         return np.array([v, t, p])
#     elif i == 1:
#         return np.array([q, v, p])
#     elif i == 2:
#         return np.array([p, v, t])
#     elif i == 3:
#         return np.array([p, q, v])
#     elif i == 4:
#         return np.array([t, p, v])
#     elif i == 5:
#         return np.array([v, p, q])
#     else:
#         raise ValueError("Invalid h value")


# def CheckOcclusion(objs_to_check: list[bproc.types.MeshObject], res_ratio: float, threshold: float = 0.4) -> bool:
#     areas_with_occlusion = {}
#     blender_objs_to_check = [obj.blender_obj for obj in objs_to_check]
#     depsgraph = bpy.context.evaluated_depsgraph_get()
#     camera = bpy.context.scene.objects["Camera"]
#     top_right, _, bottom_left, top_left = camera.data.view_frame(scene=bpy.context.scene)
#     camera_quaternion = camera.matrix_world.to_quaternion()
#     camera_translation = camera.matrix_world.translation
#     x_range = np.linspace(top_left[0], top_right[0], int(resolution_w * res_ratio))
#     y_range = np.linspace(top_left[1], bottom_left[1], int(resolution_h * res_ratio))
#     z_dir = top_left[2]
#     pixel_vectors = []
#     for x in x_range:
#         line = []
#         for y in y_range:
#             pixel_vector = Vector((x, y, z_dir))
#             pixel_vector.rotate(camera_quaternion)
#             pixel_vector.normalize()
#             line.append(pixel_vector)
#         pixel_vectors.append(line)

#     for obj_to_check in blender_objs_to_check:
#         areas_with_occlusion[obj_to_check] = 0
#     for i, x in enumerate(x_range):
#         for j, y in enumerate(y_range):
#             is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
#             if is_hit and hit_obj in blender_objs_to_check:
#                 areas_with_occlusion[hit_obj] += 1

#     for obj_to_check in blender_objs_to_check:
#         # hide other objs
#         for obj in blender_objs_to_check:
#             if obj != obj_to_check:
#                 obj.hide_set(True)
#         area_without_occlusion = 0
#         for i, x in enumerate(x_range):
#             for j, y in enumerate(y_range):
#                 is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
#                 if is_hit and hit_obj == obj_to_check:
#                     area_without_occlusion += 1
#         # show other objs
#         for obj in blender_objs_to_check:
#             if obj != obj_to_check:
#                 obj.hide_set(False)
#         if area_without_occlusion == 0 or areas_with_occlusion[obj_to_check] / area_without_occlusion < threshold:
#             return False
#     return True


# def CheckInView(points: list[np.ndarray]) -> bool:
#     for point in points:
#         if not bproc.camera.is_point_inside_camera_frustum(point):
#             return False
#     return True


# skips = 0
# i = 0
# while i < iterations:
#     bproc.utility.reset_keyframes()
#     # f.write(f"[{time.asctime()}]: iteration {i} started\n")
#     logging.info(f"iteration {i} started")

#     while True:
#         if s.CreateScene(6) is True:
#             break
#     # s = SceneGraph.LoadScene("test_scene_graph")

#     obj_names = [objectNode.obj_id for objectNode in s.objectNodes]

#     selected_objs: list[bproc.types.MeshObject] = []
#     for j, obj_name in enumerate(obj_names):
#         selected_obj = bproc.filter.one_by_cp(active_objs, "obj_id", obj_name).duplicate()
#         selected_obj.set_cp("coordinate", s.objectNodes[j].coordinate)
#         selected_obj.set_cp("scene_id", j)
#         selected_objs.append(selected_obj)

#     for obj in selected_objs:
#         obj.enable_rigidbody(True, friction=1)
#         obj.hide(False)

#     selected_objs = bproc.object.sample_poses_on_surface(
#         selected_objs, surface, sample_pose, min_distance=0.001, max_distance=10, check_all_bb_corners_over_surface=False
#     )

#     bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)

#     # define a light and set its location and energy level
#     light.set_location(np.random.uniform([-0.5, -0.5, 0.5], [0.5, 0.5, 1.5]))
#     light.set_energy(np.random.uniform(5, 100))
#     light.set_color(hsv2rgb(np.random.uniform([0, 0, 1], [1, 0.7, 1])))

#     # poi = bproc.object.compute_poi(selected_objs)
#     poi = np.array(
#         [
#             np.mean(
#                 [
#                     np.max(np.array([obj.get_bound_box()[:, 0] for obj in selected_objs])),
#                     np.min(np.array([obj.get_bound_box()[:, 0] for obj in selected_objs])),
#                 ]
#             ),
#             np.mean(
#                 [
#                     np.max(np.array([obj.get_bound_box()[:, 1] for obj in selected_objs])),
#                     np.min(np.array([obj.get_bound_box()[:, 1] for obj in selected_objs])),
#                 ]
#             ),
#             np.mean(
#                 [
#                     np.max(np.array([obj.get_bound_box()[:, 2] for obj in selected_objs])),
#                     np.min(np.array([obj.get_bound_box()[:, 2] for obj in selected_objs])),
#                 ]
#             ),
#         ]
#     )

#     render_flag = True
#     for j in range(images_per_iteration):
#         retry = 20
#         all_objs_in_view = False
#         all_objs_not_occluded = False
#         while retry:
#             all_objs_in_view = False
#             all_objs_not_occluded = False
#             # Sample random camera location above objects
#             location = np.random.uniform([0.5 + poi[0], -0.1 + poi[1], 0.2], [0.8 + poi[0], 0.1 + poi[1], 0.7])

#             # check if all objs in view
#             # Compute rotation based on vector going from location towards poi
#             rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi / 12, np.pi / 12))
#             # Add homog cam pose based on location an rotation
#             cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
#             bproc.camera.add_camera_pose(cam2world_matrix, frame=j)
#             # check if all objs in view
#             all_objs_in_view = CheckInView([obj.get_location() for obj in selected_objs])
#             if all_objs_in_view:
#                 # check if all objs not occluded
#                 all_objs_not_occluded = CheckOcclusion(selected_objs, 0.05)
#             if all_objs_in_view and all_objs_not_occluded:
#                 break
#             retry -= 1
#         if retry == 0:
#             render_flag = False
#             if not all_objs_in_view:
#                 print("put all objs in view failed")
#                 # f.write(f"[{time.asctime()}]: WARNING: put all objs in view failed for frame {j}. Skip render.\n")
#                 logging.warning(f"put all objs in view failed for frame {j}. Skip render.")
#                 break
#             elif not all_objs_not_occluded:
#                 print("check occlusion failed")
#                 # f.write(f"[{time.asctime()}]: WARNING: check occlusion failed for frame {j}. Skip render.\n")
#                 logging.warning(f"check occlusion failed for frame {j}. Skip render.")
#                 break
#             else:
#                 print("unknown error")
#                 # f.write(f"[{time.asctime()}]: WARNING: unknown error. Skip render.\n")
#                 logging.warning("unknown error. Skip render.")
#                 break

#     if not render_flag:
#         for obj in selected_objs:
#             obj.delete()
#         skips += 1
#         print(f"no suitable camera pose found, skip iteration {i}")
#         # f.write(f"[{time.asctime()}]: no suitable camera pose found, skip iteration {i}\n")
#         logging.info(f"no suitable camera pose found, skip iteration {i}")
#         continue
#     print(f"iteration {i} started rendering")
#     # f.write(f"[{time.asctime()}]: iteration {i} started rendering\n")
#     logging.info(f"iteration {i} started rendering")

#     scene_graph_idx = SceneGraph.WriteSceneGraphToFile(s, os.path.join(output_dir, "scene_graphs"), "scene_graph")
#     # f.write(f"[{time.asctime()}]: wrote scene graph: {scene_graph_idx}\n")
#     logging.info(f"wrote scene graph: {scene_graph_idx}")

#     bproc.camera.set_resolution(resolution_w, resolution_h)

#     # activate normal and depth rendering
#     # bproc.renderer.enable_normals_output()

#     bproc.renderer.enable_segmentation_output(map_by=["scene_id"])
#     # render the whole pipeline
#     data = bproc.renderer.render()
#     data["depth"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])
#     # f.write(f"[{time.asctime()}]: rendered {images_per_iteration} images\n")
#     logging.info(f"rendered {images_per_iteration} images")
#     for obj in selected_objs:
#         obj.delete()

#     # write expressions and images
#     depth_idxs = DatasetUtils.write_image(data, os.path.join(output_dir, "depth"), "depth", file_name_prefix="depth", append_to_exsiting_file=True)
#     print(f"write depth images: {depth_idxs}")
#     # f.write(f"[{time.asctime()}]: wrote depth images: {depth_idxs}\n")
#     logging.info(f"wrote depth images: {depth_idxs}")
#     colors_idxs = DatasetUtils.write_image(data, os.path.join(output_dir, "rgb"), "colors", file_name_prefix="rgb", append_to_exsiting_file=True)
#     print(f"write rgb images: {colors_idxs}")
#     # f.write(f"[{time.asctime()}]: wrote rgb images: {colors_idxs}\n")
#     logging.info(f"wrote rgb images: {colors_idxs}")
#     segmaps_idxs = DatasetUtils.write_image(
#         data, os.path.join(output_dir, "mask"), "scene_id_segmaps", file_name_prefix="mask", append_to_exsiting_file=True
#     )
#     print(f"write mask images: {segmaps_idxs}")
#     # f.write(f"[{time.asctime()}]: wrote mask images: {segmaps_idxs}\n")
#     logging.info(f"wrote mask images: {segmaps_idxs}")

#     expressions = []
#     for j, expression in enumerate(s.GetComplexReferringExpressions()):
#         node = list(s.referringExpressionStructures.keys())[j]
#         # get id(int) from obj_id(str) like obj_000001
#         obj_id = int(node.obj_id.split("_")[1])
#         expressions.append({"obj": {"id": obj_id, "scene_id": node.scene_id}, "expression": expression})

#     expressions_idx = DatasetUtils.write_expressions(
#         segmaps_idxs, expressions, save_path=os.path.join(output_dir, "temp"), file_name_prefix="expressions"
#     )
#     print(f"write expressions: {expressions_idx}")
#     # f.write(f"[{time.asctime()}]: wrote expressions json: {expressions_idx}\n")
#     logging.info(f"wrote expressions json: {expressions_idx}")

#     if not DatasetUtils.check_image_file_nums(output_dir, ["depth", "rgb", "mask"]):
#         # f.write(f"[{time.asctime()}]: ERROR: image file numbers not match\n")
#         logging.error("image file numbers not match")
#         raise Exception("image file numbers not match. see output_log.txt for details")

#     i += 1

# num_expressions = DatasetUtils.merge_expressions(output_dir, os.path.join(output_dir, "temp"))
# # f.write(f"[{time.asctime()}]: merged expressions jsons\n")
# logging.info("merged expressions jsons")
# if num_expressions != scene_graph_idx + 1:
#     # f.write(
#     #     f"[{time.asctime()}]: ERROR: number of expressions jsons ({num_expressions}) and number of scene graphs ({scene_graph_idx+1}) not match\n"
#     # )
#     logging.error(f"number of expressions jsons ({num_expressions}) and number of scene graphs ({scene_graph_idx+1}) not match")
#     raise Exception("number of expressions not match. see output_log.txt for details")
# print(f"process finished with {iterations} successful iterations, {skips} skipped. see output_log.txt for details")
# # f.write(f"[{time.asctime()}]: process finished with {iterations} successful iterations, {skips} skipped\n")
# logging.info(f"process finished with {iterations} successful iterations, {skips} skipped")
# # f.close() #


def background_scene_init(entities: list[bproc.types.Entity]) -> tuple[list[bproc.types.MeshObject], bproc.types.Light]:
    """
    Initialize the background scene with the background entities. Set up lights, render settings, etc.

    Args:
        entities (list[bproc.types.Entity]): background entities.

    Returns:
        tuple[list[bproc.types.MeshObject], bproc.types.Light]: a tuple of (surfaces list, light).
    """
    # set up background
    for entity in entities:
        entity.set_cp("obj_id", None)
        entity.set_cp("scene_id", 255)
        if type(entity) == bproc.types.MeshObject and entity.has_rigidbody_enabled():
            entity.disable_rigidbody()

    # get surfaces on which objects will be placed
    surfaces: list[bproc.types.MeshObject] = bproc.filter.by_attr(entities, "name", "^surface.*", regex=True)
    for surface in surfaces:
        surface.hide(True)
        surface.enable_rigidbody(True, friction=1)

    # set up area light
    area_light_data = bpy.data.lights.new(name="area_light", type="AREA")
    area_light_data.energy = 50
    area_light_data.size = 20
    area_light_obj = bpy.data.objects.new(name="area_light", object_data=area_light_data)
    area_light_obj.location = (0, 0, 3)
    bpy.context.collection.objects.link(area_light_obj)

    # set up point light
    point_light = bproc.types.Light(light_type="POINT", name="point_light")

    # set up render settings
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(64)
    # full global illumination
    bpy.context.scene.cycles.max_bounces = 32
    bpy.context.scene.cycles.diffuse_bounces = 32
    bpy.context.scene.cycles.glossy_bounces = 32
    bpy.context.scene.cycles.transmission_bounces = 32
    bpy.context.scene.cycles.volume_bounces = 32
    bpy.context.scene.cycles.transparent_max_bounces = 32
    # 2k textures
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.cycles.texture_limit_render = "2048"
    # persistent data
    bpy.context.scene.render.use_persistent_data = False

    return surfaces, point_light


def load_objects(obj_dir: str) -> list[bproc.types.MeshObject]:
    active_objs: list[bproc.types.MeshObject] = []
    for file_name in os.listdir(obj_dir):
        if file_name.endswith(".obj"):
            obj = bproc.loader.load_obj(os.path.join(obj_dir, file_name), use_legacy_obj_import=True)[0]
            active_objs.append(obj)
            obj.set_cp("obj_id", file_name.split(".")[0])  # should be like obj_000001
        #######
        if len(active_objs) > 5:
            break
        #######

    for i, obj in enumerate(active_objs):
        obj.set_origin(mode="CENTER_OF_MASS")
        obj.set_location([0, 0, 5])
        obj.hide(True)

    return active_objs


def set_random_background(background_obj_names: list[str]):
    for obj_name in background_obj_names:
        BlenderUtils.set_random_material(bpy.data.objects[obj_name])


def sample_objects(scene_graph: SceneGraph, active_objs: list[bproc.types.MeshObject], surfaces: list[bproc.types.MeshObject]):
    while scene_graph.CreateScene(6) is False:
        continue
    obj_names = [objectNode.obj_id for objectNode in scene_graph.objectNodes]
    selected_objs: list[bproc.types.MeshObject] = []
    # copy objects in scene graph
    for j, obj_name in enumerate(obj_names):
        selected_obj: bproc.types.MeshObject = bproc.filter.one_by_cp(active_objs, "obj_id", obj_name).duplicate()
        selected_obj.set_cp("coordinate", scene_graph.objectNodes[j].coordinate)
        selected_obj.set_cp("scene_id", j)
        selected_obj.enable_rigidbody(True, friction=1)
        selected_obj.hide(False)
        selected_objs.append(selected_obj)

    surface: bproc.types.MeshObject = np.random.choice(surfaces)
    # TODO: seed not working
    sampled_objs = bproc.object.sample_poses_on_surface(
        selected_objs, surface, sample_pose, min_distance=0.001, max_distance=10, check_all_bb_corners_over_surface=True
    )
    if not len(sampled_objs) == len(selected_objs):
        logging.warning("Some objects cannot be placed in scene, skip this scene")
        for obj in selected_objs:
            obj.delete(remove_all_offspring=True)
        return [], None
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)
    return sampled_objs, surface


def construct_scene_for_single_image(surface: bproc.types.MeshObject, light: bproc.types.Light, selected_objs: list[bproc.types.MeshObject]) -> bool:
    # sample camera pose
    retry = 50
    not_in_view_times = 0
    occluded_times = 0
    while retry:
        camera_location = BlenderUtils.sample_point_in_cuboid(
            BlenderUtils.add_relative_translation_to_matrix(np.array(surface.blender_obj.matrix_world), [1, 0, 1]), np.array([0.5, 0.2, 1])
        )
        camera_rotation = bproc.camera.rotation_from_forward_vec(
            BlenderUtils.poi(selected_objs) - camera_location, inplane_rot=np.random.uniform(-np.pi / 12, np.pi / 12)
        )
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(camera_location, camera_rotation), frame=0)
        # check if all objects are in view
        all_objs_in_view = BlenderUtils.check_in_view([obj.get_location() for obj in selected_objs])
        if not all_objs_in_view:
            retry -= 1
            not_in_view_times += 1
            continue
        # check if all objects are not occluded
        all_objs_not_occluded = BlenderUtils.check_occlusion(selected_objs, 0.05, threshold=0.5)
        if not all_objs_not_occluded:
            retry -= 1
            occluded_times += 1
            continue
        break
    if retry == 0:
        logging.warning(f"No suitable camera pose found ({not_in_view_times} retries out of view, {occluded_times} retries occluded), skip render")
        return False
    else:
        logging.info(f"Camera pose found after {50 - retry} retries ({not_in_view_times} retries out of view, {occluded_times} retries occluded)")

    # set light
    light_location = BlenderUtils.sample_point_in_cuboid(
        BlenderUtils.add_relative_translation_to_matrix(np.array(surface.blender_obj.matrix_world), [0.5, 0, 1]), np.array([0.8, 1, 0])
    )
    light.set_location(light_location)
    light.set_energy(np.random.uniform(10, 150))
    light.set_color(BlenderUtils.hsv2rgb(np.random.uniform([0, 0, 1], [1, 0.7, 1])))

    # set background
    set_random_background(["background_floor", "background_wall"])

    return True


if __name__ == "__main__":
    logging.info("Started")
    start_time = time.time()
    # * Check file nums in output_dir
    if not DatasetUtils.check_image_file_nums(OUTPUT_DIR, ["depth", "rgb", "mask"]):
        logging.error("Image file numbers not match")
        raise Exception("Image file numbers not match")

    # * Initialize blenderproc & load scene
    bproc.init()
    background_entities = bproc.loader.load_blend(BLENDER_SCENE_FILE_PATH)
    surfaces, light = background_scene_init(background_entities)
    logging.info("Loaded background scene")

    # * Load objects
    active_objs = load_objects(OBJ_DIR)
    logging.info("Loaded objects")

    # * Initialize scene graph
    scene_graph = SceneGraph((SCENE_GRAPH_ROWS, SCENE_GRAPH_COLS))
    scene_graph.LoadModelsInfo(MODELS_INFO_FILE_PATH)
    logging.info("Scene graph initialized")

    # * Dataset generation
    i = 0
    while i < ITERATIONS:
        random_seed = int(time.time()) if SEED is None else SEED
        np.random.seed(random_seed)
        random.seed(random_seed)
        logging.info(f"Iteration {i} started with random seed {random_seed}")
        # place objects in scene
        placed_objects, surface = sample_objects(scene_graph, active_objs, surfaces)
        if not placed_objects:
            logging.warning(f"Cannot place some objects in scene for iteration {i}, restart iteration")
            continue
        logging.info("Placed objects in scene")
        # sample environment and render images
        image_indices = []
        for image_index in range(IMAGES_PER_ITERATION):
            bproc.utility.reset_keyframes()
            # sample camera pose, light and background material
            render_flag = construct_scene_for_single_image(surface, light, placed_objects)
            if not render_flag:
                break
            logging.info(f"Constructed scene for image ({image_index + 1}/{IMAGES_PER_ITERATION})")
            # render image
            bproc.renderer.enable_segmentation_output(map_by=["scene_id"], default_values={"scene_id": 255})
            data = bproc.renderer.render()
            data["depth"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])
            logging.info(f"Rendered image ({image_index + 1}/{IMAGES_PER_ITERATION})")
            # write image
            depth_idxs = DatasetUtils.write_image(
                data, os.path.join(OUTPUT_DIR, "depth"), "depth", file_name_prefix="depth", append_to_exsiting_file=True
            )
            logging.info(f"Wrote depth images: {[f'depth_{idx:08d}.png' for idx in depth_idxs]}")
            colors_idxs = DatasetUtils.write_image(
                data, os.path.join(OUTPUT_DIR, "rgb"), "colors", file_name_prefix="rgb", append_to_exsiting_file=True
            )
            logging.info(f"Wrote rgb images: {[f'rgb_{idx:08d}.png' for idx in colors_idxs]}")
            segmaps_idxs = DatasetUtils.write_image(
                data, os.path.join(OUTPUT_DIR, "mask"), "scene_id_segmaps", file_name_prefix="mask", append_to_exsiting_file=True
            )
            logging.info(f"Wrote mask images: {[f'mask_{idx:08d}.png' for idx in segmaps_idxs]}")
            assert depth_idxs == colors_idxs == segmaps_idxs
            image_indices.append(depth_idxs)

        if len(image_indices) > 0:
            logging.info(f"Iteration {i} finished with {len(image_indices)} images rendered")
            # write expressions
            expressions = []
            for j, expression in enumerate(scene_graph.GetComplexReferringExpressions()):
                node = list(scene_graph.referringExpressionStructures.keys())[j]
                obj_id = int(node.obj_id.split("_")[1])
                expressions.append({"obj": {"id": obj_id, "scene_id": node.scene_id}, "expression": expression})
            expressions_idx = DatasetUtils.write_expressions(
                image_indices, expressions, save_path=os.path.join(OUTPUT_DIR, "expressions"), file_name_prefix="expressions"
            )
            logging.info(f"Wrote expressions json: expressions_{expressions_idx:08d}.json")
            i += 1
        else:
            logging.warning(f"No images rendered for iteration {i}, restart iteration")
        # clean up
        for obj in placed_objects:
            obj.delete(remove_all_offspring=True)
    logging.info("Dataset generation finished")

    # * Merge expressions
    num_expressions = DatasetUtils.merge_expressions(os.path.join(OUTPUT_DIR), os.path.join(OUTPUT_DIR, "expressions"))
    logging.info(f"Merged expressions jsons ({num_expressions} scenes)")
    # calculate time H:M:S
    logging.info(f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    logging.info(f"Process finished with {i} successful iterations, see {os.path.abspath(LOG_FILE_PATH)} for details")
