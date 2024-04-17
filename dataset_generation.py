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

# set parameters
OBJ_DIR = "./models/ycb_models"  # obj files directory
BLENDER_SCENE_FILE_PATH = "./blender_files/background.blend"  # background scene file path
OUTPUT_DIR = "./output/temp5"
MODELS_INFO_FILE_PATH = "./models/models_info_test.xlsx"
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "./dataset_generation.log")
ITERATIONS = 5
IMAGES_PER_ITERATION = 5
RESOLUTION_WIDTH = 512
RESOLUTION_HEIGHT = 512
SCENE_GRAPH_ROWS = 4
SCENE_GRAPH_COLS = 4
SEED = None

# create logging directory
if not os.path.exists(os.path.dirname(LOG_FILE_PATH)):
    os.makedirs(os.path.dirname(LOG_FILE_PATH))
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


def sample_pose(obj: bproc.types.MeshObject, surface: bproc.types.MeshObject):
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


def background_scene_init(entities: list[bproc.types.Entity]) -> tuple[list[bproc.types.MeshObject], bproc.types.Light]:
    """
    Initialize the background scene with the background entities. Set up lights, render settings, etc.

    Args:
        entities (list[bproc.types.Entity]): background entities.

    Returns:
        tuple[list[bproc.types.MeshObject], bproc.types.Light]: a tuple of (surfaces list, light).
    """
    # get surfaces on which objects will be placed
    surfaces: list[bproc.types.MeshObject] = bproc.filter.by_attr(entities, "name", "^surface.*", regex=True)
    for surface in surfaces:
        surface.enable_rigidbody(True, friction=1)
        surface.blender_obj.hide_render = True

    # set up background
    for entity in entities:
        entity.set_cp("obj_id", None)
        entity.set_cp("scene_id", 255)
        # hide background objects in viewport to speed up ray_cast
        entity.blender_obj.hide_set(True)
        if entity not in surfaces and isinstance(entity, bproc.types.MeshObject):
            # disable rigidbody for background objects
            entity.disable_rigidbody() if entity.has_rigidbody_enabled() else None

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
    bproc.renderer.set_max_amount_of_samples(128)
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
    bpy.context.scene.render.use_persistent_data = True

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
        obj.blender_obj.hide_render = True
        obj.blender_obj.hide_set(True)

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
        selected_obj.blender_obj.hide_render = False
        selected_obj.blender_obj.hide_set(False)
        selected_obj.enable_rigidbody(True, friction=1)
        selected_objs.append(selected_obj)

    surface: bproc.types.MeshObject = np.random.choice(surfaces)
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


def construct_scene_for_single_image(
    surface: bproc.types.MeshObject,
    light: bproc.types.Light,
    selected_objs: list[bproc.types.MeshObject],
    background_entities: list[bproc.types.Entity],
) -> bool:
    # sample camera pose
    retry = 50
    not_in_view_times = 0
    occluded_times = 0
    while retry:
        camera_location = BlenderUtils.sample_point_in_cuboid(
            BlenderUtils.add_relative_translation_to_matrix(np.array(surface.blender_obj.matrix_world), [0.45, 0, 0.6]), np.array([0.3, 0.3, 0.8])
        )
        camera_rotation = bproc.camera.rotation_from_forward_vec(
            BlenderUtils.poi(selected_objs) - camera_location, inplane_rot=np.random.uniform(-np.pi / 12, np.pi / 12)
        )
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(camera_location, camera_rotation), frame=0)
        # check if all objects are in view
        # all_objs_in_view = BlenderUtils.check_in_view([obj.get_location() for obj in selected_objs])
        if not BlenderUtils.check_points_in_view([obj.get_location() for obj in selected_objs]):
            retry -= 1
            not_in_view_times += 1
            continue
        # check if all objects are not occluded
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # rough check
        for entity in background_entities:
            entity.blender_obj.hide_set(False)
        all_objs_not_occluded_rough = BlenderUtils.check_occlusion_rough(
            [selected_obj.blender_obj for selected_obj in selected_objs], depsgraph=depsgraph
        )
        for entity in background_entities:
            entity.blender_obj.hide_set(True)
        if not all_objs_not_occluded_rough:
            retry -= 1
            occluded_times += 1
            continue
        # fine check
        all_objs_not_occluded = BlenderUtils.check_occlusion(
            [selected_obj.blender_obj for selected_obj in selected_objs], 0.1, threshold=0.4, depsgraph=depsgraph
        )
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
        BlenderUtils.add_relative_translation_to_matrix(np.array(surface.blender_obj.matrix_world), [0.5, 0, 1]), np.array([1.6, 2, 0])
    )
    light.set_location(light_location)
    light.set_energy(np.random.uniform(10, 150))
    light.set_color(BlenderUtils.hsv2rgb(np.random.uniform([0, 0, 1], [1, 0.7, 1])))

    # set background
    set_random_background(["background_floor", "background_wall"])

    return True


if __name__ == "__main__":
    logging.info("Started")
    logging.info(
        f"Parameters: OBJ_DIR={os.path.abspath(OBJ_DIR)}, BLENDER_SCENE_FILE_PATH={os.path.abspath(BLENDER_SCENE_FILE_PATH)}, OUTPUT_DIR={os.path.abspath(OUTPUT_DIR)}, "
        f"MODELS_INFO_FILE_PATH={os.path.abspath(MODELS_INFO_FILE_PATH)}, LOG_FILE_PATH={os.path.abspath(LOG_FILE_PATH)}, ITERATIONS={ITERATIONS}, "
        f"IMAGES_PER_ITERATION={IMAGES_PER_ITERATION}, RESOLUTION_WIDTH={RESOLUTION_WIDTH}, RESOLUTION_HEIGHT={RESOLUTION_HEIGHT}, "
        f"SCENE_GRAPH_ROWS={SCENE_GRAPH_ROWS}, SCENE_GRAPH_COLS={SCENE_GRAPH_COLS}, SEED={SEED}"
    )
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
            render_flag = construct_scene_for_single_image(surface, light, placed_objects, background_entities)
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
            image_indices += depth_idxs

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
