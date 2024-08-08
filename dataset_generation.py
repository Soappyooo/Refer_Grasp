import blenderproc as bproc
import blenderproc.python.renderer.RendererUtility as RendererUtility
import numpy as np
import os
import sys
import time
import bpy
from mathutils import Vector
import logging
import random
from tqdm import tqdm
import argparse
import psutil
import signal

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# for blender debugging
# sys.path.append(os.getcwd())
###

from core.scene_graph import SceneGraph
from utils.dataset_utils import DatasetUtils
from utils.blender_utils import BlenderUtils

# set parameters
GPU_ID = 0  # GPU device id used for rendering
OBJ_DIR = os.path.abspath("./models")  # obj files directory
BLENDER_SCENE_FILE_PATH = os.path.abspath("./blender_files/background.blend")  # background scene file path
OUTPUT_DIR = os.path.abspath("./output_demo")
MODELS_INFO_FILE_PATH = os.path.abspath("./models/models_info_all.xlsx")
LOG_FILE_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "./dataset_generation.log"))
ITERATIONS = 10
IMAGES_PER_ITERATION = 5  # total images <= IMAGES_PER_ITERATION * ITERATIONS
RESOLUTION_WIDTH = 512
RESOLUTION_HEIGHT = 512
SCENE_GRAPH_ROWS = 4
SCENE_GRAPH_COLS = 4
SEED = None
PERSISITENT_DATA_CLEANUP_INTERVAL = 100  # clean up persistent data may speed up rendering for large dataset generation
TEXTURE_LIMIT = "2048"
CPU_THREADS = 0
SAMPLES = 512

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, default=GPU_ID, help="GPU device id used for rendering")
parser.add_argument("--obj-dir", type=str, default=OBJ_DIR, help="obj files directory")
parser.add_argument("--blender-scene-file-path", type=str, default=BLENDER_SCENE_FILE_PATH, help="background scene file path")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="output directory")
parser.add_argument("--models-info-file-path", type=str, default=MODELS_INFO_FILE_PATH, help="models info file path")
parser.add_argument("--log-file-path", type=str, default=LOG_FILE_PATH, help="log file path")
parser.add_argument("--iterations", type=int, default=ITERATIONS, help="number of iterations")
parser.add_argument("--images-per-iteration", type=int, default=IMAGES_PER_ITERATION, help="total images <= iterations * images_per_iteration")
parser.add_argument("--resolution-width", type=int, default=RESOLUTION_WIDTH, help="render resolution width")
parser.add_argument("--resolution-height", type=int, default=RESOLUTION_HEIGHT, help="render resolution height")
parser.add_argument("--scene-graph-rows", type=int, default=SCENE_GRAPH_ROWS, help="scene graph rows")
parser.add_argument("--scene-graph-cols", type=int, default=SCENE_GRAPH_COLS, help="scene graph cols")
parser.add_argument("--seed", type=int, default=SEED, help="random seed, should not be set unless debugging")
parser.add_argument(
    "--persistent-data-cleanup-interval", type=int, default=PERSISITENT_DATA_CLEANUP_INTERVAL, help="clean up persistent data interval"
)
parser.add_argument("--texture-limit", type=str, default=TEXTURE_LIMIT, help="texture limit")
parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS, help="CPU threads used for rendering")
parser.add_argument("--samples", type=int, default=SAMPLES, help="max samples for rendering")
args = parser.parse_args()
# convert path to absolute path
args.obj_dir = os.path.abspath(args.obj_dir)
args.blender_scene_file_path = os.path.abspath(args.blender_scene_file_path)
args.output_dir = os.path.abspath(args.output_dir)
args.models_info_file_path = os.path.abspath(args.models_info_file_path)
args.log_file_path = os.path.abspath(args.log_file_path)

# create logging directory and configure logging
if not os.path.exists(os.path.dirname(args.log_file_path)):
    os.makedirs(os.path.dirname(args.log_file_path))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - line %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=args.log_file_path,
    filemode="a",
)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - line %(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(stream_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def signal_handler(sig, frame):
    logging.error(f"Received signal {sig}, exiting...")
    # exit blender
    bpy.ops.wm.quit_blender()
    logging.info("Blender exited")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def sample_pose(obj: bproc.types.MeshObject, surface: bproc.types.MeshObject):
    """
    Sample a random pose for the object on the surface according to its coordinate in the scene graph.

    Args:
        obj (bproc.types.MeshObject): The object to be placed.
        surface (bproc.types.MeshObject): The surface on which the object will be placed.
    """
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


def render_settings_init():
    # set up render settings
    bproc.renderer.enable_segmentation_output(map_by=["scene_id"], default_values={"scene_id": 255})
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    RendererUtility.set_render_devices(desired_gpu_ids=[args.gpu_id])
    # set up resolution
    bpy.context.scene.render.resolution_x = args.resolution_width
    bpy.context.scene.render.resolution_y = args.resolution_height
    # full global illumination
    RendererUtility.set_light_bounces(32, 32, 32, 32, 32, 32, 32)
    # 2k textures
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.cycles.texture_limit_render = args.texture_limit
    # persistent data
    bpy.context.scene.render.use_persistent_data = True
    # dynamic bvh
    bpy.context.scene.cycles.debug_bvh_type = "DYNAMIC_BVH"
    # log cpu and gpu used
    is_gpu = False
    for device_type in ["OPTIX", "CUDA", "HIP"]:
        try:
            device_name = bpy.context.preferences.addons["cycles"].preferences.get_devices_for_type(device_type)[args.gpu_id].name
            logging.info(f"CPU threads: {bpy.context.scene.render.threads}, " + f"GPU devices: {device_name} ({device_type})")
            is_gpu = True
            break
        except:
            continue
    if not is_gpu:
        logging.info(f"CPU threads: {bpy.context.scene.render.threads}, GPU devices: None")


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
    area_light_data.energy = 100
    area_light_data.size = 20
    area_light_obj = bpy.data.objects.new(name="area_light", object_data=area_light_data)
    area_light_obj.location = (0, 0, 3)
    bpy.context.collection.objects.link(area_light_obj)

    # set up point light
    point_light = bproc.types.Light(light_type="POINT", name="point_light")

    return surfaces, point_light


def load_objects(obj_dir: str) -> list[bproc.types.MeshObject]:
    """
    Load objects from the obj files in the obj_dir and set up properties for rendering.

    Args:
        obj_dir (str): The directory containing the obj files.

    Returns:
        list[bproc.types.MeshObject]: a list of loaded objects.
    """
    active_objs: list[bproc.types.MeshObject] = []
    for file_name in os.listdir(obj_dir):
        if file_name.endswith(".obj"):
            obj = bproc.loader.load_obj(os.path.join(obj_dir, file_name))[0]
            active_objs.append(obj)
            obj.set_cp("obj_id", file_name.split(".")[0])  # should be like obj_000001
        ####### for debugging
        # if len(active_objs) > 5:
        #     break
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


def sample_objects(
    scene_graph: SceneGraph, active_objs: list[bproc.types.MeshObject], surfaces: list[bproc.types.MeshObject]
) -> tuple[list[bproc.types.MeshObject], bproc.types.MeshObject]:
    """
    Place objects in the scene according to the scene graph.

    Args:
        scene_graph (SceneGraph): The scene graph containing the objects and their coordinates.
        active_objs (list[bproc.types.MeshObject]): The objects to be placed.
        surfaces (list[bproc.types.MeshObject]): The surfaces on which the objects will be placed.

    Returns:
        tuple[list[bproc.types.MeshObject], bproc.types.MeshObject]: a tuple of (sampled objects, surface).
    """
    while scene_graph.create_scene(6) is False:
        continue
    obj_names = [objectNode.obj_id for objectNode in scene_graph.object_nodes]
    selected_objs: list[bproc.types.MeshObject] = []
    # copy objects in scene graph
    for j, obj_name in enumerate(obj_names):
        selected_obj: bproc.types.MeshObject = bproc.filter.one_by_cp(active_objs, "obj_id", obj_name).duplicate()
        selected_obj.set_cp("coordinate", scene_graph.object_nodes[j].coordinate)
        selected_obj.set_cp("scene_id", j)
        selected_obj.blender_obj.hide_render = False
        selected_obj.blender_obj.hide_set(False)
        selected_obj.enable_rigidbody(True, friction=1)
        selected_objs.append(selected_obj)

    surface: bproc.types.MeshObject = np.random.choice(surfaces)
    sampled_objs = bproc.object.sample_poses_on_surface(
        selected_objs, surface, sample_pose, min_distance=0.001, max_distance=10, check_all_bb_corners_over_surface=True, max_tries=100
    )
    if not len(sampled_objs) == len(selected_objs):
        # logging.warning("Some objects cannot be placed in scene, skip this scene")
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
    """
    Construct the scene for a single image. Sample camera pose, light and background material.

    Args:
        surface (bproc.types.MeshObject): The surface on which the objects are placed. Used for setting up camera and light pose.
        light (bproc.types.Light): The light in the scene.
        selected_objs (list[bproc.types.MeshObject]): The objects which have been placed in the scene.
        background_entities (list[bproc.types.Entity]): The background entities.

    Returns:
        bool: True if the scene is successfully constructed, False otherwise.
    """
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
        # sample camera focal length
        bpy.context.scene.camera.data.angle = np.random.uniform(0.65, 1.1)
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
    logging.info(args)
    start_time = time.time()
    # * Check file nums in output_dir
    if not DatasetUtils.check_image_file_nums(args.output_dir, ["depth", "rgb", "mask"]):
        logging.error("Image file numbers not match")
        raise Exception("Image file numbers not match")

    # * Initialize blenderproc & load scene
    bproc.init()
    render_settings_init()
    background_entities = bproc.loader.load_blend(args.blender_scene_file_path)
    surfaces, light = background_scene_init(background_entities)
    logging.info("Loaded background scene")

    # * Load objects
    active_objs = load_objects(args.obj_dir)
    logging.info("Loaded objects")

    # * Initialize scene graph
    scene_graph = SceneGraph((args.scene_graph_rows, args.scene_graph_cols))
    scene_graph.load_objects_info(args.models_info_file_path)
    logging.info("Scene graph initialized")

    # * Dataset generation
    pbar = tqdm(total=args.iterations, ncols=80, desc="Progress")
    i = 0
    while i < args.iterations:
        if i % args.persistent_data_cleanup_interval == 0:
            # clear persistent data may speed up rendering for large dataset generation
            bpy.context.scene.render.use_persistent_data = False
            bpy.context.scene.render.use_persistent_data = True
            for block in bpy.data.meshes:
                if block.users == 0:
                    bpy.data.meshes.remove(block)

            for block in bpy.data.materials:
                if block.users == 0:
                    bpy.data.materials.remove(block)

            for block in bpy.data.textures:
                if block.users == 0:
                    bpy.data.textures.remove(block)

            for block in bpy.data.images:
                if block.users == 0:
                    bpy.data.images.remove(block)
            logging.info(f"Cleaned up persistent data at iteration {i}")
        random_seed = (int(time.time()) * os.getpid()) % (2**32 - 1) if args.seed is None else args.seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        logging.info(f"Iteration {i} started with random seed {random_seed}")
        # log memory usage and gpu memory usage
        logging.info(f"Memory usage: { psutil.virtual_memory().used / 1024 ** 3:.2f} GB / {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")
        logging.info(f"Swap memory usage: {psutil.swap_memory().used / 1024 ** 3:.2f} GB / {psutil.swap_memory().total / 1024 ** 3:.2f} GB")
        # place objects in scene
        placed_objects, surface = sample_objects(scene_graph, active_objs, surfaces)
        if not placed_objects:
            logging.warning(f"Cannot place some objects in scene for iteration {i}, restart iteration")
            continue
        logging.info("Placed objects in scene")
        sys.stdout.flush()
        # set pass index for segmentation
        for index, obj in enumerate([obj for obj in bpy.context.scene.objects if obj.type == "MESH"]):
            obj.pass_index = index + 1
        # sample environment and render images
        image_indices = []
        for image_index in range(args.images_per_iteration):
            bproc.utility.reset_keyframes()
            # sample camera pose, light and background material
            render_flag = construct_scene_for_single_image(surface, light, placed_objects, background_entities)
            if not render_flag:
                break
            logging.info(f"Constructed scene for image ({image_index + 1}/{args.images_per_iteration})")
            # render image
            data = bproc.renderer.render()
            data["depth"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])
            logging.info(f"Rendered image ({image_index + 1}/{args.images_per_iteration})")
            sys.stdout.flush()
            # write image
            depth_idxs = DatasetUtils.write_image(
                data, os.path.join(args.output_dir, "depth"), "depth", file_name_prefix="depth", append_to_exsiting_file=True
            )
            logging.info(f"Wrote depth images: {[f'depth_{idx:08d}.png' for idx in depth_idxs]}")
            colors_idxs = DatasetUtils.write_image(
                data, os.path.join(args.output_dir, "rgb"), "colors", file_name_prefix="rgb", append_to_exsiting_file=True
            )
            logging.info(f"Wrote rgb images: {[f'rgb_{idx:08d}.jpg' for idx in colors_idxs]}")
            segmaps_idxs = DatasetUtils.write_image(
                data, os.path.join(args.output_dir, "mask"), "scene_id_segmaps", file_name_prefix="mask", append_to_exsiting_file=True
            )
            logging.info(f"Wrote mask images: {[f'mask_{idx:08d}.png' for idx in segmaps_idxs]}")
            assert depth_idxs == colors_idxs == segmaps_idxs
            image_indices += depth_idxs

        if len(image_indices) > 0:
            logging.info(f"Iteration {i} finished with {len(image_indices)} images rendered")
            # write expressions
            expressions = []
            for j, expression in enumerate(scene_graph.get_complex_referring_expressions()):
                node = list(scene_graph.expression_structures.keys())[j]
                # obj_id = int(node.obj_id.split("_")[1])
                obj_id = node.obj_id
                expressions.append({"obj": {"id": obj_id, "scene_id": node.scene_id}, "expression": expression})
            expressions_idx = DatasetUtils.write_expressions(
                image_indices, expressions, save_path=os.path.join(args.output_dir, "expressions"), file_name_prefix="expressions"
            )
            logging.info(f"Wrote expressions json: expressions_{expressions_idx:08d}.json")
            # write scene graph
            scene_graph_idx = SceneGraph.write_scene_graph_to_file(scene_graph, os.path.join(args.output_dir, "scene_graphs"), "scene_graph")
            logging.info(f"Wrote scene graph json: scene_graph_{scene_graph_idx:08d}.json")
            i += 1
            pbar.update(1)
            logging.info(str(pbar))
        else:
            logging.warning(f"No images rendered for iteration {i}, restart iteration")
        sys.stdout.flush()
        # clean up
        for obj in placed_objects:
            obj.delete(remove_all_offspring=True)
    pbar.close()
    logging.info("Dataset generation finished")

    # * Merge expressions
    num_expressions = DatasetUtils.merge_expressions(
        os.path.join(args.output_dir), os.path.join(args.output_dir, "expressions"), delete_temp=False, num_files=args.iterations
    )
    logging.info(f"Merged expressions jsons ({num_expressions} scenes total)")
    # calculate time H:M:S
    logging.info(f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    logging.info(f"Process finished with {i} successful iterations, see {os.path.abspath(args.log_file_path)} for details")
