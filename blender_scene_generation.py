import blenderproc as bproc
import numpy as np
import os

obj_dir = "./ycb models/object+ply"  # 模型存放目录
background_dir = "."  # 背景blend文件存放目录
background_file = "background1.blend"  # 背景blend文件名
hdf5_output_dir = "./output/hdf5"  # 输出hdf5文件目录
coco_output_dir = "./output/coco"  # 输出coco文件目录
iterations = 1
images_per_iteration = 5  # 每个iteration生成的图片数量，不同视角
resolution_w = 256
resolution_h = 256


bproc.init()

print("loading background...")
# load background into the scene
background_objs = bproc.loader.load_blend(os.path.join(background_dir, background_file))
for obj in background_objs:
    obj.set_cp("category_id", 0)

print("loading objects...")
# load all objects with suffix "obj" from obj_dir into the scene
active_objs: list[bproc.types.MeshObject] = []
for file_name in os.listdir(obj_dir):
    if file_name.endswith(".obj"):
        active_objs += bproc.loader.load_obj(os.path.join(obj_dir, file_name), use_legacy_obj_import=True)

    # for test
    # if len(active_objs) > 10:
    #     break
    # #######

for j, obj in enumerate(active_objs):
    obj.set_cp("category_id", j + 1)


surface: bproc.types.MeshObject = bproc.filter.one_by_attr(background_objs, "name", "plane")
surface.hide(True)
planes: list[bproc.types.MeshObject] = bproc.filter.by_attr(background_objs, "name", "^plane(?!$).*$", regex=True)

for obj in active_objs:
    obj.set_location([2, 2, 2])
    obj.hide(True)

for plane in planes:
    plane.enable_rigidbody(False, friction=1)

light = bproc.types.Light()
light.set_type("POINT")

# activate normal and depth rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])


def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=[surface], min_height=0.3, max_height=1, use_ray_trace_check=False))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))


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


for i in range(iterations):

    bproc.utility.reset_keyframes()
    # random select 3-6 active objects
    selected_objs: list[bproc.types.MeshObject] = np.random.choice(active_objs, np.random.randint(3, 6), replace=False)

    for obj in selected_objs:
        obj.enable_rigidbody(True, friction=1)
        obj.hide(False)

    surface.set_location(np.random.uniform([-0.7, 0.4, 0], [-0.4, 0.7, 0]))  # sample surface location

    selected_objs = bproc.object.sample_poses_on_surface(selected_objs, surface, sample_pose, min_distance=0.005, max_distance=10)

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)

    # define a light and set its location and energy level
    light.set_location(np.random.uniform([-1, 0, 0.5], [0, 1, 1.5]))
    light.set_energy(np.random.uniform(5, 100))
    light.set_color(hsv2rgb(np.random.uniform([0, 0, 1], [1, 0.7, 1])))

    poi = bproc.object.compute_poi(selected_objs)

    for i in range(images_per_iteration):
        # Sample random camera location above objects
        location = np.random.uniform([0.5 + poi[0], -0.8 + poi[1], 0.1], [0.8 + poi[0], -0.5 + poi[1], 0.5])
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi / 6, np.pi / 6))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    bproc.camera.set_resolution(resolution_w, resolution_h)

    # render the whole pipeline
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
    data = bproc.renderer.render()
    data["depth"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])

    for obj in selected_objs:
        obj.disable_rigidbody()
        obj.hide(True)

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(hdf5_output_dir, data, append_to_existing_output=True)
    bproc.writer.write_coco_annotations(
        coco_output_dir,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )
