import bpy
import numpy as np
from mathutils import *
import time
from utils.blender_utils import BlenderUtils


def check_occlusion(objs_to_check: list[bpy.types.Object], res_ratio: float, threshold: float = 0.4, min_hit: int = 1) -> bool:
    """
    Check if the objects are occluded by other objects in the scene.
    `res_ratio` (0-1) downsamples the resolution for occlusion checking to speed up the process.
    If visible area / total area is below `threshold`, the object is considered occluded,
    i.e., the bigger `threshold` promises better object visibility.

    Args:
        objs_to_check (list[bpy.types.Object]): the objects to check.
        res_ratio (float): Resolution ratio (0-1) for occlusion checking.
        threshold (float, optional): Threshold for determining occluded or not. Defaults to 0.4.
        min_hit (int, optional): Minimum number of ray hits to consider the object visible, \
        bigger means larger visible object area (with other objects not hidden). Defaults to 1.

    Returns:
        bool: whether the objects are occluded or not.
    """
    # TODO: change objects to fasten ray_cast
    areas_with_occlusion = {}
    areas_without_occlusion = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = bpy.context.scene.objects["Camera"]
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=bpy.context.scene)
    camera_quaternion = camera.matrix_world.to_quaternion()
    camera_translation = camera.matrix_world.translation
    x_range = np.linspace(top_left[0], top_right[0], int(bpy.context.scene.render.resolution_x * res_ratio))
    y_range = np.linspace(top_left[1], bottom_left[1], int(bpy.context.scene.render.resolution_y * res_ratio))
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

    for obj_to_check in objs_to_check:
        areas_with_occlusion[obj_to_check] = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
            if is_hit and hit_obj in objs_to_check:
                areas_with_occlusion[hit_obj] += 1

    for obj_to_check in objs_to_check:
        areas_without_occlusion[obj_to_check] = 0
        # hide other objs
        for obj in objs_to_check:
            if obj != obj_to_check:
                obj.hide_viewport = True
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, pixel_vectors[i][j])
                if is_hit and hit_obj == obj_to_check:
                    areas_without_occlusion[obj_to_check] += 1
        # show other objs
        for obj in objs_to_check:
            if obj != obj_to_check:
                obj.hide_viewport = False
        if (
            areas_without_occlusion[obj_to_check] == 0
            or areas_with_occlusion[obj_to_check] < min_hit
            or areas_with_occlusion[obj_to_check] / areas_without_occlusion[obj_to_check] < threshold
        ):
            return False
    return True


def check_occlusion_new(objs_to_check: list[bpy.types.Object], res_ratio: float, threshold: float = 0.4, min_hit: int = 1) -> bool:
    camera = bpy.context.scene.objects["Camera"]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=bpy.context.scene)
    camera_quaternion = camera.matrix_world.to_quaternion()
    camera_translation = camera.matrix_world.translation
    x_range = np.linspace(top_left[0], top_right[0], int(bpy.context.scene.render.resolution_x * res_ratio))
    y_range = np.linspace(top_left[1], bottom_left[1], int(bpy.context.scene.render.resolution_y * res_ratio))
    z_dir = top_left[2]
    rays = []
    pixel_vectors_of_obj = {}
    areas_with_occlusion = {}
    areas_without_occlusion = {}

    def get_camera_view_bbox_approximation(obj: bpy.types.Object) -> list[Vector]:
        bbox = BlenderUtils.get_bound_box_world_coordinates(obj)
        bbox_in_camera_coords = []
        for i in range(8):
            point_projected = camera.matrix_world.inverted() @ Vector(bbox[i])
            point_projected *= z_dir / point_projected[2]
            bbox_in_camera_coords.append(point_projected)
        # bbox_in_camera_coords_2d[0]: min bbox_in_camera_coords[i][0], max bbox_in_camera_coords[i][1]
        # bbox_in_camera_coords_2d[1]: max bbox_in_camera_coords[i][0], min bbox_in_camera_coords[i][1]
        bbox_in_camera_coords_2d = [
            Vector((min([point.x for point in bbox_in_camera_coords]), max([point.y for point in bbox_in_camera_coords]), z_dir)),
            Vector((max([point.x for point in bbox_in_camera_coords]), min([point.y for point in bbox_in_camera_coords]), z_dir)),
        ]
        return bbox_in_camera_coords_2d

    bbox_2d_of_obj = {}
    for obj in objs_to_check:
        bbox_2d_of_obj[obj] = get_camera_view_bbox_approximation(obj)

    for x in x_range:
        for y in y_range:
            ray = Vector((x, y, z_dir))
            # check within the bbox of any object
            ignore_ray = True
            relevant_objs = []
            for key, bbox_2d in bbox_2d_of_obj.items():
                if bbox_2d[0].x <= x <= bbox_2d[1].x and bbox_2d[1].y <= y <= bbox_2d[0].y:
                    ignore_ray = False
                    relevant_objs.append(key)
            if ignore_ray:
                continue
            ray.rotate(camera_quaternion)
            rays.append(ray)
            for obj in relevant_objs:
                if obj not in pixel_vectors_of_obj:
                    pixel_vectors_of_obj[obj] = []
                pixel_vectors_of_obj[obj].append(ray)

    for obj_to_check in objs_to_check:
        areas_with_occlusion[obj_to_check] = 0
    for ray in rays:
        is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, ray)
        if is_hit and hit_obj in objs_to_check:
            areas_with_occlusion[hit_obj] += 1

    for obj_to_check in objs_to_check:
        areas_without_occlusion[obj_to_check] = 0
        if pixel_vectors_of_obj.get(obj_to_check) is None:
            return False
        # hide other objs
        for obj in objs_to_check:
            if obj != obj_to_check:
                obj.hide_viewport = True
        for ray in pixel_vectors_of_obj[obj_to_check]:
            is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, ray)
            if is_hit and hit_obj == obj_to_check:
                areas_without_occlusion[obj_to_check] += 1
        # show other objs
        for obj in objs_to_check:
            if obj != obj_to_check:
                obj.hide_viewport = False
        if (
            areas_without_occlusion[obj_to_check] == 0
            or areas_with_occlusion[obj_to_check] < min_hit
            or areas_with_occlusion[obj_to_check] / areas_without_occlusion[obj_to_check] < threshold
        ):
            return False
    return True


def check_occlusion_rough(objs_to_check: list[bpy.types.Object]) -> bool:
    """
    Check if anything is occluding the objects in the scene. Cast rays from camera to object centers, if hit anything not in `objs_to_check`, return False.

    Args:
        objs_to_check (list[bpy.types.Object]): the objects to check.

    Returns:
        bool: whether the objects are occluded or not. True if not occluded.
    """
    # ray cast from camera to object. If hit anything not in the list, return False
    camera = bpy.context.scene.objects["Camera"]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_translation = camera.matrix_world.translation
    for obj in objs_to_check:
        is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, obj.location - camera_translation)
        if is_hit and hit_obj not in objs_to_check:
            return False
    return True


objs = [
    bpy.data.objects["obj_000000"],
    bpy.data.objects["obj_000001"],
    bpy.data.objects["obj_000002"],
    bpy.data.objects["obj_000003"],
    bpy.data.objects["obj_000004"],
    bpy.data.objects["obj_000005"],
]
print(check_occlusion(objs, 0.05, 0.5))
