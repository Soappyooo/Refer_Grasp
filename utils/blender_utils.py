import blenderproc as bproc
import bpy
import numpy as np
from mathutils import Vector, Matrix
from typing import Union


class BlenderUtils:
    @staticmethod
    def set_random_material(obj: bpy.types.Object):
        """
        Set the slot 0 material of the object to a random material from the other slots.

        Args:
            obj (bpy.types.Object): the object to set the material.
        """
        obj.material_slots[0].material = obj.material_slots[np.random.randint(1, len(obj.material_slots))].material

    @staticmethod
    def sample_point_in_cuboid(transform_matrix: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        """
        Sample a random point inside a cuboid defined by its dimensions and transformation matrix.

        Args:
            transform_matrix (np.ndarray): A transformation matrix (4x4) of the cuboid center.
            dimensions (np.ndarray): (length, width, height) of the cuboid.

        Returns:
            np.ndarray: the random point inside the cuboid in the world coordinate system.
        """

        # Extract translation component from the transformation matrix
        translation = transform_matrix[:3, 3]

        # Extract rotation component from the transformation matrix
        rotation = transform_matrix[:3, :3]

        # Generate random point inside the cuboid in its local coordinate system
        local_point = np.random.rand(3) * dimensions - dimensions / 2

        # Transform the point to the world coordinate system
        world_point = np.dot(rotation, local_point) + translation

        return world_point

    @staticmethod
    def add_relative_translation_to_matrix(matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Add a relative translation to a transformation matrix.
        i.e. the translation is added in the direction of the object's local coordinate system.

        Args:
            matrix (np.ndarray): the transformation matrix.
            translation (np.ndarray): the translation to add.

        Returns:
            np.ndarray: the new transformation matrix.
        """

        new_matrix = matrix.copy()
        new_matrix[:3, 3] += np.dot(matrix[:3, :3], translation)
        return new_matrix

    @staticmethod
    def get_bound_box_local_coordinates(obj: bpy.types.Object) -> list[np.ndarray]:
        """
        Get the bounding box of the object in the local coordinates.

        Args:
            obj (bpy.types.Object): the object.

        Returns:
            list[np.ndarray]: the bounding box in the local coordinates.
        """

        return [np.array(corner) for corner in obj.bound_box]

    @staticmethod
    def get_bound_box_world_coordinates(obj: bpy.types.Object) -> list[np.ndarray]:
        """
        Get the bounding box of the object in the world coordinates.

        Args:
            obj (bpy.types.Object): the object.

        Returns:
            list[np.ndarray]: the bounding box in the world coordinates.
        """

        return [np.array(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]

    @staticmethod
    def check_occlusion(
        objs_to_check: list[bpy.types.Object], res_ratio: float, threshold: float = 0.4, min_hit: int = 1, depsgraph: bpy.types.Depsgraph = None
    ) -> bool:
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
            depsgraph (bpy.types.Depsgraph, optional): Dependency graph. Defaults to None.

        Returns:
            bool: whether the objects are occluded or not. True if not occluded.
        """
        camera = bpy.context.scene.objects["Camera"]
        if depsgraph is None:
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

    @staticmethod
    def check_occlusion_deprecated(
        objs_to_check: list[bpy.types.Object], res_ratio: float, threshold: float = 0.4, min_hit: int = 1, depsgraph: bpy.types.Depsgraph = None
    ) -> bool:
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
            depsgraph (bpy.types.Depsgraph, optional): Dependency graph. Defaults to None.

        Returns:
            bool: whether the objects are occluded or not. True if not occluded.
        """
        # TODO: change objects to fasten ray_cast
        areas_with_occlusion = {}
        areas_without_occlusion = {}
        if depsgraph is None:
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

    @staticmethod
    def check_occlusion_rough(objs_to_check: list[bpy.types.Object], depsgraph: bpy.types.Depsgraph = None) -> bool:
        """
        Check if anything is occluding the objects in the scene. Cast rays from camera to object centers, if hit anything not in `objs_to_check`, return False.

        Args:
            objs_to_check (list[bpy.types.Object]): the objects to check.
            depsgraph (bpy.types.Depsgraph, optional): Dependency graph. Defaults to None.

        Returns:
            bool: whether the objects are occluded or not. True if not occluded.
        """
        # ray cast from camera to object. If hit anything not in the list, return False
        camera = bpy.context.scene.objects["Camera"]
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        camera_translation = camera.matrix_world.translation
        for obj in objs_to_check:
            is_hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, camera_translation, obj.matrix_world.translation - camera_translation)
            if is_hit and hit_obj not in objs_to_check:
                return False
        return True

    @staticmethod
    def check_points_in_view(points: list[np.ndarray], max_distance: float = 10) -> bool:
        """
        Check if the points are in the camera view.

        Args:
            points (list[np.ndarray]): the points to check.
            max_distance (float, optional): the maximum distance from the camera. Defaults to 10.

        Returns:
            bool: whether the points are in the camera view.
        """
        for point in points:
            if not bproc.camera.is_point_inside_camera_frustum(point):
                return False
            # calculate distance
            distance = np.linalg.norm(np.array(bpy.context.scene.camera.location) - point)
            if distance > max_distance:
                return False
        return True

    @staticmethod
    def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
        """
        Convert HSV color to RGB color.

        Args:
            hsv (np.ndarray): HSV color (h:0-1, s:0-1, v:0-1).

        Raises:
            ValueError: Invalid h value.

        Returns:
            np.ndarray: RGB color (r:0-1, g:0-1, b:0-1).
        """
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

    @staticmethod
    def poi(objs: list[bproc.types.MeshObject]) -> np.ndarray:
        """
        Calculate the point of interest (volumn center) of the objects.

        Args:
            objs (list[bproc.types.MeshObject]): the objects.

        Returns:
            np.ndarray: the POI.
        """
        poi = np.array(
            [
                np.mean(
                    [
                        np.max(np.array([obj.get_bound_box()[:, 0] for obj in objs])),
                        np.min(np.array([obj.get_bound_box()[:, 0] for obj in objs])),
                    ]
                ),
                np.mean(
                    [
                        np.max(np.array([obj.get_bound_box()[:, 1] for obj in objs])),
                        np.min(np.array([obj.get_bound_box()[:, 1] for obj in objs])),
                    ]
                ),
                np.mean(
                    [
                        np.max(np.array([obj.get_bound_box()[:, 2] for obj in objs])),
                        np.min(np.array([obj.get_bound_box()[:, 2] for obj in objs])),
                    ]
                ),
            ]
        )
        return poi
