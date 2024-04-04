import blenderproc as bproc
import bpy
from mathutils import Vector
import numpy as np
bproc.init()
objs=bproc.loader.load_blend("D:/BlenderScripts/scene.blend")
active_objs=bproc.filter.by_attr(objs,"name","^(?!plane).*$",regex=True)

resolution_w=512
resolution_h=512
def CheckOcclusion(objs_to_check: list[bproc.types.MeshObject], res_ratio:float, threshold:float=0.4) -> bool:
    areas_with_occlusion={}
    blender_objs_to_check=[obj.blender_obj for obj in objs_to_check]
    depsgraph=bpy.context.evaluated_depsgraph_get()
    camera=bpy.context.scene.objects["Camera"]
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=bpy.context.scene)
    camera_quaternion = camera.matrix_world.to_quaternion()
    camera_translation = camera.matrix_world.translation
    x_range = np.linspace(top_left[0], top_right[0], int(resolution_w*res_ratio))
    y_range = np.linspace(top_left[1], bottom_left[1], int(resolution_h*res_ratio))
    z_dir = top_left[2]
    pixel_vectors=[]
    for x in x_range:
        line=[]
        for y in y_range:
            pixel_vector=Vector((x,y,z_dir))
            pixel_vector.rotate(camera_quaternion)
            pixel_vector.normalize()
            line.append(pixel_vector)
        pixel_vectors.append(line)  
    for obj_to_check in blender_objs_to_check:
        areas_with_occlusion[obj_to_check]=0
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            is_hit,_,_,_,hit_obj,_=bpy.context.scene.ray_cast(depsgraph,camera_translation,pixel_vectors[i][j])
            if is_hit and hit_obj in blender_objs_to_check:
                areas_with_occlusion[hit_obj]+=1    
    for obj_to_check in blender_objs_to_check:
        # hide other objs
        for obj in blender_objs_to_check:
            if obj!=obj_to_check:
                obj.hide_set(True)
        area_without_occlusion=0
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                is_hit,_,_,_,hit_obj,_=bpy.context.scene.ray_cast(depsgraph,camera_translation,pixel_vectors[i][j])
                if is_hit and hit_obj==obj_to_check:
                    area_without_occlusion+=1
        # show other objs
        for obj in blender_objs_to_check:
            if obj!=obj_to_check:
                obj.hide_set(False)
        if area_without_occlusion==0 or areas_with_occlusion[obj_to_check]/area_without_occlusion<threshold:
            return False
    return True

CheckOcclusion(active_objs,0.5)