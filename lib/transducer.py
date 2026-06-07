# from mathutils import Vector
from math import radians
import octree
import importlib
import tbox
importlib.reload(octree)
importlib.reload(tbox)

import yaml
import os
dir = "lib"

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        RENDER = config['RENDER']
except:
    RENDER = True

if RENDER:
    import bpy

from mathutils import Vector
from mathutils import Euler

tissue_color = {
            "skin": [255, 204, 153],
            "bone": [255, 255, 0],
            "liver": [255, 153, 153], 
            "tumor": [0, 255, 0]
        } # Tissue color table

class Transducer:
    def __init__(self, O, v, step, size, box, max_dis = 300):
        '''
        Parameters:
        O: Origin of ray in world coordinate
        v: Ray direction
        step: Ray step size
        size: Size of transducer
        box: Tumor box
        max_dis: Maximal distance of ray
        '''
        self.O = O
        self.v = v
        self.step = step
        self.size = size
        self.box = box
        self.max_dis = max_dis                     

    def ray_trace_train(self):
        del_num = 0
        bone_num = 0
        scan_dis = 0
        for dis in range(0, self.max_dis, 2):
            point = self.O + dis*self.v
            ret = self.box.scan(point)
            if ret == -1:
                bone_num += 1
                return del_num, bone_num, 0
            elif ret == 1:
                scan_dis += dis
                del_num += 1
        return del_num, bone_num, scan_dis

    def ray_trace(self, tumor_offset):
        objects = bpy.data.objects
        skin = objects["skin"]
        bone = objects["bone"]
        liver = objects["liver"]
        tumor = objects["tumor"]
        bone_dis = float("inf") 
        tissues = [bone, skin, liver, tumor]
        pixel_col = [[0.0, 0.0, 0.0] for _ in range(self.max_dis)]
        del_num = 0
        for tissue in tissues:
            ray_begin = self.O
            ray_direction = self.v
            tissue_stack = []
            while True:
                result, loc, _, _ = tissue.ray_cast(ray_begin, ray_direction)
                if result:
                    dis = (loc-self.O).magnitude # Distance until next reflection
                    if tissue == bone:
                        # Stop, when detect the first interface of bone
                        bone_dis = dis # Distance of the first interface of bone
                        pixel_col[round(dis)] = tissue_color[tissue.name]
                        break
                        # pass
                    else:
                        # Stop, if the interface is behind the first interface of bone
                        if dis >= bone_dis:
                            break
                            # pass
                    # No bone is in front of the tissue
                    dis = round(dis)
                    # if tissue == tumor:
                    #     n = self.tree.del_point(self.tree.root, [loc[0], loc[1], loc[2]])
                    #     del_num += n
                        # if n > 0:
                        #    bpy.ops.mesh.primitive_cube_add(size = 4, location = loc,
                        #        rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
#                        print(n)
                    if len(tissue_stack) == 0:
                        tissue_stack.append(dis)
                        pixel_col[dis] = tissue_color[tissue.name]
                    else:
                        last_dis = tissue_stack.pop()
                        # Fill the color
                        for d in range(last_dis, dis+1):
                            point = self.O + self.v*d
                            point = [point[0], point[1], point[2]]
                            n = self.tree.del_point(self.tree.root, [point[0], point[1], point[2]], [tumor_offset[0], tumor_offset[1], tumor_offset[2]], self.box)
                            del_num += n
                            if d != last_dis:
                                pixel_col[d] = tissue_color[tissue.name]
                    ray_begin = loc + 0.1*ray_direction
                else:
                    break
        return pixel_col, del_num
                

if __name__ == "__main__":
    t = Transducer((0,0,0), (1,1,1), 1, 1)
