from matplotlib import pyplot as plt
from multiprocessing import Process
import math
import numpy as np
import transducer
import octree
import sys
import importlib
importlib.reload(transducer)
importlib.reload(octree)
import os
import yaml
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

class Probe:
    def __init__(self, T, p2w_euler, N, max_dis, width, length, height, box, r = 0, shape = 'Rectangular', render = False):
        '''
        Parameters:
        T: Translation vector
        p2w_euler: Rotation from probe to world in euler angle
        N: Number of transducers
        max_dis: Maximal ray distance
        width: Width of probe 
        length: Length of probe
        height: Height of probe
        box: Tumor box
        r: Only for fan-shaped probe: Radius from the origin of the probe to the origin of the ray
        shape: Rectangular or Fan
        render: Whether render available
        '''
        self.T = T
        self.p2w_euler = p2w_euler
        self.p2w_mat = p2w_euler.to_matrix()
        self.N = N
        self.max_dis = max_dis
        self.width = width
        self.length = length
        self.height = height
        self.box = box
        self.r = r
        self.shape = shape
        self.t_size = r/(N-1) # Transducer size
        self.render = render
        self.adjust = True
        self.success = False
        if RENDER:
            bpy.data.objects['probe'].location = T
            bpy.data.objects['probe'].rotation_euler = p2w_euler

        # bpy.ops.mesh.primitive_cube_add(size = 1, location = T,
        #     rotation = p2w_euler)
        # bpy.data.objects['Cube'].name = 'probe'
        # bpy.data.objects['probe'].dimensions = [width, height, length]
        # bpy.data.objects['probe']

        self.generate_transducers()

    def update_tree(self, tree):
        '''
        Delete the old tree and update the octree for tumor
        Parameters:
            tree: New octree for tumor
        '''
        del self.tree
        self.tree = tree
    
    def update_box(self, box):
        '''
        Delete the old box and update the tumor box
        Parameters:
            box: New tumor box
        '''
        del self.box
        self.box = box
    
    def update_pose(self, T, euler):
        '''
        update the pose of probe
        Parameters:
            T: New location of probe
            euler: New rotation of probe
        '''
        if RENDER:
            probe = bpy.data.objects['probe']
            probe.location = T
            probe.rotation_euler = euler
        self.T = T
        self.p2w_euler = euler
        self.p2w_mat = self.p2w_euler.to_matrix()
        self.generate_transducers()
    
    def is_pose_feasible(self):
        '''
        Return True if the probe is on the outward surface of the skin
        Otherwise return False
        '''
        # Check probe height
        z = self.T[2]
        if z >= 260 or z <= 100:
            print("Infeasible height")
            return False
        # Check if the pose(location and rotation) of probe is feasible
        objects = bpy.data.objects
        skin = objects['skin']
        dis_th = 150 # Distance threshold for determining if it is on mesh
        # Check if it is on the outward surface
        bd = skin.bound_box
        bd_x = [point[0] for point in bd]
        bd_y = [point[1] for point in bd]
        bd_z = [point[2] for point in bd]
        x_max = max(bd_x)
        x_min = min(bd_x)
        y_max = max(bd_y)
        y_min = min(bd_y)
        z_max = max(bd_z)
        z_min = min(bd_z)
        center = Vector(((x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2)) # Center of bound box
        any_feasible_ray = False
        v_normal_avg = Vector((0, 0, 0))
        for t in self.transducers:
            O_world = t.O # Origin of the ray in world coordinate
            res, loc, v_normal, _ = skin.closest_point_on_mesh(O_world, distance = dis_th)
            v_normal_avg += v_normal
            if not res: 
                continue # The origin of the ray is not on the skin surface
                         # Check for the next ray
            v_center = loc - center # The vector from the center of bounding box to the probe location
#            print("closest:", loc)
#            print("point", O_world)

            # if v_center @ v_normal <= 0:
            #     print("There is ray on the inward surface")
            #     return False # The origin of the ray is not on the outward surface

            v_surface = loc - O_world # The vector from the origin of ray to the closest point
#             if v_surface @ v_normal > 0:
# #                print(v_surface)
# #                print(v_normal)
#                 print("There is ray between two surfaces")
#                 return False # The origin of the ray is between two surfaces
            any_feasible_ray = True

        # if not any_feasible_ray:
        #     print("No ray on the surface is found")
        #     return False # No origin of the ray is on skin surface
        
        # Check if the angle between the surface and probe is bigger than 30 degrees
        v = self.p2w_mat @ Vector((0, 0, 1)) # Direction of probe
        v_normal_avg = v_normal_avg.normalized() # Average normal vector
        v = v.normalized()
        angle = math.acos(v @ v_normal_avg) # Angle between the probe and surface
        # print("Angle:", math.degrees(angle))
        if angle > math.radians(120) and angle < math.radians(180):
            return True
        print("Infeasible angle")
        return False

    def generate_transducers(self):
        '''
        Generate the transducers
        '''
        self.transducers = []
        step = 1
        if self.shape == 'Rectangular':
            for i in range(self.N):
                x = self.width/2 - i*self.width/(self.N-1)
                O_probe = Vector((x, 0, -self.length/2)) # Origin of ray in probe coordinate
                O_world = self.p2w_mat @ O_probe + self.T # Origin of ray in world coordinate
                v = self.p2w_mat @ Vector((0, 0, 1))
                t = transducer.Transducer(O_world, v, step, self.t_size, self.box)
                self.transducers.append(t)
        else:
            for i in range(self.N):
                theta = -math.pi/4 + i*math.pi/(self.N-1)
                t2p_R = mathutils.Matrix.Rotation(theta, 3, 'Y') # Rotation matrix from transducer to probe
                O_probe = t2p_R @ Vector((0, 0, self.r)) # Origin of ray in probe coordinate
                O_world = self.p2w_mat @ O_probe + self.T # Origin of ray in world coordinate
                v = self.p2w_mat @ t2p_R @ Vector((0, 0, 1)) # Norminalized ray direction in world coordinate
                t = transducer.Transducer(O_world, v, step, self.t_size)
                self.transducers.append(t)
                
    def simulate_us_image(self, th = 3, need_image = False):
        '''
        Simulate the ultrasound image and save it
        '''
        # If adjusting position or the last scan is not successful, do not delete tumor points
        self.box.after_scan(last_success = self.success)
        tot_del_num = 0
        size = (self.N, self.max_dis)
        us_data = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        tot_bone_num = 0
        tot_avg_dis = 0
        for i in range(size[0]):
            # ray_trace_train only used for training
            del_num, bone_num, dis = self.transducers[i].ray_trace_train()
            tot_bone_num += bone_num
            tot_avg_dis += dis
            tot_del_num += del_num
            if need_image:
                for j in range(size[1]):
                    us_data[i][j] = column[j]
        # If adjusting position, the voxel value of tumor is 0.5
        if self.adjust:
            self.box.box[:][:][:][2] /= 2
        if need_image:
            us_data = np.transpose(us_data, (1, 0, 2))
            plt.imshow(us_data, interpolation='nearest')
            plt.savefig("images/us_image.png")
        
        # Scan is only successful, when the number of bone voxels scanned are lower than or equal to th and the probe is not in adjusting mode
        if tot_bone_num <= th and (not self.adjust):
            self.success = True
        else:
            self.success = False
            # If not success, revert the number of deleted tumor points
            self.box.tumor_num += tot_del_num 
            tot_del_num = 0
        # print(self.success, self.box.tumor_num, tot_bone_num, self.adjust, tot_del_num)
        if tot_del_num == 0:
            return tot_del_num, 0, 0
        tot_avg_dis /= tot_del_num
        # print("bone", tot_bone_num)
        return tot_del_num, 1-tot_bone_num/self.N, tot_avg_dis
            
if __name__ == "__main__":
    pass
        
            
            
            
    
