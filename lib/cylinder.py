import math
import random
import yaml
import sys
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

INIT_ANGLE = 225
INIT_Z = 180
INIT_X_ROT = 90
INIT_Y_ROT = 0
ANGLE_UB = 270
ANGLE_LB = 180
Z_UB = 250
Z_LB = 100
X_ROT_UB = 120
X_ROT_LB = 30

class Cylinder():
    def __init__(self, center, radius):
        '''
        Parameters:
            center: Center of the cylinder without z coordinate
            radius: Radius of the cylinder
        '''
        self.center = center
        self.radius = radius
        # self.delta_angle = 0
        # self.delta_z = 0
        # self.delta_x_rot = 0
        # self.delta_z_rot = 0
        self.angle = INIT_ANGLE
        self.z_rot = 0
        self.z = INIT_Z
        self.x_rot = 0
        self.inverse = 1
        x = self.center[0] + math.cos(math.radians(self.angle)) * self.radius
        y = self.center[1] + math.sin(math.radians(self.angle)) * self.radius
        z = INIT_Z
        self.cur_pos = Vector((x, y, z))
        self.cur_rot = Euler((math.radians(90), math.radians(0), math.radians(-225)))

    def move(self, delta_angle = 0, delta_z = 0, delta_x_rot = 0, delta_z_rot = 0):
        # if self.z + delta_z < Z_LB or self.z + delta_z > Z_UB:
        #     return False
        # if self.angle + delta_angle < ANGLE_LB or self.angle + delta_angle > ANGLE_UB:
        #     return False
        # if self.x_rot + delta_x_rot < X_ROT_LB or self.x_rot + delta_x_rot > X_ROT_UB:
        #     return False
        
        if self.z + delta_z > 360 or self.z + delta_z < 0: 
            print("Infeasible height")
            return False
        else:
            self.z += delta_z

        if self.x_rot + delta_x_rot > 60 or self.x_rot + delta_x_rot < -60: 
            print("Infeasible x rot")
            return False
        else:
            self.x_rot += delta_x_rot
        
        if self.angle + delta_angle < 150 or self.angle + delta_angle > 300:
            print("Infeasible angle")
            return False
        else:
            self.angle += delta_angle 
        
        '''
        self.z += delta_z
        self.x_rot += delta_x_rot
        self.angle += delta_angle
        '''
        self.z_rot += delta_z_rot
        if self.z_rot > 180:
            self.z_rot -= 360
        if self.z_rot < -180:
            self.z_rot += 360
        if self.z_rot < 0:
            self.inverse = -1
        else:
            self.inverse = 1
        self.delta_rot = Euler((math.radians(delta_x_rot*self.inverse), math.radians(0), math.radians(delta_z_rot)))
        self.delta_rot_pre = Euler((math.radians(0), math.radians(0), math.radians(delta_angle)))
        self.cur_rot = (self.delta_rot_pre.to_matrix() @ self.cur_rot.to_matrix() @ self.delta_rot.to_matrix()).to_euler()
        # self.angle += delta_angle
        x = self.center[0] + math.cos(math.radians(self.angle)) * self.radius
        y = self.center[1] + math.sin(math.radians(self.angle)) * self.radius
        self.cur_pos[0] = x
        self.cur_pos[1] = y
        self.cur_pos[2] += delta_z
        return True

    def cylinder2world(self):
        '''
        Return the world coordinate of probe
        '''
        # world_pos = Vector((0, 0, 0))
        # world_rot = Euler((0, 0, 0), 'XYZ')
        # world_pos[2] = self.z
        # world_pos[0] = self.center[0] + math.cos(math.radians(self.angle)) * self.radius
        # world_pos[1] = self.center[1] + math.sin(math.radians(self.angle)) * self.radius
        # world_rot[2] = math.radians(self.angle - 450)
        # world_rot[0] = math.radians(self.x_rot)
        # world_rot[1] = math.radians(self.y_rot)
        world_pos = self.cur_rot.to_matrix() @ Vector((0, 0, 75)) + self.cur_pos
        return world_pos, self.cur_rot
    
    def reset(self):
        # random.seed(seed)
        # self.angle = INIT_ANGLE
        # self.z = INIT_Z
        # self.x_rot = INIT_X_ROT
        # self.y_rot = INIT_Y_ROT
        self.angle = INIT_ANGLE
        self.z_rot = 0
        self.z = INIT_Z
        self.x_rot = 0
        self.inverse = 1
        x = self.center[0] + math.cos(math.radians(self.angle)) * self.radius
        y = self.center[1] + math.sin(math.radians(self.angle)) * self.radius
        z = INIT_Z
        self.cur_pos = Vector((x, y, z))
        self.cur_rot = Euler((math.radians(90), math.radians(0), math.radians(-225)))

if __name__ == "__main__":
    import torch
    import torchvision
    print(torch.__version__)
    print(torch.version.cuda)
    print(torchvision.__version__)
        
