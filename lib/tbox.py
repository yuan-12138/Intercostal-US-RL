import numpy as np
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

class TBox:
    def __init__(self, size, origin, original_size):
        self.size = size
        self.box = np.zeros((size, size, size, 3)) # When initialize, the box only contains bone
        self.origin = origin
        self.original_size = original_size
        self.num = 0
        self.tumor_num = 0
        self.smallest = np.ones(3)*100
        self.biggest = np.zeros(3)
        self.center = np.zeros(3)
        self.scanning_points = []
        self.vertices = [] # Vertices of tumors (for augmentation)
    
    def insert(self, point, feature):
        index = [0, 0, 0]
        # self.vertices.append([point[0], point[1], point[2]])
        for i in range(3):
            dis = point[i] - self.origin[i]
            index[i] = int(round(dis / self.original_size[i] * self.size))
            if index[i] < 0 or index[i] > self.size-1:
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(point),
                #                     rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
                # print(point)
                # print(feature)
                return
        # If not occupied, insert feature
        if sum(self.box[index[0]][index[1]][index[2]]) == 0:
            self.box[index[0]][index[1]][index[2]] = feature
            self.num += 1
            if feature[0] == 1:
                self.tumor_num += 1
                if index[0] < self.smallest[0] and index[1] < self.smallest[1] and index[2] < self.smallest[2]:
                    for i in range(3):
                        self.smallest[i] = index[i]
                if index[0] > self.biggest[0] and index[1] > self.biggest[1] and index[2] > self.biggest[2]:
                    for i in range(3):
                        self.biggest[i] = index[i]
                # print(self.box[index[0]][index[1]][index[2]])
                # print(Vector(self.origin)+4*Vector(index))
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                     rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
    
    def calc_center(self, tumor_id):
        objects = bpy.data.objects
        tumor = objects[tumor_id]
        bd = tumor.bound_box
        bd_x = [point[0] for point in bd]
        bd_y = [point[1] for point in bd]
        bd_z = [point[2] for point in bd]
        x_max = max(bd_x)
        x_min = min(bd_x)
        y_max = max(bd_y)
        y_min = min(bd_y)
        z_max = max(bd_z)
        z_min = min(bd_z)
        point = tumor.matrix_world @ Vector(((x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2)) # Center of bound box
        self.center = [point[0], point[1], point[2]]
        # for i in range(3):
        #     dis = point[i] - self.origin[i]
        #     self.center[i] = int(round(dis / self.original_size[i] * self.size))
    
    def show(self):
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    if sum(self.box[i][j][k]) != 0:
                        if self.box[i][j][k][0] == 1:
                            bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector([i,j,k]),
                                                rotation = Euler((0, 0, 0), 'XYZ'))
                            obj = bpy.context.object
                            # Create a material
                            mat = bpy.data.materials.new("Red")

                            # Activate its nodes
                            mat.use_nodes = True

                            mat.diffuse_color = (1,0,0,1)
                            obj.data.materials.append(mat)
                        # elif self.box[i][j][k][1] == 1:
                        #     bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector([i,j,k]),
                        #                         rotation = Euler((0, 0, 0), 'XYZ'))
                        #     obj = bpy.context.object
                        #     # Create a material
                        #     mat = bpy.data.materials.new("Blue")

                        #     # Activate its nodes
                        #     mat.use_nodes = True

                        #     mat.diffuse_color = (1,1,0,1)
                        #     obj.data.materials.append(mat)
                        
        # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.center),
        #                                      rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
    
    def augment(self, scale_x=1, scale_y=1, scale_z=1, offset=[0, 0, 0], rotation=(0, 0, 0)):
        offset = offset[0] * Vector([0.7, 0.7 ,0])+\
                offset[1] * Vector([0.7, -0.7 ,0])+\
                offset[2] * Vector([0, 0, 1])
        for v in self.vertices:
            # print(v)
            R = Euler(rotation)
            v = Vector(v)
            # offset = Vector(offset)
            center = Vector(self.center)
            v = R.to_matrix() @ (v-center) + center # Rotate
            v = center + Vector([(v-center)[0]*scale_x, (v-center)[1]*scale_y, (v-center)[2]*scale_z]) # Scale
            v = v + offset # Offset
            self.insert(v, np.array([1, 0, 0]))
            

    def scan(self, point):
        index = [0, 0, 0]
        for i in range(3):
            dis = point[i] - self.origin[i]
            index[i] = int(round(dis / self.original_size[i] * self.size))
            if index[i] < 0:
                return 0
            if index[i] > self.size - 1:
                return 0
        if self.box[index[0]][index[1]][index[2]][0] == 1:
            if self.box[index[0]][index[1]][index[2]][2] == 0:
                self.box[index[0]][index[1]][index[2]][2] = 1
                self.scanning_points.append(index)
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                     rotation = Euler((0, 0, 0), 'XYZ'))
                # bpy.ops.mesh.primitive_cube_add(size = 1, location = Vector(index) + Vector(offset),
                #                 rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
                self.tumor_num -= 1
                return 1
            return 0
        elif self.box[index[0]][index[1]][index[2]][1] == 1:
            if self.box[index[0]][index[1]][index[2]][2] == 0:
                self.box[index[0]][index[1]][index[2]][2] = 1
                self.scanning_points.append(index)
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                                 rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
            return -1
        else:
            if self.box[index[0]][index[1]][index[2]][2] == 0:
                self.box[index[0]][index[1]][index[2]][2] = 1
                self.scanning_points.append(index)
                self.num += 1
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                              rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
            return 0

    def after_scan(self, last_success = False):
        # print(self.scanning_points)
        for index in self.scanning_points:
            if self.box[index[0]][index[1]][index[2]][1] == 1:
                # If the point is bone, delete the US ray on it
                self.box[index[0]][index[1]][index[2]][2] = 0
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                 rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
            else:
                # If the point is not bone, then it means it is a scanned tumor point or a US ray point
                if last_success:
                    # If last scan is successful, delete the tumor point
                    self.box[index[0]][index[1]][index[2]] = np.zeros(3)
                else:
                    # Otherwise, only delete the US ray point
                    self.box[index[0]][index[1]][index[2]][1] = 0
                    self.box[index[0]][index[1]][index[2]][2] = 0
                # bpy.ops.mesh.primitive_cube_add(size = 4, location = Vector(self.origin)+4*Vector(index),
                #                 rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
        self.scanning_points = []