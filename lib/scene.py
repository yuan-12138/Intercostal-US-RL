import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform.rotation import Rotation

import os, sys
import time
import torch
import numpy as np
import copy
import pickle

dir = "lib"
if not dir in sys.path:
    sys.path.append(dir)
import probe
import octree
import importlib
import cylinder
import tbox
import statistics
importlib.reload(cylinder)
importlib.reload(probe)
importlib.reload(octree)
importlib.reload(tbox)

import yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        RENDER = config['RENDER']
except:
    RENDER = True

if RENDER:
    import cv2
    import bpy

from mathutils import Vector
from mathutils import Euler

# Initial pose of probe
INIT_T = Vector((75, 145, 172))
INIT_EULER = Euler((math.radians(90), 0, math.radians(-225)))
MAX_STEPS = 100
TUMOR_POINTS_NUM = 675

class Scene:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.full_tumor_tree = self.create_tumor_tree() # Full octree for tumor
        # tree = copy.deepcopy(self.full_tumor_tree)
        # self.save_tumors()
        file = 'tumor_{}.pickle'.format(6)
        path = os.path.join('tumors', file)
        try:
            with open(path, 'rb') as f:
                self.full_box = pickle.load(f)
        except:
            path = os.path.join(os.getcwd(), 'tumors', 'tumor_6.pickle')
            with open(path, 'rb') as f:
                self.full_box = pickle.load(f)
        # print(self.full_box.center)
        # self.full_box.augment(rotation=(-90, -90, 0), scale=1, offset = [15, 15, 15])
        # self.full_box.show()
        box = copy.deepcopy(self.full_box)
        # Add probe
        center_cynlinder = Vector((200, 268, 172))
        self.cylinder = cylinder.Cylinder(center_cynlinder, 200)
        probe_T, probe_euler = self.cylinder.cylinder2world()
        self.probe = probe.Probe(probe_T, probe_euler, 15, 300, 50, 150, 3, box)
        self.cur_steps = 0
        # self.seed = 41
        # np.random.seed(42)
        self.tumor_offset = Vector(np.array([0., 0., 0.]))
        self.tumor_dis = []
        self.tumor_percentage = []
        self.success_cnt = 0
        self.avg_dis = 0
        self.avg_percentage = 0
        self.reset()
    
    def save_tumors(self):
        for i in range(1, 21):
            for j in range(1, 9):
                if i == 1:
                    tumor_id = 'tumor'
                else:
                    tumor_id = 'tumor' + str(i)
                if j == 1:
                    bone_id = 'bone'
                else:
                    bone_id = 'bone' + str(j)
                self.full_box = self.create_box(tumor_id, bone_id)
                self.full_box.calc_center(tumor_id)
                file = 'tumor_{}-bone_{}.pickle'.format(i, j)
                path = os.path.join(os.getcwd(), 'tumors', file)
                with open(path, 'wb') as f:
                    pickle.dump(self.full_box, f)

    def render(self):
        '''
        Render the camera view
        '''
        image_path = os.path.join(os.getcwd(), "images/Image0163.png")
#        bpy.data.objects["probe"].hide_render = True
        bpy.ops.render.render() 
        img = cv2.imread(image_path) # reads an image in the BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        return img
    
    def setup_composite_nodes(self, img_type):
        '''
        Parameters:
        img_type: "depth": depth image
                  "rbg": rbg image
        '''
        bpy.data.objects["skin"].hide_render = True
        bpy.data.objects["liver"].hide_render = True
        self.setup_composite_nodes("rbg")   
        if img_type == "depth":
            bpy.context.scene.use_nodes = True
            tree = bpy.context.scene.node_tree
            # Clear default nodes
            for n in tree.nodes:
                tree.nodes.remove(n)
            links = tree.links
            rl = tree.nodes.new(type = 'CompositorNodeRLayers')
            norm = tree.nodes.new(type = 'CompositorNodeNormalize')
            viewer = tree.nodes.new(type = 'CompositorNodeViewer')
            comp = tree.nodes.new(type = 'CompositorNodeComposite')
            output_file = tree.nodes.new(type = 'CompositorNodeOutputFile')
            output_file.base_path = os.path.join(os.getcwd(), "images")
            links.new(rl.outputs[2], norm.inputs[0])
            links.new(norm.outputs[0], viewer.inputs[0])
            links.new(norm.outputs[0], comp.inputs[0])
            links.new(norm.outputs[0], output_file.inputs[0])
        else:
            bpy.context.scene.use_nodes = True
            tree = bpy.context.scene.node_tree
            # Clear default nodes
            for n in tree.nodes:
                tree.nodes.remove(n)
            links = tree.links
            rl = tree.nodes.new(type = 'CompositorNodeRLayers')
            viewer = tree.nodes.new(type = 'CompositorNodeViewer')
            comp = tree.nodes.new(type = 'CompositorNodeComposite')
            output_file = tree.nodes.new(type = 'CompositorNodeOutputFile')
            output_file.base_path = os.path.join(os.getcwd(), "images")
            links.new(rl.outputs[0], viewer.inputs[0])
            links.new(rl.outputs[0], comp.inputs[0])
            links.new(rl.outputs[0], output_file.inputs[0])
    
    def get_us_image(self, th=3, need_image = False):
        '''
        Save the simulated ultrasound image
        '''
        n = self.probe.simulate_us_image(need_image=need_image, th=th)
        return n
        
    def delete_probe(self):
        for obj in bpy.data.objects:
            obj.select_set(False)
        if "probe" in bpy.data.objects:
            bpy.data.objects["probe"].select_set(True)
            bpy.ops.object.delete() 
    
    def delete_camera(self):
        for obj in bpy.data.objects:
            obj.select_set(False)
        if "Camera" in bpy.data.objects:
            bpy.data.objects["Camera"].select_set(True)
            bpy.ops.object.delete() 
            
    def set_tissue_color(self):
        '''
        Set the material for the tissue
        '''
        # Check if material exists
        if bpy.data.materials.get("Bone") and bpy.data.materials.get("Tumor") and bpy.data.materials.get("Background"):
            pass
        else:
            # Material for bone
            mat_bone = bpy.data.materials.new("Bone")
            mat_bone.use_nodes = True
            tree = mat_bone.node_tree
            nodes = tree.nodes
            bsdf_bone = mat_bone.node_tree.nodes["Principled BSDF"]
            bsdf_bone.inputs["Base Color"].default_value = (1, 1, 0, 1)
            # Material for tumor
            mat_tumor = bpy.data.materials.new("Tumor")
            mat_tumor.use_nodes = True
            bsdf_tumor = mat_tumor.node_tree.nodes["Principled BSDF"]
            bsdf_tumor.inputs["Base Color"].default_value = (0, 1, 0, 1)
            # Material for probe
            mat_probe = bpy.data.materials.new("Background")
            mat_probe.use_nodes = True
            tree = mat_probe.node_tree
            nodes = tree.nodes
            bsdf_probe = mat_probe.node_tree.nodes["Principled BSDF"]
            bsdf_probe.inputs["Base Color"].default_value = (0, 0, 1, 1)
            # Set material
            bpy.data.objects['bone'].active_material = mat_bone
            bpy.data.objects['tumor'].active_material = mat_tumor
            bpy.data.objects['background'].active_material = mat_probe

    def create_box(self, tumor_id, bone_id, box_size = 30):
        '''
        Initialize the 3d box
        '''
        origin = [60, 107, 127] 
        original_size = [120, 120, 120]
        box = tbox.TBox(box_size, origin, original_size)
        tumor = bpy.data.objects[tumor_id]
        box.vertices = [([(tumor.matrix_world @ v.co)[0], 
                          (tumor.matrix_world @ v.co)[1], 
                          (tumor.matrix_world @ v.co)[2]]) for v in tumor.data.vertices]
        # s = time.time()
        # for co in coords:
        #     box.insert(co, np.array([1, 0, 0]))
        # e = time.time()
        # print(e-s)
        bone = bpy.data.objects[bone_id]
        coords = [(bone.matrix_world @ v.co) for v in bone.data.vertices]
        for co in coords:
            box.insert(co, np.array([0, 1, 0]))
        return box

    def create_tumor_box(self, box_size = 60, ply_path = "D:\\RobUS\\tumor_4.ply"):
        '''
        Initialize the 3d box for tumor
        '''
        reader = vtk.vtkPLYReader()
        reader.SetFileName(ply_path)
        reader.Update()
        voxels = np.array(reader.GetOutput().GetPoints().GetData())
        tumor = bpy.data.objects['tumor']
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
        center = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2] # Center of tumor bound box
        origin = [x_min, y_min, z_min]
        self.tumor_center = center
        self.tumor_origin = origin
        size = [0, 0, 0]
        size[0] = x_max - x_min + 2
        size[1] = y_max - y_min + 2
        size[2] = z_max - z_min + 2
        tumor_box = tbox.TBox(box_size, origin, size)
        for v in voxels:
            point = [v[0], v[1], v[2]]
            tumor_box.insert(point)
        return tumor_box

    def create_tumor_tree(self, ply_path = "D:\\RobUS\\tumor_4.ply"):
        '''
        Initialize the octree for tumor
        '''
        # Read the voxel of tumor
        reader = vtk.vtkPLYReader()
        reader.SetFileName(ply_path)
        reader.Update()
        voxels = np.array(reader.GetOutput().GetPoints().GetData())
        # print(tumor_voxel)
        # Get the diagonal length of tumor bounding box 
        tumor = bpy.data.objects['tumor']
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
        center = Vector(((x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2)) # Center of tumor bound box
        self.tumor_center = center
        dis = (Vector((bd[0][0], bd[0][1], bd[0][2]))-center).length
        size = dis*2
        # Save the points in octree
        tumor_tree = octree.Octree(center, size)
        # voxels = tumor_voxel.get_voxels()
        for v in voxels:
            point = [v[0], v[1], v[2]]
#            bpy.ops.mesh.primitive_cube_add(size = 8, location = Vector(point), rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
            tumor_tree.insert_point(tumor_tree.root, size, point)
        return tumor_tree
    
    # def get_unscanned_points(self):
    #     '''
    #     Return a list which contains all of the unscanned points of tumor
    #     '''
    #     return np.array(self.probe.tree.traverse(self.probe.tree.root), dtype='float32')

    def reset(self, mode = "train", tumor_size = 'all'):
        '''
        Reset the blender scene
        '''
        self.cur_steps = 0
        self.cylinder.reset()
        self.tumor_dis = []
        self.tumor_percentage = []
        self.tumor_offset = Vector([0., 0., 0.])
        T, euler_p2w = self.cylinder.cylinder2world()
        # tree = copy.deepcopy(self.full_tumor_tree)
        # self.probe.update_tree(tree)
        if mode == "train":
            tumor_id = np.random.randint(1, 16)
            bone_id = np.random.randint(1, 7)
            self.bone_id = bone_id
        elif mode == "test":
            tumor_id = np.random.randint(16, 21)
            bone_id = np.random.randint(7, 9)
            self.bone_id = bone_id
        else:
            raise ValueError("Invalid mode for resetting blender scene")
        if tumor_size == 'multi':
            tumor_num = np.random.randint(1, 4)
            if tumor_num == 2:
                interval = 20
                bias = 15
            elif tumor_num == 3:
                interval = 7
                bias = 15
        else:
            tumor_num = 1
        
        # print(id)
        file = 'tumor_{}-bone_{}.pickle'.format(tumor_id, bone_id)
        path = os.path.join('tumors', file)
        try:
            with open(path, 'rb') as f:
                self.full_box = pickle.load(f)
        except:
            path = os.path.join(os.getcwd(), 'tumors', file)
            with open(path, 'rb') as f:
                self.full_box = pickle.load(f)

        if tumor_size == 'all':
            a = 1.5
            b = 0.5
        elif tumor_size == 'small':
            a = 0.5
            b = 0.5
        elif tumor_size == 'medium':
            a = 0.5
            b = 1
        elif tumor_size == 'large':
            a = 0.5
            b = 1.5
        elif tumor_size == 'multi':
            a = 0
            b = 0
        else:
            raise ValueError("Invalid tumor size")
        
        scale_x = np.random.rand()*a+b 
        scale_y = np.random.rand()*a+b 
        scale_z = np.random.rand()*a+b 
        off_x = np.random.rand()*20 - 5 # [-5, 15]
        off_y = np.random.rand()*55 - 30 # [-30, 25]
        off_z = np.random.rand()*40 - 20 # [-20, 20]

        rot_x = np.random.rand()*360
        rot_y = np.random.rand()*360
        rot_z = np.random.rand()*360
        if tumor_size == 'multi':
            # print(self.full_box.scanning_points)
            if tumor_num == 1:
                scale_x = np.random.rand()*0.5+0.5 # [0.5, 1.5]
                scale_y = np.random.rand()*0.5+0.5 # [0.5, 1.5]
                scale_z = np.random.rand()*0.5+0.5 # [0.5, 1.5]
                off_x = np.random.rand()*20 - 5 # [-5, 15]
                off_y = np.random.rand()*55 - 30 # [-30, 25]
                off_z = np.random.rand()*40 - 20 # [-20, 20]
                rot_x = np.random.rand()*360
                rot_y = np.random.rand()*360
                rot_z = np.random.rand()*360
            else:
                # If there are multiple tumors, the scale of the tumors should be smaller and they should not overlap
                scale_x = np.random.rand() # [0, 1]
                scale_y = np.random.rand() # [0, 1]
                scale_z = np.random.rand() # [0, 1]
                off_x = np.random.rand()*20 - 5 # [-5, 15]
                off_y = np.random.rand()*interval - 30 # [-30, 25] 
                off_z = np.random.rand()*40 - 20 # [-20, 20]
                rot_x = np.random.rand()*360
                rot_y = np.random.rand()*360
                rot_z = np.random.rand()*360

        # print(scale, off_x, off_y, off_z, rot_x, rot_y, rot_z)

        self.full_box.augment(rotation=(rot_x, rot_y, rot_z),
                               scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, 
                               offset = [off_x, off_y, off_z])
        # self.full_box.show()
        # print(self.full_box.tumor_num)

        # Add multiple tumors
        if tumor_num > 1:
            for i in range(tumor_num-1):
                if mode == "train":
                    new_tumor_id = np.random.randint(1, 16)
                elif mode == "test":
                    new_tumor_id = np.random.randint(16, 21)
                else:
                    raise ValueError("Invalid mode for resetting blender scene")
                file = 'tumor_{}-bone_{}.pickle'.format(new_tumor_id, bone_id)
                path = os.path.join('tumors', file)
                try:
                    with open(path, 'rb') as f:
                        new_box = pickle.load(f)
                except:
                    path = os.path.join('D:/RobUS/tumors', file)
                    with open(path, 'rb') as f:
                        new_box = pickle.load(f)
                self.full_box.vertices = new_box.vertices
                scale_x = np.random.rand() # [1, 2]
                scale_y = np.random.rand() # [1, 2]
                scale_z = np.random.rand() # [1, 2]
                off_x = np.random.rand()*20 - 5 # [-5, 15]
                off_y = np.random.rand()*interval - 30 + (bias+interval)*(i+1) # [-30, 25]
                off_z = np.random.rand()*40 - 20 # [-20, 20]
                rot_x = np.random.rand()*360
                rot_y = np.random.rand()*360
                rot_z = np.random.rand()*360
                # print(scale, off_x, off_y, off_z, rot_x, rot_y, rot_z)
                self.full_box.augment(rotation=(rot_x, rot_y, rot_z),
                                        scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, 
                                        offset = [off_x, off_y, off_z])
        # print(self.full_box.tumor_num)
        box = copy.deepcopy(self.full_box)
        self.probe.update_box(box)
        self.probe.update_pose(T, euler_p2w)
        self.probe.adjust = True
        self.get_us_image()
    
    def encode_obs(self, render = False, video_id = 0):
        '''
        Encode the observation
        Return the box of scene and render image if needed
        '''
        # Visulization
        if render:
            img = self.render()
            path = os.path.join(os.getcwd(), 'images', 'video{}'.format(str(video_id)))
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = os.path.join(path, "Image%d.png"%(self.cur_steps))
            os.rename(path, file_name)

        box = np.array(np.transpose(self.probe.box.box, (3, 0, 1, 2)), dtype=np.float32)
        return box

    def move_probe(self, action, th = 3, w_dis = 0.8, w_percentage = 0.2):    
        '''
        Move the probe and scan the tumor
        params:
        action: action to take
        th: threshold for successful scan
        w_dis: weight for distance
        w_percentage: weight for coverage
        '''
        self.cur_steps += 1
        done = False
        if self.cur_steps >= 80:
            done = True
        # Move up
        if action == 0:
            if self.cylinder.move(delta_z=4):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Move down
        if action == 1:
            if self.cylinder.move(delta_z=-4):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Move left
        if action == 2:
            if self.cylinder.move(delta_angle=-3):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Move right
        if action == 3:
            if self.cylinder.move(delta_angle=3):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Rotate X axis clockwise
        if action == 4:
            if self.cylinder.move(delta_x_rot=-2):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Rotate X axis counter clockwise
        if action == 5:
            if self.cylinder.move(delta_x_rot=2):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Rotate Y axis clockwise
        if action == 6:
            if self.cylinder.move(delta_z_rot=-2):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Rotate Y axis counter clockwise
        if action == 7:
            if self.cylinder.move(delta_z_rot=2):
                T, p2w_euler = self.cylinder.cylinder2world()
                self.probe.update_pose(T, p2w_euler)
            else:
                return -1, True
        # Adjust the probe location
        if action == 8:
            self.probe.adjust = not self.probe.adjust

        # print(self.cur_steps, self.probe.adjust)

        tot_del_num, percentage, dis = self.get_us_image(th=th)
        if tot_del_num > 0:
            self.tumor_dis.append(dis)
            self.tumor_percentage.append(percentage)
            reward = tot_del_num/30*(1+w_dis*(100/dis)+w_percentage*percentage)
        elif self.probe.adjust:
            reward = 0
        else:
            reward = -0.3
        if action == 8:
            reward -= 1
        # print("Reward:{}".format(reward))
        if self.probe.box.tumor_num < self.full_box.tumor_num*0.05:
            if len(self.tumor_dis) and len(self.tumor_percentage):
                avg_dis = statistics.mean(self.tumor_dis)
                avg_percentage = statistics.mean(self.tumor_percentage)
                self.success_cnt += 1
                self.avg_dis = (self.avg_dis*(self.success_cnt-1)+avg_dis)/self.success_cnt
                self.avg_percentage = (self.avg_percentage*(self.success_cnt-1)+avg_percentage)/self.success_cnt
                reward = 10*(1+w_dis*(100/avg_dis)+w_percentage*avg_percentage)
                print("In this episode: Avg_dis:{} Avg_coverage:{}".format(avg_dis, avg_percentage))
                print("Total average: Avg_dis:{} Avg_coverage:{}".format(self.avg_dis, self.avg_percentage))
                print("Final Reward:{}".format(reward))
            else:
                reward = 10
            done = True
            # print(self.probe.box.tumor_num)
            # print(self.probe.box.tumor_num/self.full_box.tumor_num)
        return reward, done
    
    def get_points_left(self):
        return self.probe.box.tumor_num
        
if __name__ == "__main__":
    scene_blender = Scene()
    r = 0
    scene_blender.save_tumors()
#    scene_blender.reset()
#    scene_blender.probe.box.show()
    # for i in range(30):
    #     s = time.time()
    #     for j in range(1):
    #         action = np.random.randint(9)
    #         reward, done = scene_blender.move_probe(action)
    #         r += reward
    #     #        print(reward)
    #     #        reward, done = scene_blender.move_probe(5)
    #     #        r += reward
    #     #        print(reward)
    #     #        reward, done = scene_blender.move_probe(7)
    #     #        r += reward
    #         print("reward", reward)
    #     #       if not scene_blender.probe.is_pose_feasible():
    #     #           break
    #         scene_blender.encode_obs()
    #     #        scene_blender.get_us_image(need_image=True)
    #     e = time.time()
    #     print(e-s)
    # print(scene_blender.full_box.tumor_num)
    # print(scene_blender.probe.box.tumor_num)

