import bpy

class Camera:
    def __init__(self, T, c2w_euler, scale):
        '''
        Parameters:
        T: Location of camera
        c2w_euler: Rotation of camera
        scale: Scale of camera
        '''
        self.T = T
        self.c2w_euler = c2w_euler
        self.scale = scale
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_object = bpy.data.objects.new('Camera', camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        camera = bpy.data.objects['Camera']
        camera.location = T
        camera.rotation_euler = c2w_euler
        camera.scale = scale
        bpy.context.scene.camera = camera

if __name__ == "__main__":
    pass
        
        
        