import bpy
import bmesh

class Skin:
    def __init__(self, object):
        self.object = object
        self.mesh_data = self.object.to_mesh()
    
    