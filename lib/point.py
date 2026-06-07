import bpy
from mathutils import Vector
import mathutils
import numpy as np

class Point:
    def __init__(self, loc, feat):
        self.loc = loc
        self.feat = feat
    