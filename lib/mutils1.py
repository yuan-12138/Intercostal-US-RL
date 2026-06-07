from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform.rotation import Rotation
import numpy as np

class Vector():
    def __init__(self, v:np.ndarray):
        self.v = np.array(v)
    
    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, value):
        self.v[key] = value
    
    def __add__(self, other):
        if isinstance(other, (Vector)):
            return Vector(self.v + other.v)
        else:
            raise TypeError("Cannot add Vector with {}".format(type(other)))
    
    def __sub__(self, other):
        if isinstance(other, (Vector)):
            return Vector(self.v - other.v)
        else:
            raise TypeError("Cannot subtract Vector with {}".format(type(other)))
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.v * other)
        else:
            raise TypeError("Cannot multiply Vector with {}".format(type(other)))
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.v * other)
        else:   
            raise TypeError("Cannot multiply Vector with {}".format(type(other)))
    
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Vector(other.rot.apply(self.v))
        elif isinstance(other, Euler):
            return Vector(other.rot.apply(self.v))
        else:
            raise TypeError("Cannot multiply Vector with {}".format(type(other)))
    
    def __str__(self) -> str:
        return str(self.v) 
    
class Euler():
    def __init__(self, v, order='xyz'):
        order = order.lower()
        if isinstance(v, Rotation):
            self.rot = v
            self.v = v.as_euler('xyz', degrees=False)
        elif isinstance(v, np.ndarray):
            self.v = v
            self.rot = R.from_euler('xyz', v, degrees=False)
        elif isinstance(v, tuple):
            self.v = np.array(v)
            self.rot = R.from_euler('xyz', v, degrees=False)
        elif isinstance(v, list):
            self.v = np.array(v)
            self.rot = R.from_euler('xyz', v, degrees=False)
        else:
            raise TypeError("Cannot create Euler from {}".format(type(v)))
    
    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, value):
        self.v[key] = value
        self.rot = R.from_euler('xyz', v, degrees=False)

    def __matmul__(self, other):
        if isinstance(other, Euler):
            return Euler(self.rot * other.rot)
        elif isinstance(other, Vector):
            return Vector(self.rot.apply(other.v))
        else:
            raise TypeError("Cannot multiply Euler with {}".format(type(other)))
    
    def to_matrix(self):
        return Matrix(self.rot)
    
    def __str__(self) -> str:
        return str(self.v)

class Matrix():
    def __init__(self, m):
        if isinstance(m, Rotation):
            self.rot = m
            self.m = m.as_matrix()
        elif isinstance(m, np.ndarray):
            self.m = m
            self.rot = R.from_matrix(m)
        else:
            raise TypeError("Cannot create Matrix from {}".format(type(m)))
    
    def __getitem__(self, key):
        return self.m[key]
    
    def __setitem__(self, key, value):
        self.m[key] = value
        self.rot = R.from_matrix(m)
    
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.rot * other.rot)
        elif isinstance(other, Vector):
            return Vector(self.rot.apply(other.v))
        else:
            raise TypeError("Cannot multiply Matrix with {}".format(type(other)))
    
    def to_euler(self):
        return Euler(self.rot.as_euler('xyz', degrees=False))

    def __str__(self) -> str:
        return str(self.m)

if __name__ == '__main__':
    v = Vector([1, 2, 3])
    e = Euler([1, 2, 3])
    m = Matrix(R.from_euler('xyz', [1, 2, 3], degrees=False))
    v[0] = 2
    print(v)
    print(e.to_matrix().to_euler().v)
    print(m.to_euler().v)
    print(e @ v)
    print(m @ e.to_matrix())