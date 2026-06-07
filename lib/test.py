import math
# import mathutils
from mathutils import Vector
from mathutils import Euler
from mathutils import Matrix

if __name__ == "__main__":
    # v = Vector([1, 2, 3])
    # e = Euler([1, 2, 3])
    # m = Matrix(R.from_euler('xyz', [1, 2, 3], degrees=False))
    # v[0] = 2
    # print(v)
    # print(e.to_matrix().to_euler().v)
    # print(m.to_euler().v)
    # print(type(e), type(v))
    # print(e.to_matrix() @ v)
    # print(m @ e.to_matrix())
    v = Vector((0, 0, 0))
    e = Vector((0, 0, 1))
    print(v+e)