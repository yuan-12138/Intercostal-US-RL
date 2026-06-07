import yaml
import os   
# from mathutils import Vector
# import mathutils
import numpy as np

dir = "lib"

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        RENDER = config['RENDER']
except:
    RENDER = True

# This defines the maximum objects an LeafNode can hold, before it gets subdivided again.
MAX_OBJECTS_PER_CUBE = 10

class OctNode:
    def __init__(self, position, size, data):
        '''
        Parameters:
            position: Position of the node
            size: Size of the node
            data: Points data contained in the node
        '''
        self.position = position
        self.size = size

        # All OctNodes will be leaf nodes at first
        # Then subdivided later as more objects get added
        self.is_leaf_node = True

        # store our object, typically this will be one, but maybe more
        self.data = data
        
        # might as well give it some emtpy branches while we are here.
        self.branches = [None, None, None, None, None, None, None, None]

        # The cube's bounding coordinates -- Not currently used
        self.ldb = (position[0] - (size / 2), position[1] - (size / 2), position[2] - (size / 2))
        self.ruf = (position[0] + (size / 2), position[1] + (size / 2), position[2] + (size / 2))
        

class Octree:
    def __init__(self, center, world_size):
        '''
        Parameters:
           center: Center of the object
           world_size: Size of the whole object 
        '''
        # Init the world bounding root cube
        # all world geometry is inside this
        # it will first be created as a leaf node (ie, without branches)
        # this is because it has no objects, which is less than MAX_OBJECTS_PER_CUBE
        # if we insert more objects into it than MAX_OBJECTS_PER_CUBE, then it will subdivide itself.
        self.root = self.add_node(center, world_size, [])
        self.world_size = world_size
        self.num = 0 # Number of points

    def add_node(self, position, size, data):
        # This creates the actual OctNode itself.
        return OctNode(position, size, data)
    
    def insert_point(self, root, size, point):
        self.num += 1
        self.insert_node(root, size, root, point)
    
    def insert_node(self, root, size, parent, point):
        if root == None:
            # we're inserting a single object, so if we reach an empty node, insert it here
            # Our new node will be a leaf with one object, our object
            # More may be added later, or the node maybe subdivided if too many are added
            # Find the Real Geometric centre point of our new node:
            # Found from the position of the parent node supplied in the arguments
            pos = parent.position
            # offset is halfway across the size allocated for this node
            offset = size / 2
            # find out which direction we're heading in
            branch = self.find_branch(parent, point)
            # new center = parent position + (branch direction * offset)
            new_center = [0,0,0]
            if branch == 0:
                # left down back
                new_center = [pos[0] - offset, pos[1] - offset, pos[2] - offset]
                
            elif branch == 1:
                # left down forwards
                new_center = [pos[0] - offset, pos[1] - offset, pos[2] + offset]
                
            elif branch == 2:
                # right down forwards
                new_center = [pos[0] - offset, pos[1] + offset, pos[2] - offset]
                
            elif branch == 3:
                # right down back
                new_center = [pos[0] - offset, pos[1] + offset, pos[2] + offset]

            elif branch == 4:
                # left up back
                new_center = [pos[0] + offset, pos[1] - offset, pos[2] - offset]

            elif branch == 5:
                # left up forward
                new_center = [pos[0] + offset, pos[1] - offset, pos[2] + offset]
                
            elif branch == 6:
                # right up forward
                new_center = [pos[0] + offset, pos[1] + offset, pos[2] - offset]

            elif branch == 7:
                # right up back
                new_center = [pos[0] + offset, pos[1] + offset, pos[2] + offset]
            # Now we know the centre point of the new node
            # we already know the size as supplied by the parent node
            return self.add_node(new_center, size, [point])
        
        #else: are we not at our position, but not at a leaf node either
        elif root.position != point and root.is_leaf_node == False:
            
            # we're in an octNode still, we need to traverse further
            branch = self.find_branch(root, point)
            # Find the new scale we working with
            new_size = root.size / 2
            # Perform the same operation on the appropriate branch recursively
            root.branches[branch] = self.insert_node(root.branches[branch], new_size, root, point)
        # else, is this node a leaf node with objects already in it?
        elif root.is_leaf_node:
            # We've reached a leaf node. This has no branches yet, but does hold
            # some objects, at the moment, this has to be less objects than MAX_OBJECTS_PER_CUBE
            # otherwise this would not be a leafNode (elementary my dear watson).
            # if we add the node to this branch will we be over the limit?
            if len(root.data) < MAX_OBJECTS_PER_CUBE:
                # No? then Add to the Node's list of objects and we're done
                root.data.append(point)
                #return root
            elif len(root.data) == MAX_OBJECTS_PER_CUBE:
                # Adding this object to this leaf takes us over the limit
                # So we have to subdivide the leaf and redistribute the objects
                # on the new children. 
                # Add the new object to pre-existing list
                root.data.append(point)
                # copy the list
                points_list = root.data
                # Clear this node's data
                root.data = None
                # Its not a leaf node anymore
                root.is_leaf_node = False
                # Calculate the size of the new children
                new_size = root.size / 2
                # distribute the objects on the new tree
                # print "Subdividing Node sized at: " + str(root.size) + " at " + str(root.position)
                for p in points_list:
                    branch = self.find_branch(root, p)
                    root.branches[branch] = self.insert_node(root.branches[branch], new_size, root, p)
        return root

    def find_position(self, root, position):
        # Basic collision lookup that finds the leaf node containing the specified position
        # Returns the child objects of the leaf, or None if the leaf is empty or none
        if root == None:
            return None
        elif root.is_leaf_node:
            return root
        else:
            branch = self.find_branch(root, position)
            return self.find_position(root.branches[branch], position)
            

    def find_branch(self, root, position):
        # returns an index corresponding to a branch
        # pointing in the direction we want to go
        vec1 = root.position
        vec2 = position
        result = 0
        # Binary code 
        # XYZ
        for i in range(3):
            if vec2[i] >= vec1[i]:
                result += 2**(2-i)
        return result
    
    def del_point(self, root, point, offset, box):
        '''
        Delete all the points around the given point
        Return the number of deleted points
        '''
        dis_th = 3
        del_num = 0
        root = self.find_position(root, [point[0]-offset[0], point[1]-offset[1], point[2]-offset[2]])
        if not root:
            return False
        for p in root.data:
            p1 = np.array(p)
            point1 = np.array(point)
            if np.linalg.norm(p1 - point1) < dis_th:
                # root.data.remove(p)
                n = box.scan(p1)
                # bpy.ops.mesh.primitive_cube_add(size = 1, location = Vector(point1), scale = [8, 8, 8])
                # bpy.ops.mesh.primitive_cube_add(size = 8, location = Vector(p1) + Vector(offset), rotation = mathutils.Euler((0, 0, 0), 'XYZ'))
                self.num -= n 
                del_num += n
        return del_num
    
    def traverse(self, root):
        res = []
        if root == None:
            return []
        if root.is_leaf_node:
            return root.data
        else:
            for b in root.branches:
                res.extend(self.traverse(b))
            return res 
        
if __name__ == "__main__":
    center = [0, 0, 0]
    size = 8
    tree = Octree(center, 8)
    tree.insert_node(tree.root, 8, tree.root, [1, 1, 0])
    tree.insert_node(tree.root, 8, tree.root, [1, 1, -1])
    tree.insert_node(tree.root, 8, tree.root, [1, 1, -3])
#    tree.insert_node(tree.root, 8, tree.root, [1, 1, 1])
    print(tree.find_position(tree.root, [1, 1, 0]))
    print(tree.find_position(tree.root, [1, 1, -1]))
#    print(tree.find_position(tree.root, [1, 1, -3]]
    print(tree.find_position(tree.root, [1, 1, 1]))
