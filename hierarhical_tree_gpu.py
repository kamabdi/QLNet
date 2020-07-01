from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import  KMeans
import numpy as np
from scipy.spatial import distance
import pickle
import torch
from torch.autograd import Variable

class Node:
    def __init__(self, parent, previous, n, c, centroid):
        self.parent = parent
        self.previous = previous # previous sibling (save parent, same depth)
        self.next = n  # next sibling
        self.child = None # first child
        self.centroid = centroid # clister center
        self.density = 0


class Stack:
    def __init__(self,depth, parent, data):
        self.data = data
        self.depth = depth
        self.parent = parent

def addNode(centroids, tree, parent, depth, node_id,stack, data, labels):
        n = np.shape(centroids)[0]

        for i in xrange(0, n):
            if i==0: # first node
                tree.append(Node(parent, None, (node_id+1), None, centroids[i]))
            elif i==n-1: # last node
                tree.append(Node(parent, node_id-1, None, None, centroids[i]))
            else:
                tree.append(Node(parent, node_id-1, node_id+1, None, centroids[i]))

            stack, density = splitData(stack, data, labels, depth,node_id , i)
            tree[node_id].density = density
            print 'Node id ' + str(node_id) + ' Density: ' + str(density)
            node_id = node_id+1
        return tree, stack, node_id

def splitData(stack, data, labels, depth, parent, i):
    clustered_data = data[labels==i]
    density = np.shape(clustered_data)[0]
    stack.append(Stack(depth+1, parent, clustered_data))

    return stack, density


def construct(data, n_cl, density, max_depth):
    tree = []
    stack = []
    node_id = 0 # to keep track of nodes in tree
    depth = 0 # to keep track of depth in the tree
    parent = None

    # 1. generate n root nodes by k-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_cl,batch_size=3000)
    kmeans.fit(data)
    y = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 2. Add root nodes to the tree
    tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data, y)
    # 3.
    while stack:
        data_subset = stack.pop()
        depth = data_subset.depth

        if np.shape(data_subset.data)[0] > density and depth < max_depth:
            parent = data_subset.parent
            #kmeans = KMeans(n_clusters=n_cl_2, n_jobs=-1)
            kmeans = MiniBatchKMeans(n_clusters=n_cl,batch_size=3000)
            kmeans.fit(data_subset.data)
            y = kmeans.labels_
            centroids = kmeans.cluster_centers_
            temp = node_id # id of a first child
            tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data_subset.data, y)
            tree[parent].child = temp

        else:
            continue

    return tree

def euclidian(a,b):
    return torch.sqrt(((a-b)**2).sum(-1))
#    return (torch.abs(a-b)).sum(-1)

def convolution(a,b):
    return torch.nn.functional.conv2d(a,b)

def cosine(a,b):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(a,b)

def measure_distance(current_node, data, tree,smallest_distance):
    nodes = tree
    dist = euclidian(data,nodes)
#    val, best_node = torch.max(dist,-1) for cosine
    val, best_node = torch.min(dist,-1)
    best_node = best_node.type(torch.IntTensor).cpu().numpy()
    return best_node, smallest_distance, nodes

def predict(tree, dataset):
    prediction = predict_image(tree, dataset)
    cent = prediction[0]
    ids = prediction[1]
    return  cent, ids


def predict_image(tree, data):
        current_node = 0
        best_node = current_node
        smallest_distance = 555555555

        best_node, smallest_distance, nodes = measure_distance(current_node, data, tree, smallest_distance)
        nodes = nodes.view(nodes.size()[1], nodes.size()[2])
        result = torch.stack([nodes[best_node[i]] for i in xrange (0, np.shape(best_node)[0])])
        return result, best_node


def save_tree(tree,filename):
    # open the file for writing
    fileObject = open(filename,'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(tree, fileObject)
    # here we close the fileObject
    fileObject.close()

def load_tree(filename):
    fileObject = open(filename,'r')
    # load the object from the file into var b
    tree = pickle.load(fileObject)
    return tree
