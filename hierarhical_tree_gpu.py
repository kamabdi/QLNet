#from sklearn.cluster import MiniBatchKMeans
#from sklearn.cluster import  KMeans
#from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import distance
import pickle
import torch
from torch.autograd import Variable
#from kmeans_pytorch import kmeans, kmeans_predict
import sys


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

        for i in range(0, n):
            if i==0: # first node
                tree.append(Node(parent, None, (node_id+1), None, centroids[i]))
            elif i==n-1: # last node
                tree.append(Node(parent, node_id-1, None, None, centroids[i]))
            else:
                tree.append(Node(parent, node_id-1, node_id+1, None, centroids[i]))

            stack, density = splitData(stack, data, labels, depth,node_id , i)
            tree[node_id].density = density
            print('Node id ' + str(node_id) + ' Density: ' + str(density))
            node_id = node_id+1
        return tree, stack, node_id

def splitData(stack, data, labels, depth, parent, i):
    # clustered_data = data[labels==i]
    clustered_data = data[np.where(labels==i)[0], :]
    # print(clustered_data.shape)
    density = np.shape(clustered_data)[0]
    stack.append(Stack(depth+1, parent, clustered_data))

    return stack, density


def construct_sklearn(data, n_cl, density, max_depth):
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

def construct_alternative(data, n_cl, density, max_depth):
    from sklearn.cluster import AgglomerativeClustering
    tree = []
    stack = []
    node_id = 0 # to keep track of nodes in tree
    depth = 0 # to keep track of depth in the tree
    parent = None

    # 1. generate n root nodes by k-means clustering
    # kmeans = DBSCAN(eps=0.5, min_samples=30).fit(data)
    kmeans = AgglomerativeClustering(linkage='ward', n_clusters=10).fit(data)
    y = kmeans.labels_
    # centroids = kmeans.components_
    centroids = kmeans.labels_
    analyze_disctance(centroids, y, n_cl, data)

    # 2. Add root nodes to the tree
    tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data, y)
    # # 3.
    # while stack:
    #     data_subset = stack.pop()
    #     depth = data_subset.depth
    #
    #     if np.shape(data_subset.data)[0] > density and depth < max_depth:
    #         parent = data_subset.parent
    #         #kmeans = KMeans(n_clusters=n_cl_2, n_jobs=-1)
    #         kmeans = MiniBatchKMeans(n_clusters=n_cl,batch_size=3000)
    #         kmeans.fit(data_subset.data)
    #         y = kmeans.labels_
    #         centroids = kmeans.cluster_centers_
    #         temp = node_id # id of a first child
    #         tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data_subset.data, y)
    #         tree[parent].child = temp
    #
    #     else:
    #         continue

    return tree

def construct_pytorch(data, n_cl, density, max_depth):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tree = []
    stack = []
    node_id = 0 # to keep track of nodes in tree
    depth = 0 # to keep track of depth in the tree
    parent = None
    print(data.shape)

    # 1. generate n root nodes by k-means clustering
    data = torch.from_numpy(data)
    y, centroids = kmeans(X=data, num_clusters=n_cl, distance='euclidean', device=device)
    centroids = centroids.numpy()

    # y = kmeans_predict(data, centroids, 'euclidean', device=device)


    # 2. Add root nodes to the tree
    tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data, y)
    # 3.
    while stack:
        data_subset = stack.pop()
        depth = data_subset.depth

        if np.shape(data_subset.data)[0] > density and depth < max_depth:
            parent = data_subset.parent
            # kmeans = MiniBatchKMeans(n_clusters=n_cl,batch_size=3000)
            # kmeans.fit(data_subset.data)
            # y = kmeans.labels_
            # centroids = kmeans.cluster_centers_
            # data = torch.from_numpy(data_subset.data)
            y, centroids = kmeans(X=data_subset.data, num_clusters=n_cl, distance='euclidean', device=device)
            centroids = centroids.numpy()
            temp = node_id # id of a first child
            tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data_subset.data, y)
            tree[parent].child = temp

        else:
            continue

    return tree

def get_centers(num_centroids, dist, cluster_id):
    """
    Find the most central molecule in each cluster
    :param num_centroids: number of cluster centers
    :param dist: list of distances from the cluster center
    :param cluster_id: list of cluster identifiers
    :return:
    """
    nearest = [[sys.float_info.max, -1]] * num_centroids
    for i, (d, c) in enumerate(zip(dist, cluster_id)):
        if d < nearest[c][0]:
            nearest[c] = [d, i]
    return nearest

def analyze_disctance(D, y, n_cl, data):
    from sklearn.manifold import TSNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    for i in range(0, n_cl):
        ids = np.where(y==i)[0]
        # print("-------------> Cluster %d"%i)
        # print("Mean %.3f"%np.mean(D[ids]))
        # print("Min %.3f"%np.min(D[ids]))
        # print("Max %.3f"%np.max(D[ids]))
        # print("Median %.3f"%np.median(D[ids]))
        # print("Number of samples %d"%len(D[ids]))
        # hist, bin_edges = np.histogram(D[ids], [0,2,4,5,6,6.5,7,7.5,8,8.5,9,9.3,9.6,9.9,10.2,10.4,1bins=40)
        hist, bin_edges = np.histogram(D[ids], bins=10)
        print(hist)
        print(bin_edges)

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # Y = TSNE(n_components=2).fit_transform(data)
    Y = PCA(n_components=2).fit_transform(data)
    # Y = SpectralEmbedding(n_components=2).fit_transform(data)
    fig, ax = plt.subplots()

    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap=cmap)
    # plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=cmap)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Clusters")
    ax.add_artist(legend1)
    # plt.legend()
    plt.show()


def construct(data, n_cl, density, max_depth):
    import faiss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tree = []
    stack = []
    node_id = 0 # to keep track of nodes in tree
    depth = 0 # to keep track of depth in the tree
    parent = None
    print(data.shape)
    use_gpu=False

    # 1. generate n root nodes by k-means clustering
    kmeans = faiss.Kmeans(data.shape[1], n_cl, spherical=True, gpu=use_gpu, niter=100, verbose=True)

    kmeans.train(data)
    D, y = kmeans.index.search(data, 1)
    centroids = kmeans.centroids
    # analyze_disctance(D, y, n_cl, data)
    print("Level 1 is DONE")

    # 2. Add root nodes to the tree
    tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data, y)
    # 3.
    while stack:
        data_subset = stack.pop()
        depth = data_subset.depth

        if np.shape(data_subset.data)[0] > density and depth < max_depth:
            parent = data_subset.parent
            # print(data_subset.data[0])
            kmeans = faiss.Kmeans((data_subset.data).shape[1], n_cl, spherical=True, gpu=use_gpu, niter=100, verbose=True)
            kmeans.train(data_subset.data)
            D, y = kmeans.index.search(data_subset.data, 1)
            centroids = kmeans.centroids
            # analyze_disctance(D, y, n_cl, data_subset.data)

            temp = node_id # id of a first child
            tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data_subset.data, y)
            tree[parent].child = temp
        else:
            continue

    return tree

def construct_analyze(data, n_cl, density, max_depth):
    import faiss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tree = []
    stack = []
    node_id = 0 # to keep track of nodes in tree
    depth = 0 # to keep track of depth in the tree
    parent = None
    print(data.shape)

    # 1. generate n root nodes by k-means clustering
    kmeans = faiss.Kmeans(data.shape[1], n_cl, spherical=True, gpu=True, niter=100, verbose=True)
    kmeans.train(data)
    D, y = kmeans.index.search(data, 1)
    centroids = kmeans.centroids
    analyze_disctance(D, y, n_cl, data)
    print("Level 1 is DONE")
    # d = D.shape[1]
    # index = faiss.IndexFlatL2 (d)
    # index.add(centroids)
    # D, I = index.search(kmeans.centroids, 15)

    # max_cluster = n_cl
    # for i in range(0, max_cluster):
    #     ids = np.where(y==i)[0]
    #     median = np.median(D[ids])
    #     ids2 = np.where(D[ids]>=median)[0]
    #     y[ids2]=max_cluster
    #     max_cluster = max_cluster+1
    #     analyze_disctance(D, y, max_cluster, data)
    #     if max_cluster>100:
    #         break

    # 2. Add root nodes to the tree
    tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data, y)
    # 3.
    while stack:
        data_subset = stack.pop()
        depth = data_subset.depth

        if np.shape(data_subset.data)[0] > density and depth < max_depth:
            parent = data_subset.parent
            # print(data_subset.data[0])
            kmeans = faiss.Kmeans((data_subset.data).shape[1], n_cl, spherical=True, gpu=True, niter=100, verbose=True)
            kmeans.train(data_subset.data)
            D, y = kmeans.index.search(data_subset.data, 1)
            centroids = kmeans.centroids
            # analyze_disctance(D, y, n_cl, data_subset.data)

            temp = node_id # id of a first child
            tree,stack,node_id = addNode(centroids, tree, parent, depth, node_id, stack, data_subset.data, y)
            tree[parent].child = temp
        else:
            continue


    nodes = np.asarray([tree[i].centroid for i in range (0, np.shape(tree)[0])])
    print(len(nodes))
    d = data.shape[1]

    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexFlat(d, metric)
    index.add(nodes)
    D, y = index.search(data, 1)
    print(y)
    analyze_disctance(D, y, len(nodes), data)

    return tree

def euclidian(a,b):
    # print(b[:,:,:10].shape)
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
        result = torch.stack([nodes[best_node[i]] for i in range (0, np.shape(best_node)[0])])
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
