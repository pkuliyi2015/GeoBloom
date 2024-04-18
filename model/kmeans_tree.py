'''
    This script dedicates to the construction of the spatial tree for all datasets.
    NOTE: GeoGLUE is heavily anonymized (it has 2.8M POIs but only roughly 5,000 unique locations), 
        so we don't use k-means to construct the tree for GeoGLUE when parent node number is larger than 6000.

'''
import time
import math

from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from typing import List

class TreeNode:
    '''
        This tree node class only contains the location information.
        It is used to construct the tree. When actually running the model, 
        we use a different TreeNode class that contains the bloom filter and node radius.
    '''
    def __init__(self, location):
        self.location: list = location
        if len(location) == 2:
            self.min_x = location[0]
            self.max_x = location[0]
            self.min_y = location[1]
            self.max_y = location[1]
        else:
            self.max_x: float = -1e9
            self.min_x: float = 1e9
            self.max_y: float = -1e9
            self.min_y: float = 1e9
        self.child: List[TreeNode] = []
        self.parent: TreeNode = None

    def __repr__(self):
        return f'Node: {self.location}'
    
    def add_child(self, child):
        self.child.append(child)
        child.parent = self
        self.max_x = max(self.max_x, child.max_x)
        self.min_x = min(self.min_x, child.min_x)
        self.max_y = max(self.max_y, child.max_y)
        self.min_y = min(self.min_y, child.min_y)
    
    def compute_radius(self):
        self.radius = math.sqrt((self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2) / 2
        self.location = [(self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2]

def build_kmeans_tree(dataset, poi_locs, width=8):
    leaf_nodes: List[TreeNode] = []
    num_pois = len(poi_locs)

    for i in range(num_pois):
        poi_node = TreeNode(poi_locs[i])
        leaf_nodes.append(poi_node)

    print('Constructing the tree via KMeans...')
    # if dataset is GeoGLUE, we do anti-anonymization
    if dataset == 'GeoGLUE':
        unique_locs = OrderedDict()
        for i, node in enumerate(leaf_nodes):
            loc = tuple(node.location)
            if loc not in unique_locs:
                unique_locs[loc] = []
            unique_locs[loc].append(i)

        # remove the locations with few nodes from the unique_locs
        unique_locs_few_nodes = OrderedDict()
        for loc, node_idxs in unique_locs.items():
            if len(node_idxs) < width // 2:
                unique_locs_few_nodes[loc] = node_idxs

        if len(unique_locs_few_nodes) > 0:

            for loc in unique_locs_few_nodes:
                del unique_locs[loc]

            coords = list(unique_locs.keys())
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            # if a location has no more than 8 nodes, we merge it with the closest location
            for loc, node_idxs in unique_locs_few_nodes.items():
                _, idx = tree.query(loc, k=1)
                closest_loc = coords[idx]
                unique_locs[closest_loc].extend(node_idxs)

        for loc, node_idxs in unique_locs.items():
            for node_idx in node_idxs:
                leaf_node = leaf_nodes[node_idx]
                leaf_node.max_x = loc[0]
                leaf_node.min_x = loc[0]
                leaf_node.max_y = loc[1]
                leaf_node.min_y = loc[1]
                leaf_node.location = loc

    levels = []
    cluster_dump = []
    prev_level = leaf_nodes
    while len(prev_level) > width * width:
        start = time.time()
        clusters = kmeans_cluster(dataset, prev_level, width)
        current_level = build_from_clusters(clusters, prev_level)
        levels.append(current_level)
        prev_level = current_level
        cluster_dump.append(clusters)
        end = time.time()
        print(f'Level with {len(current_level)} nodes constructed in {end - start:.3f}s')

    return cluster_dump

# Build the first level via merging the leaf nodes (sort by geohash)

def kmeans_cluster(dataset, nodes, width):

    if 'GeoGLUE' in dataset and len(nodes) > 40000:
        # If the dataset is GeoGLUE, we do special design to bypass the anonymization
        # First, if the nodes are far more than unique locations, we split the nodes at the same location.
        unique_locs = OrderedDict()
        for i, node in enumerate(nodes):
            loc = tuple(node.location)
            if loc not in unique_locs:
                unique_locs[loc] = []
            unique_locs[loc].append(i)

        # remove the locations with few nodes from the unique_locs
        unique_locs_few_nodes = OrderedDict()
        for loc, node_idxs in unique_locs.items():
            if len(node_idxs) < width // 2:
                unique_locs_few_nodes[loc] = node_idxs

        if len(unique_locs_few_nodes) > 0:

            for loc in unique_locs_few_nodes:
                del unique_locs[loc]

            coords = list(unique_locs.keys())
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            # if a location has no more than 8 nodes, we merge it with the closest location
            for loc, node_idxs in unique_locs_few_nodes.items():
                _, idx = tree.query(loc, k=1)
                closest_loc = coords[idx]
                unique_locs[closest_loc].extend(node_idxs)

        print(f'Unique locations: {len(unique_locs)}, nodes: {len(nodes)}')

        if len(nodes) / len(unique_locs) > 1.2:
            clusters = OrderedDict()
            for loc, node_idxs in unique_locs.items():
                # split the node_idxs into clusters
                while len(node_idxs) > width:
                    clusters[len(clusters)] = node_idxs[:width]
                    node_idxs = node_idxs[width:]
                if len(node_idxs) > 0:
                    if len(node_idxs) >= width // 2:
                        clusters[len(clusters)] = node_idxs
                    else:
                        clusters[len(clusters) - 1].extend(node_idxs)

            return list(clusters.values())

    coords = [x.location for x in nodes]
    kmeans = MiniBatchKMeans(n_clusters=len(nodes) // width, random_state=0, batch_size=1000, n_init='auto')
    labels = kmeans.fit_predict(coords)
    clusters = OrderedDict()
    for i in range(len(nodes)):
        if labels[i] not in clusters:
            clusters[labels[i]] = []
        clusters[labels[i]].append(i)
    
    return list(clusters.values())

def build_from_clusters(clusters, prev_level):
    current_level = []
    for cluster in clusters:
        new_node = TreeNode([])
        for node_id in cluster:
            new_node.add_child(prev_level[node_id])
        new_node.compute_radius()
        current_level.append(new_node)
    return current_level