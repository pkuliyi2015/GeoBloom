import os
import math
import torch
import struct

from kmeans_tree import build_kmeans_tree
from tqdm import tqdm, trange
from typing import List

from bloom_filter import NUM_BITS, NUM_SLICES

class TreeNode:
    def __init__(self, bloom_filter, location, child=None, parent=None, poi_idx=None):
        self.bloom_filter: set = bloom_filter
        self.location: list = location
        if len(location) == 2:
            self.max_x: float = location[0]
            self.min_x: float = location[0]
            self.max_y: float = location[1]
            self.min_y: float = location[1]
        else:
            self.max_x: float = -1e9
            self.min_x: float = 1e9
            self.max_y: float = -1e9
            self.min_y: float = 1e9

        self.radius: float = 0
        self.child: List[TreeNode] = child
        self.parent: TreeNode = parent
        self.poi_idx: int = poi_idx

        self.torch_bloom_filter: torch.Tensor = None
        self.torch_location: torch.Tensor = None
        self.torch_radius: torch.Tensor = None

    def add_child(self, child):
        self.bloom_filter.update(child.bloom_filter)
        if self.child is None:
            self.child = []
        self.child.append(child)
        child.parent = self
        self.max_x = max(self.max_x, child.max_x)
        self.min_x = min(self.min_x, child.min_x)
        self.max_y = max(self.max_y, child.max_y)
        self.min_y = min(self.min_y, child.min_y)
    
    def compute_radius(self):
        self.radius = math.sqrt((self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2) / 2
        self.location = [(self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2]

    def __repr__(self):
        return f'Node: {self.location}, depth: {self.depth}, radius: {self.radius}, bloom_filter: {len(self.bloom_filter)}'
    

class BloomFilterTree:
    def __init__(self, dataset, poi_bloom_filters, poi_locs, width=8) -> None:
        self.dataset = dataset
        self.levels: List[List[TreeNode]] = []
        self.leaf_nodes: List[TreeNode] = []
        self.width: int = width
        self.num_nodes: int = len(poi_bloom_filters)

        for i in range(len(poi_bloom_filters)):
            self.leaf_nodes.append(TreeNode(poi_bloom_filters[i], poi_locs[i], poi_idx=i))

        self.levels.insert(0, self.leaf_nodes)

        print('Building the bloom filter tree...')
        if not os.path.exists(f'data_bin/{dataset}/tree.bin'):
            cluster_by_layers = build_kmeans_tree(dataset, poi_locs, width=width)
            self.serialize(cluster_by_layers)
        else:
            cluster_by_layers = self.deserialize()

        prev_level = self.leaf_nodes
        for cluster in cluster_by_layers:
            current_level = self.build_from_clusters(cluster, prev_level)
            self.num_nodes += len(current_level)
            self.levels.insert(0, current_level)
            prev_level = current_level

        # Remove the first several levels
        if dataset == 'MeituanBeijing':
            first_level_width = 50
        elif dataset == 'MeituanShanghai':
            first_level_width = 200
        elif dataset == 'GeoGLUE':
            first_level_width = 4000
        elif dataset == 'GeoGLUE_clean':
            first_level_width = 1000
        else:
            raise NotImplementedError

        truncate_index = 0
        for i in range(len(self.levels)):
            if len(self.levels[i]) > first_level_width:
                truncate_index = i
                break
        self.levels = self.levels[truncate_index:]
        self.depth = len(self.levels)
        self.init_candidates = self.levels[0]
        for node in self.init_candidates:
            node.parent = None

        # Transfer to gpu to save time (Abandoned as we always infer nodes for nnue engine)
        # if 'Meituan' in dataset:
        tbar = tqdm(total=self.num_nodes, desc='Moving bloom filters to GPU')

        def move_to_gpu(node: TreeNode):
            node.torch_bloom_filter = torch.tensor(list(node.bloom_filter), dtype=torch.int16, device='cuda')
            node.torch_location = torch.tensor(node.location, dtype=torch.float32, device='cuda') 
            node.torch_radius = torch.tensor(node.radius, dtype=torch.float32, device='cuda')
            tbar.update(1)
            if node.child is not None:
                for child in node.child:
                    move_to_gpu(child)

        for node in self.init_candidates:
            move_to_gpu(node)
        tbar.close()

        self.init_bloom_filter, self.init_loc, self.init_radius = self.collate_nodes(self.init_candidates)

    def compute_idf_vec(self):
        idf_vec = torch.zeros((NUM_BITS * NUM_SLICES), device='cuda')
        for node in self.leaf_nodes:
            idf_vec[node.torch_bloom_filter.long()]+=1
        idf_vec = torch.log(len(self.leaf_nodes)/(idf_vec + 1))
        return idf_vec

    def serialize(self, levels):
        with open(f'data_bin/{self.dataset}/tree.bin', 'wb') as f:
            f.write(struct.pack('I', len(levels)))
            for level in levels:
                f.write(struct.pack('I', len(level)))
                for cluster in level:
                    f.write(struct.pack('I', len(cluster)))
                    for node_id in cluster:
                        f.write(struct.pack('I', node_id))

    def deserialize(self):
        with open(f'data_bin/{self.dataset}/tree.bin', 'rb') as f:
            num_levels = struct.unpack('I', f.read(4))[0]
            levels = []
            for _ in range(num_levels):
                num_clusters = struct.unpack('I', f.read(4))[0]
                clusters = []
                for _ in range(num_clusters):
                    num_nodes = struct.unpack('I', f.read(4))[0]
                    cluster = []
                    for _ in range(num_nodes):
                        node_id = struct.unpack('I', f.read(4))[0]
                        cluster.append(node_id)
                    clusters.append(cluster)
                levels.append(clusters)
            return levels
    
    def build_from_clusters(self, clusters, prev_level):
        current_level = []
        for cluster in tqdm(clusters):
            new_node = TreeNode(set(), [], [])
            for node_id in cluster:
                new_node.add_child(prev_level[node_id])
            new_node.compute_radius()
            current_level.append(new_node)
        return current_level


    def get_truth_path(self, poi_idx):
        current_node = self.leaf_nodes[poi_idx]
        path = [current_node]
        while current_node.parent is not None:
            path.insert(0, current_node.parent)
            current_node = current_node.parent
        return path
    
    def collate_nodes(self, node_list: List[TreeNode]):
        node_bloom_filter_list = [node.torch_bloom_filter for node in node_list]
        node_col = torch.hstack(node_bloom_filter_list)
        node_row = torch.repeat_interleave(torch.arange(len(node_bloom_filter_list), dtype=torch.int16, device='cuda'), torch.tensor([x.shape[0] for x in node_bloom_filter_list], device='cuda'))
        node_values = torch.ones_like(node_col, dtype=torch.float16, device='cuda')
        node_idx = torch.vstack([node_row, node_col])
        node_bloom_filter = torch.sparse_coo_tensor(node_idx, node_values, (len(node_bloom_filter_list), NUM_SLICES * NUM_BITS), check_invariants=False)
        node_bloom_filter._coalesced_(True)
        # collate locations
        node_loc_list = [node.torch_location for node in node_list]
        node_loc = torch.vstack(node_loc_list)
        # collate node radius
        node_radius_list = [node.torch_radius for node in node_list]
        node_radius = torch.vstack(node_radius_list)
        return node_bloom_filter, node_loc, node_radius
    
    def load_candidates(self, file_path, num_rows, beam_widths, topk=20):
        '''
            This function is used to deserialize the candidates from the C++ inference engine.
            The bin file: [query_id][depth][beam_width], unsigned int32
            candidates: [depth][query_id][beam_width]
        '''
        if not isinstance(beam_widths, list):
            beam_widths = [beam_widths] * (self.depth + 1)
        candidates = [[[] for i in range(num_rows)] for j in range(self.depth - 1)]
        topk_list = [[] for _ in range(num_rows)]
        with open(file_path, 'rb') as f:
            for query_id in trange(num_rows, desc='Deserializing candidates'):
                for depth in range(1, self.depth):
                    beam_width = beam_widths[depth]
                    for node_idx in struct.unpack(f'{beam_width}I', f.read(beam_width * 4)):
                        candidates[depth-1][query_id].append(self.levels[depth][node_idx])
                topk_list[query_id] = list(struct.unpack(f'{topk}I', f.read(topk * 4)))
        return candidates, topk_list
