import os
import time
import torch
import struct

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from kmeans_tree import build_kmeans_tree
from bloom_filter import load_data, NUM_BITS, NUM_SLICES

class QueryDataset(Dataset):
    '''
    The query dataset is used to collate the query data.
    query_bloom_filters: (batch_size, NUM_SLICES * NUM_BITS)
    query_locs: (batch_size, 2)
    truth: (batch_size)

    '''
    def __init__(self, query_bloom_filters, query_locs, truths):
        super().__init__()
        self.query_bloom_filter_set = query_bloom_filters
        self.query_loc_raw = query_locs
        self.query_bloom_filters = []
        self.query_locs = []

        self.truths = truths

    def cuda(self):
        # we transfer all the bloom filters into cuda long tensors to save time
        for query_bloom_filter in tqdm(self.query_bloom_filter_set, desc='Transfering query bloom filters'):
            self.query_bloom_filters.append(torch.tensor(list(query_bloom_filter), dtype=torch.int16, device='cuda'))
        for loc in tqdm(self.query_loc_raw, desc='Transfering query locations'):
            self.query_locs.append(torch.tensor(loc, dtype=torch.float32, device='cuda'))
        return self
    
    def __len__(self):
        return len(self.query_bloom_filters)
    
    def __getitem__(self, query_idx):
        return query_idx, self.query_bloom_filters[query_idx], self.query_locs[query_idx], self.truths[query_idx]
    
    @staticmethod
    def collate_query_fn(batch):
        '''
            This function is used to collate the batch data into a single tensor.
            It used the sparse matrix to accelerate the transfer between cpu and gpu.
            Which is 20x faster than the previous version.
        '''
        query_idxs, query_bloom_filter_list, query_loc, truths = zip(*batch)

        batch_size = len(query_bloom_filter_list)

        # query shape: (batch_size, NUM_SLICES * NUM_BITS)
        query_col = torch.hstack(query_bloom_filter_list)
        query_row = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.int16, device='cuda'), torch.tensor([x.shape[0] for x in query_bloom_filter_list], device='cuda'))
        query_values = torch.ones_like(query_col, dtype=torch.float16, device='cuda')
        query_idx = torch.vstack([query_row, query_col])
        query_bloom_filters = torch.sparse_coo_tensor(query_idx, query_values, (batch_size, NUM_SLICES * NUM_BITS), check_invariants=False)
        query_bloom_filters._coalesced_(True)
        query_locs = torch.vstack(query_loc)

        return query_idxs, query_bloom_filters, query_locs, truths


class POIDataset:
    def __init__(self, dataset, batch_size, num_workers=0, load_query=True, portion=None, preprocess=False):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = os.path.join('data', dataset)
        self.data_bin_dir = os.path.join('data_bin', dataset)
        if not os.path.exists(self.data_bin_dir):
            os.makedirs(self.data_bin_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not preprocess:
            self.poi_bloom_filters, self.poi_locs, _ = self.build_or_load('poi')
            if load_query:
                train_bloom_filters, train_locs, train_truths = self.build_or_load('train', portion)
                dev_bloom_filters, dev_locs, dev_truths = self.build_or_load('dev')
                test_bloom_filters, test_locs, test_truths = self.build_or_load('test')

                self.train_dataset = QueryDataset(train_bloom_filters, train_locs, train_truths).cuda()
                self.dev_dataset = QueryDataset(dev_bloom_filters, dev_locs, dev_truths).cuda()
                self.test_dataset = QueryDataset(test_bloom_filters, test_locs, test_truths).cuda()
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=QueryDataset.collate_query_fn)
                self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=QueryDataset.collate_query_fn)
                self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=QueryDataset.collate_query_fn)
                self.infer_train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=QueryDataset.collate_query_fn)
        else:
            self.build_if_not_exist(portion=portion)

    def build_or_load(self, split, portion=None):
        assert split in ['train', 'dev', 'test', 'poi']
        if portion is not None:
            bin_file = os.path.join(self.data_bin_dir, f'portion/{split}_{portion}.bin')
        else:
            bin_file = os.path.join(self.data_bin_dir, f'{split}.bin')
        if os.path.exists(bin_file):
            bloom_filters, locs, truths = self.deserialize(bin_file)
        else:
            raw_file = os.path.join(self.dataset_dir, f'{split}.txt' if portion is None else f'portion/{split}_{portion}.txt')
            bloom_filters, locs, truths = load_data(raw_file, is_query=split != 'poi')
            self.serialize(bloom_filters, locs, bin_file, truths)
        return bloom_filters, locs, truths
    
    def build_if_not_exist(self, portion=None):
        for split in ['train', 'dev', 'test', 'poi']:
            if portion is not None:
                bin_file = os.path.join(self.data_bin_dir, f'portion/{split}_{portion}.bin')
            else:
                bin_file = os.path.join(self.data_bin_dir, f'{split}.bin')
            if not os.path.exists(bin_file):
                raw_file = os.path.join(self.dataset_dir, f'{split}.txt' if portion is None else f'portion/{split}_{portion}.txt')
                bloom_filters, locs, truths = load_data(raw_file, is_query=split != 'poi')
                self.serialize(bloom_filters, locs, bin_file, truths)
    
    @staticmethod
    def serialize(bloom_filters, locations, file_dir, truths):
        '''
        serialize the bloom filters and locations into a binary file.
        bloom filters are NUM_SLICES * NUM_BITS bits binary, locations are two 32-bit float.
        '''
        num_rows = len(bloom_filters)
        num_cols = NUM_SLICES * NUM_BITS
        if len(truths) > 0:
            assert len(truths) == num_rows
        if not os.path.exists(os.path.dirname(file_dir)):
            os.makedirs(os.path.dirname(file_dir))
        with open(file_dir, 'wb') as file:
            start = time.time()
            file.write(struct.pack('IIH', num_rows, num_cols, 0 if len(truths)==0 else 1))
            # The uint16 is used to store the bloom filter
            # The float64 is used to store the location
            # We try to make the read/write process as fast as possible.
            bloom_filter_lengths = [len(bloom_filter) for bloom_filter in bloom_filters]
            file.write(struct.pack(f'{num_rows}H', *bloom_filter_lengths))
            bloom_filter_data = []
            for bloom_filter in bloom_filters:
                bloom_filter_data.extend(bloom_filter)
            file.write(struct.pack(f'{sum(bloom_filter_lengths)}H', *bloom_filter_data))
            file.write(struct.pack(f'{num_rows * 2}d', *[x for loc in locations for x in loc]))
            if len(truths) > 0:
                for i in range(num_rows):
                    file.write(struct.pack('H', len(truths[i])))
                    for t in truths[i]:
                        file.write(struct.pack('I', t))
                file.write(struct.pack('dd', locations[i][0], locations[i][1]))
            print(f'Serializing {file_dir} takes {time.time() - start} seconds.')

    @staticmethod
    def deserialize(file_dir):
        '''
            deserialize the binary file into bloom filters and locations.
        '''
        with open(file_dir, 'rb') as file:
            start = time.time()
            num_rows, _, has_truth = struct.unpack('IIH', file.read(10))
            bloom_filter_lengths = struct.unpack(f'{num_rows}H', file.read(num_rows * 2))
            bloom_filter_data = struct.unpack(f'{sum(bloom_filter_lengths)}H', file.read(sum(bloom_filter_lengths) * 2))
            bloom_filters = [None] * num_rows
            start_idx = 0
            for row, bloom_filter_length in enumerate(tqdm(bloom_filter_lengths, desc='Constructing bloom filter set')):
                bloom_filters[row] = set(bloom_filter_data[start_idx:start_idx + bloom_filter_length])
                start_idx += bloom_filter_length
            locations = []
            for _ in range(num_rows):
                locations.append(struct.unpack('dd', file.read(16)))
            truths = []
            if has_truth == 1:
                for _ in range(num_rows):
                    num_truths = struct.unpack('H', file.read(2))[0]
                    truths.append(struct.unpack(f'{num_truths}I', file.read(num_truths * 4)))
            print(f'Deserializing {file_dir} takes {time.time() - start} seconds.')
            return bloom_filters, locations, truths


class NodeEncodeDataset(Dataset):
    '''
        This dataset is used to infer the node representations for the C++ inference engine.
    '''
    def __init__(self, levels):
        super().__init__()
        self.nodes = []
        for level in levels:
            self.nodes.extend(level)

    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        node = self.nodes[idx]
        if node.torch_bloom_filter is None:
            node.torch_bloom_filter = torch.tensor(list(node.bloom_filter), dtype=torch.int16, device='cuda')
        return node.torch_bloom_filter
    
    def collate_fn(self, batch):
        node_bloom_filter_list = batch
        node_bloom_filter = torch.hstack(node_bloom_filter_list)
        node_row = torch.repeat_interleave(torch.arange(len(node_bloom_filter_list), dtype=torch.int16, device='cuda'), torch.tensor([x.shape[0] for x in node_bloom_filter_list], device='cuda'))
        node_values = torch.ones_like(node_bloom_filter, dtype=torch.float16, device='cuda')
        node_idx = torch.vstack([node_row, node_bloom_filter])
        node_bloom_filter = torch.sparse_coo_tensor(node_idx, node_values, (len(node_bloom_filter_list), NUM_SLICES * NUM_BITS), check_invariants=False)
        node_bloom_filter._coalesced_(True)
        return node_bloom_filter
    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GeoGLUE_clean')
    parser.add_argument('--portion', type=str, default='1')

    args = parser.parse_args()
    dataset = args.dataset
    portion = args.portion
    portion = None if portion == '1' else portion

    poi_dataset = POIDataset(dataset, batch_size=0, num_workers=0, load_query=True, portion=portion, preprocess=True)

    # building BloomFilterTree in advance.
    if not os.path.exists(f'data_bin/{dataset}/tree.bin'):
        poi_locs = []
        with open(f'data/{dataset}/poi.txt', 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading POI locations'):
                line = line.strip().split('\t')
                x, y = float(line[1]), float(line[2])
                poi_locs.append([x, y])

        levels = build_kmeans_tree(dataset, poi_locs, width=8)
        with open(f'data_bin/{dataset}/tree.bin', 'wb') as f:
            f.write(struct.pack('I', len(levels)))
            for level in levels:
                f.write(struct.pack('I', len(level)))
                for cluster in level:
                    f.write(struct.pack('I', len(cluster)))
                    for node_id in cluster:
                        f.write(struct.pack('I', node_id))

    