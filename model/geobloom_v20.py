'''
    Developing logs:

    v1: The GeoBloom search engine should be a lightweight MLP that retrieves relevant geographic objects.
        - It accepts two bloom filters as input, one is from the query, the other is from the tree-node.
        - When the not trained, it should output the dot product of the two bloom filters exactly.
    
    v3: Works well on the MeituanBeijing dataset, but fails on GeoGLUE
    
    v6: Try to use Listwise loss (lambdarank)
        - Works well on MeituanBeijing
        - Interaction layer works as expected
        - Distance reweighter leads to overfitting (removed in v6 temporarily)
        
    v10 - Try the separated query rewriter
        - Separating the reranking and the retrieval process.
        MeituanBeijing NDCG@5: 0.605
        GeoGLUE beam_width = 2000: 0.468

    v12 - Remove the query rewriter as it negatively affect the result.
        - Special distance function design.
        
        I'm happy to announce that the model have achieved the best performance on all datasets.
        - However, the testing requires too much time. The reason is GPU-CPU bloom filter transfer and node collecting.
        - I noticed the Pikafish chess engine which may help.


    v14: Quantization & C++ inference engine.
        In this version, the model is trained in an "Eval -> Train -> Eval loop".
        In the future, the evaluation is expected to be done by the C++ inference engine.

        1. Try to use a unified model to do all the retrieval parts and a unique head to do the reranking part.
        2. Normalize the output of input encoding layer (avgpool) to unify the model at different depths.
        3. Use ClippedReLU instead of ReLU and limit the parameters in the optimizer.
           - Verified successfully on both MeituanBeijing and GeoGLUE.

        4. Implement the "Eval -> Train -> Eval loop" and test the performance on MeituanBeijing.
           - Verified successfully on MeituanBeijing.

        5. Ranking performance improvement
            - Truncate ranking loss to k=50 at the final depth
            - Add params d to the distance function
            - Separate the retrieval and reranking head and bottleneck.
            All effective.

    v15: Use KMeans to constuct the tree index; Use special tree design for GeoGLUE to bypass the anonymization.

        NOTE: So far, the best performance requires the following tuning:
        -   Consider node radius when computing query-node distance
        -   Listwise loss (lambdaLoss) for both retrieval and reranking
        -   Concatenate query and node embedding, then normalized by their bits
        -   Large input head and output head, LeakyReLU(x) + 1 activation
        -   TextSim = [a (TextScore - mean) / std + b].sigmoid(), a = 1, b = 0 before training.
            DistSim = -log(Distance + 1)
            Score = m * DistSim + n * TextSim - DistSim * TextSim, m = 1, n = 0 before training.
        -   Separate the retrieval and reranking layer. 
            Use full lambdaLoss for retrieval, truncated lambdaLoss (scheme=lambdaRank, k=50) for reranking.
        -   The average number of child should be as close as possible to the width of the tree.
            Use KMeans to build the Meituan tree index.
            Use the unique locations to build GeoGLUE's tree index.
            Use large inference beam width for GeoGLUE (the training width is fixed to 2000)
        -   When training the model, mix nodes from different depths within each mini-batch.

        v15 will be the last pure python version. We will implement the C++ inference engine in the next version.


    v16 - C++ inference engine.
        We now implement the C++ inference engine and test on all the datasets. The python part only includes the training part.

        Effective speed-up techniques:

        - Pre-embed the node bloom filters into 32-dim int32 vectors and store in the tree.
        - Embed the incomming query bloom filters into 32-dim int32 vectors. When searching, just do add and cliprelu.
        - Chunk the output of l3 into 8 parts and use 8 leakyrelu, so that AVX2 can be used for fast inference.
        - Pruning bloom filter common bits at the same depth.
        - Heap-sort based beam search.
        
        The python part consists of:
        1. POI / query serializer that writes the POI / query bloom filters and their locations into data_bin/{dataset}/poi.bin and {split}.bin file,
        2. Tree serializer that writes the tree index into data_bin/{dataset}/tree.bin file,
        3. GeoBloom writer functions that quantize the model parameters and write them into nnue/{dataset}_v{VERSION}.nnue file.
        4. Node encoder that encodes and quantizes the node bloom filters for the C++ inference engine (following the tree structure).

        The 1st and 2nd part replaces the previous pickle files. We store all files into data_bin folder, which is shared by the C++ inference engine.

        Besides,
        - Changed the torch bloom filters from torch.int16 to torch.int16 to save VRAMs.

        The remaining problems of quantization includes:
        1. The overflow of the encoder layer for queries (note that we have up to 1024 bits in one query).
            - Solved by int32 input weights. Note that each query will only pass through it once,
                so it shouldn' be an overhead (if it is, we reduce the hidden dim later).
            - As we will finally divide everything by the number of bits, this will have nearly no impact on the final result.
        2. The final reweight layer & LeakyReLU implementation's fast inference.
            - Solved by __mm256_maddubs_epi16 intrinsic function & >> 7 bit shift per intersection element.
        TODO: Add contextual information to the model (after C++ engine works)

        Results: 
        It works well on MeituanBeijing and MeituanShanghai. 
            - Very fast (< 1ms each query); 
            - Much better than all methods.
        But it doesn't work well on GeoGLUE. 
            - Slow (30ms each query).
            - Only slightly better than best baseline; Much worse than Ziqi's method.

        We will first try to improve the model performance on GeoGLUE.

    v17 - Residual head.
        We add a residual head to the model to handle non-overlapping queries and further
        enhance the model's discriminative power to those similar queries (hard negatives).
        In this version, we use a shared head for both retrieval and reranking.
        The residual head will only add a negligible computation cost to the C++ inference engine.

        Minor changes:
        - Remove the relu in the final score computation as it does harm to the model performance.
        - Add a negative truncation to the recall loss to slightly improve the recall performance.

        The result shows that the residual head is consistently effective on all datasets.
            - Performance on GeoGLUE is still not satisfactory.
            - It may because there are too many false positives in the shallow layers 
            (We shouldn't punish them).
        
        We now have three options: 
        1. Continue to improve the model performance and speed on GeoGLUE.
            The mean challenge is to deal with the anonymization (Fake POIs).
            We can try to build a noise-robust model.
        2. Change a dataset.

    v18 - Context POIs
        We hereby add the context POIs to the model using the techniques found in context/geobloom_v18_rank_context_trial4.py.
        Some problems may occur in the recalling process, include:
            1. Initially, proper context POIs may not exist in the beam at the leaf layer. We may need to ensure their existence.
            2. Similarly, we may also need to ensure that the context's parent nodes are in the beam. This may be achieved by 
                1) adjusting the ground truth scores of the context to be the second largest score in the beam.
                2) imposing a separate loss on the context parent scores, and during the inference, expand the beam to include the context parent.

        In this first version, we just naively assume that the context and their parents are all in the beam, and will share all context scores and weights
        across all depths. We will try to improve this later if the performance is not as satisfactory as in v18_rank_context_trial4.

    v19 - Split all heads for different layers.
        It seems that the bottleneck is important. Simply splitting the heads for different layers doesn't work as good as v18.
        The result shows that the bottleneck is important. We should use different bottleneck for different layers.
        
    
    v20 - Inverse Bit Frequencies and Paper Submission.

        Experimental Results (* means the best result): We use the whole set of Meituan-Beijing and Meituan-Shanghai dataset and supplement a 
        GeoGLUE_clean dataset.

        Directly run the code on an 10900K within this repo, we should get the following results (the v19 is reported in the paper):

        Unsupervised Performance:





'''
import os
import time
import math
import random
import torch
import struct
import subprocess

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from bloom_filter import NUM_BITS, NUM_SLICES
from bloom_filter_tree import BloomFilterTree
from lambdarank import lambdaLoss
from dataset import NodeEncodeDataset, POIDataset

# On MeituanBeijing and MeituanShanghai, all user selections are equally important, there is no ranking priority.
# On GeoGLUE and GeoGLUE_clean, there is only one ground truth for each query.
# Hence, the following utility function can calculate NDCG faster than torch_metrics while with identical results.

def fast_ndcg(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

VERSION = '19'

# 4*8192 = -32767 ~ 32768 can be exactly stored with torch.int16, so it saves VRAM
# And we can compare with other methods conveniently as they all uses fp16

# Model definition
class GeoBloom(nn.Module):
    def __init__(self, depth=4, quantized_one=127.0, weight_scale=64.0, division_factor=16.0, d_threshold=1000.0):
        super(GeoBloom, self).__init__()
        self.depth = depth
        self.quantized_one = quantized_one
        self.weight_scale = weight_scale
        self.d_threshold = d_threshold

        # We use one common encoder for all depths and both queries and nodes
        # This encoder is especially large to ensure the model's capacity.
        self.encoder = nn.Linear(NUM_SLICES * NUM_BITS, 256)

        self.bottleneck_list = [
            nn.Linear(512, 32) for _ in range(depth)
        ]
        self.bottleneck = nn.ModuleList(self.bottleneck_list)

        # We use experts to handle different depths when reweighting bloom filter's intersections.
        # Note: these decoders are serialized into int8 (i.e. 64x smaller than encoder), so they are tiny and won't largely affect the size.
        # But for fair comparison on disk usage with future baselines, we will write a separate function to do int16 quantization.
        # NOTE: The torch.compile has a bug that it can't handle the non-constant index for nn.ModuleList. 
        # Hence we use the python list, but with nn.ModuleList for model.parameters().
        self.rank_list = []
        for i in range(depth):
            self.rank_list.extend([
                nn.Linear(32, 32),
                nn.Linear(32, NUM_SLICES * NUM_BITS, bias=False),
            ])
        self.rank = nn.ModuleList(self.rank_list)

        # Exactly the same structure as the decoder. It is possible to simplify the model by sharing some weights, but we leave this to future work.
        self.context_select_list = []
        for i in range(depth):
            self.context_select_list.extend([
                nn.Linear(32, 32),
                nn.Linear(32, NUM_SLICES * NUM_BITS, bias=False),
            ])

        self.context_select = nn.ModuleList(self.context_select_list)

        self.context_rank_list = []
        for i in range(depth):
            self.context_rank_list.extend([
                nn.Linear(32, 32),
                nn.Linear(32, NUM_SLICES * NUM_BITS, bias=False),
            ])
        self.context_rank = nn.ModuleList(self.context_rank_list)

        # The residual head. Similar to the context head, we use a separate head for each depth, but dim=1 for outputs.
        self.residual_list = []
        for i in range(depth):
            self.residual_list.extend([
                nn.Linear(32, 32),
                nn.Linear(32, 1, bias=False),
            ])
        self.residual = nn.ModuleList(self.residual_list)
        self.leaky_relu = nn.LeakyReLU(1.0 / division_factor)

        # The division factor must be a power of 2.
        # So that we can use bit shift instead of division to ensure the speed on the C++ side.

        # Distance Modeling
        self.a = nn.Linear(depth, 1, bias=False)
        self.b = nn.Linear(depth, 1, bias=False)
        self.c = nn.Linear(depth, 1, bias=False)
        self.d = nn.Linear(depth, 1, bias=False)

        # v20: Inverse document frequency (IDF) for Bloom filter bits.
        self.idf_vec = nn.Linear(1, NUM_SLICES * NUM_BITS, bias=False)

        # Initialize all decoder weights to 0.0 at the last layer, i.e., Zero-Projection
        for i in range(depth):
            nn.init.zeros_(self.rank[2 * i + 1].weight)
            nn.init.zeros_(self.context_select[2 * i + 1].weight)
            nn.init.zeros_(self.context_rank[2 * i + 1].weight)
            nn.init.zeros_(self.residual[2 * i + 1].weight)

        nn.init.ones_(self.a.weight)
        nn.init.zeros_(self.b.weight)
        nn.init.ones_(self.c.weight)
        nn.init.ones_(self.d.weight)
        
        # When not used, the IDF should be all ones.
        nn.init.ones_(self.idf_vec.weight)

        weight_bound = quantized_one / weight_scale
        self.weight_bound = weight_bound
        self.weight_clipping = [
            {'params' : [self.d.weight], 'min_weight' : 0, 'max_weight' : 127 },
        ]

        for i in range(depth):
            self.weight_clipping.append(
                {'params' : [self.bottleneck[i].weight], 'min_weight' : -weight_bound, 'max_weight' : weight_bound }
            )

        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(2 * depth):
                self.weight_clipping.append(
                    {'params' : [layer[i].weight], 'min_weight' : -weight_bound, 'max_weight' : weight_bound }
                )
    
    @torch.no_grad()
    def set_idf_vec(self, idf_vec):
        self.idf_vec.weight[:,0] = idf_vec

    @torch.no_grad()
    def quantize_test(self):
        '''
            This function is used to test the accuracy loss of quantization.
        '''
        quantize_encoder = lambda x: x.mul(self.quantized_one).round()/self.quantized_one
        quantize8 = lambda x: torch.clamp(x, min=-self.weight_bound, max=self.weight_bound).mul(self.weight_scale).round()/self.weight_scale
        quantize32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round()/(self.quantized_one * self.weight_scale)

        self.encoder.weight.data = quantize_encoder(self.encoder.weight.data)
        self.encoder.bias.data = quantize_encoder(self.encoder.bias.data)

        for i in range(self.depth):
            self.bottleneck[i].weight.data = quantize8(self.bottleneck[i].weight.data)
            self.bottleneck[i].bias.data = quantize32(self.bottleneck[i].bias.data)

        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(self.depth):
                layer[2 * i].weight.data = quantize8(layer[i][0].weight.data)
                layer[2 * i].bias.data = quantize32(layer[i][0].bias.data)
                layer[2 * i + 1].weight.data = quantize8(layer[i][1].weight.data)
    

    @torch.inference_mode()
    def serialize(self, path, size_test=False):
        '''
            This function is used to quantize and serialize the model parameters.
        '''

        if not size_test:
            # In production environment inference, we follows the NNUE quantization scheme, which yields slightly larger model file (~44 MB)
            serializer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int32).flatten().cpu().numpy().tobytes()
            serializer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int8).flatten().cpu().numpy().tobytes()
            serializer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int32).flatten().cpu().numpy().tobytes()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float32).tobytes()
        else:
            # In size test, we use all 16-bit format to compare with other baselines in fp16 (~40MB)
            # NOTE: This is only for theoretical comparison. the resulting model is not only larger but also can't correctly.
            serializer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float16).tobytes()
            
        buf = bytearray()
        # header: tree depth
        buf.extend(struct.pack('H', self.depth))
        # encoder layer
        buf.extend(serializer_encoder(self.encoder.weight.T))
        buf.extend(serializer_encoder(self.encoder.bias))

        # bottleneck
        for i in range(self.depth):
            buf.extend(serializer8(self.bottleneck[i].weight[:,:256]))

        # all heads
        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(self.depth):
                buf.extend(serializer8(layer[2 * i].weight))
                buf.extend(serializer32(layer[2 * i].bias))
                buf.extend(serializer8(layer[2 * i + 1].weight))

        # the parameter a, b, c, d
        buf.extend(serializer_float(self.a.weight))
        buf.extend(serializer_float(self.b.weight))
        buf.extend(serializer_float(self.c.weight))
        buf.extend(serializer_float(self.d.weight))

        # the idf vec
        buf.extend(serializer32(self.idf_vec.weight))

        # write the binary buffer into file
        with open(path, 'wb') as f:
            f.write(buf)

    def clip_weights(self):
        for group in self.weight_clipping:
            for p in group['params']:
                p_data_fp32 = p.data
                min_weight = group['min_weight']
                max_weight = group['max_weight']
                p_data_fp32.clamp_(min_weight, max_weight)
                p.data.copy_(p_data_fp32)

    def get_lr(self):
        lr_list = [
            {'params': self.encoder.parameters(), 'lr' : 1e-3},
            {'params': self.a.parameters(), 'lr': 1e-3},
            {'params': self.b.parameters(), 'lr': 1e-3},
            {'params': self.c.parameters(), 'lr': 1e-3},
            {'params': self.d.parameters(), 'lr': 1e-3},
        ]
        
        for i in range(self.depth - 1):
            lr_list.append({'params': self.bottleneck[i].parameters(), 'lr': 1e-3})
        lr_list.append({'params': self.bottleneck[-1].parameters(), 'lr': 0.002})

        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(2 * (self.depth - 1)):
                lr_list.append({'params': layer[i].parameters(), 'lr': 1e-3})
            # The last layer is the final reranking depth. We use a larger learning rate for it for faster convergence, 
            # as it is has relatively fewer training samples than retrieval stage.
            lr_list.append({'params': layer[-2].parameters(), 'lr': 0.002})
            lr_list.append({'params': layer[-1].parameters(), 'lr': 0.002})
        return lr_list

    @autocast()
    def forward(self, query_bloom_filter, node_bloom_filter, query_loc, node_loc, node_radius, depth):
        
        query_bits = torch.sum(query_bloom_filter, dim=-1, keepdim=True)
        node_bits = torch.sum(node_bloom_filter, dim=-1, keepdim=True)
        
        query_embedding = self.encoder(query_bloom_filter) / query_bits
        node_embedding = self.encoder(node_bloom_filter) / node_bits

        query_embedding = torch.clamp(query_embedding, min=0, max=1)
        node_embedding = torch.clamp(node_embedding, min=0, max=1)

        hidden = torch.cat([query_embedding.unsqueeze(1).expand_as(node_embedding), node_embedding], dim=-1)
        hidden = self.bottleneck_list[depth](hidden)
        hidden = torch.clamp(hidden, min=0, max=1)

        # Chunked LeakyReLU:
        # We split the final layer into chunks to support SIMD acceleration.
        # This will lead to significant speed increase while doesn't affect the accuracy.
        num_chunks = 8

        hidden_rank = self.rank_list[2 * depth](hidden)
        hidden_rank = torch.clamp(hidden_rank, min=0, max=1)
        hidden_rank_splits = torch.chunk(hidden_rank, num_chunks, dim=-1)
        weight_rank_splits = torch.chunk(self.rank_list[2 * depth + 1].weight.T, num_chunks, dim=0)
        hidden_rank = [self.leaky_relu(h_split.matmul(w_split)) for h_split, w_split in zip(hidden_rank_splits, weight_rank_splits)]
        hidden_rank = sum(hidden_rank) + self.idf_vec.weight.reshape(1, 1, -1)
        del hidden_rank_splits, weight_rank_splits
        intersection = torch.mul(query_bloom_filter.unsqueeze(1), node_bloom_filter)
        text_score = torch.sum(hidden_rank * intersection, dim=-1)
        del hidden_rank

        hidden_context_select = self.context_select_list[2 * depth](hidden)
        hidden_context_select = torch.clamp(hidden_context_select, min=0, max=1)
        hidden_context_select_splits = torch.chunk(hidden_context_select, num_chunks, dim=-1)
        weight_context_select_splits = torch.chunk(self.context_select_list[2 * depth + 1].weight.T, num_chunks, dim=0)
        hidden_context_select = [F.relu(h_split.matmul(w_split) + 1e-6) for h_split, w_split in zip(hidden_context_select_splits, weight_context_select_splits)]
        hidden_context_select = sum(hidden_context_select) + self.idf_vec.weight.reshape(1, 1, -1)
        del hidden_context_select_splits, weight_context_select_splits
        context_select_score = torch.sum(hidden_context_select * intersection, dim=-1)
        final_context_select_score = context_select_score
        del intersection, hidden_context_select
       
        # Context selection
        with torch.no_grad():
            node_dist = self.node_pairwise_distances(node_loc)
            mask = node_dist < self.d_threshold
            context_select_score = context_select_score.unsqueeze(1).expand_as(node_dist)
            context_select_score = context_select_score.masked_fill(~mask, -1e9)
            best_context_idx = torch.argmax(context_select_score, dim=-1)
            # best_context_idx: [batch_size, num_nodes]
            context_bloom_filter = torch.gather(node_bloom_filter, dim=1, index=best_context_idx.unsqueeze(-1).expand_as(node_bloom_filter))
            # residual matching: (query - query * node) * context = query * (context - context * node) = query * (context - node).relu(). The last formular is faster.
            context_bloom_filter = (context_bloom_filter - node_bloom_filter).relu()
            context_intersection = torch.mul(query_bloom_filter.unsqueeze(1), context_bloom_filter)
        
        del query_bloom_filter, node_bloom_filter, context_bloom_filter

        hidden_context_rank = self.context_rank_list[2 * depth](hidden)
        hidden_context_rank = torch.clamp(hidden_context_rank, min=0, max=1)
        hidden_context_rank_splits = torch.chunk(hidden_context_rank, num_chunks, dim=-1)
        weight_context_rank_splits = torch.chunk(self.context_rank_list[2 * depth + 1].weight.T, num_chunks, dim=0)
        hidden_context_rank = [self.leaky_relu(h_split.matmul(w_split)) for h_split, w_split in zip(hidden_context_rank_splits, weight_context_rank_splits)]
        hidden_context_rank = sum(hidden_context_rank)

        context_score = torch.sum(hidden_context_rank * context_intersection, dim=-1)
        del hidden_context_rank, context_intersection

        res_hidden = self.residual_list[2 * depth](hidden)
        res_hidden = torch.clamp(res_hidden, min=0, max=1)
        res_score = self.residual_list[2 * depth + 1](res_hidden)

        # We hereby add the residual head to handle the non-overlapping queries (i.e. intersection = 0)
        # Please be aware that this head ranges from [-64, 64] but intersection ranges from [0, query_bits].
        # We assume query bits is less than 1024, so we rescale the residual head to [-1024, 1024], i.e., multiply it by 16
        # NOTE: we can't rescale desc_sim by query bits; this will lead to instability.

        desc_sim = text_score + res_score.squeeze(-1) * 16 + context_score

        desc_std, desc_mean = torch.std_mean(desc_sim, dim=-1, keepdim=True)
        desc_sim = (desc_sim - desc_mean) / (desc_std + 1e-5)

        dist = torch.sum((query_loc.unsqueeze(1) - node_loc) ** 2, dim=-1).sqrt()
        dist = (dist - node_radius.squeeze(-1)).relu()

        dist_sim = - torch.log(dist + 1)
        final_score = (self.c.weight[0][depth] - (self.a.weight[0][depth] * desc_sim + self.b.weight[0][depth]).sigmoid()) * (dist_sim - self.d.weight[0][depth])

        return final_score, final_context_select_score
    
    
    @staticmethod
    def node_pairwise_distances(node_loc):
        # node_loc shape: [batch_size, num_nodes, 2]
        # Expand node_loc to shape [batch_size, num_nodes, 1, 2]
        node_loc_expanded = node_loc.unsqueeze(2)
        # Compute squared differences in both dimensions (x and y)
        diff_squared = (node_loc_expanded - node_loc_expanded.transpose(1, 2)) ** 2
        # Sum over the last dimension (x and y) and take the square root
        distances = torch.sqrt(diff_squared.sum(-1))

        return distances

    @torch.inference_mode()
    @autocast()
    def encode_node(self, node_bloom_filter, depth):
        if node_bloom_filter.is_sparse:
            node_bloom_filter = node_bloom_filter.to_dense()
        node_bits = torch.sum(node_bloom_filter, dim=-1, keepdim=True)
        node_embedding = self.encoder(node_bloom_filter) / node_bits
        node_embedding = torch.clamp(node_embedding, min=0, max=1)
        node_embedding = torch.matmul(node_embedding, self.bottleneck[depth].weight.T[256:])
        node_embedding = node_embedding + self.bottleneck[depth].bias
        return node_embedding
    


def collate_query_candidates(tree, query_idxs, src_candidates):               
    candidate_bloom_filters = []
    candidate_node_locs = []
    candidate_node_radius = []
    for query_id in query_idxs:
        candidate_nodes = src_candidates[query_id]
        bloom_filter, loc, radius = tree.collate_nodes(candidate_nodes)
        candidate_bloom_filters.append(bloom_filter)
        candidate_node_locs.append(loc)
        candidate_node_radius.append(radius)
    candidate_bloom_filters = torch.stack(candidate_bloom_filters)
    candidate_node_locs = torch.stack(candidate_node_locs)
    candidate_node_radius = torch.stack(candidate_node_radius) 
    return candidate_bloom_filters, candidate_node_locs, candidate_node_radius


def prepare_target(tree, depth, dataloader, candidates, ensure_truth=False):
    target_col = []
    target_row = []
    for batch in dataloader:
        query_idxs, _, _, truth = batch
        for i, query_idx in enumerate(query_idxs):
            truth_nodes = set()
            for poi_idx in truth[i]:
                truth_nodes.add(tree.get_truth_path(poi_idx)[depth])
            for j, node in enumerate(candidates[query_idx]):
                if node in truth_nodes:
                    target_row.append(query_idx)
                    target_col.append(j)
                    truth_nodes.remove(node)
                    if len(truth_nodes) == 0:
                        break
            if ensure_truth and len(truth_nodes) > 0:
                tail_index = len(candidates[query_idx]) - len(truth_nodes)
                candidates[query_idx] = candidates[query_idx][:tail_index] + list(truth_nodes)
                for k in range(len(truth_nodes)):
                    target_row.append(query_idx)
                    target_col.append(tail_index + k)

    target = torch.sparse_coo_tensor(
                torch.vstack([torch.tensor(target_row, dtype=torch.int32, device='cuda'), 
                                torch.tensor(target_col, dtype=torch.int32, device='cuda')]), 
                torch.ones_like(torch.tensor(target_col, dtype=torch.float16, device='cuda')), 
                (len(dataloader.dataset), len(candidates[0])),
                check_invariants=False)
    target._coalesced_(True)
    return target

def common_bit_threshold(query_bloom_filter, node_bloom_filter):
    common_bit = 0
    for query_bit in query_bloom_filter:
        if query_bit in node_bloom_filter:
            common_bit += 1
        if common_bit >= NUM_SLICES:
            return True
    return False

def train(model: GeoBloom, optimizer, tree: BloomFilterTree, train_beam_width, infer_beam_width, max_epochs, train_dataloader, infer_train_dataloader, dev_dataloader, ensure_context_in_beam=False, portion=None):
    # Train by layer
    train_candidates = [[] for _ in range(tree.depth)]
    train_candidates[0] = [tree.init_candidates] * len(train_dataloader.dataset)

    current_train_beam_width = train_beam_width[0] if isinstance(train_beam_width, list) else train_beam_width
    init_train_resample = len(train_candidates[0][0]) > current_train_beam_width
    train_batch_size = train_dataloader.batch_size
    init_train_bloom_filters, init_train_node_locs, init_train_node_radius = tree.init_bloom_filter.to_dense().unsqueeze(0).expand(train_batch_size,-1,-1), tree.init_loc.unsqueeze(0).expand(train_batch_size, -1, -1), tree.init_radius.unsqueeze(0).expand(train_batch_size, -1, -1)    
    max_metrics = [0] * tree.depth

    # We apply recall truncation on GeoGLUE as it is too noisy.
    retrieve_loss = lambda pred, truth: lambdaLoss(pred, truth, k=100, reduction='sum')
    rank_loss = lambda pred, truth: lambdaLoss(pred, truth, weighing_scheme='lambdaRank_scheme', k=30, reduction='sum')
    context_loss = lambda pred, truth: lambdaLoss(pred, truth, k=30, reduction='sum')
    
    ckpt_path = f'ckpt/{tree.dataset}_geobloom_v{VERSION}.pt' if portion is None else f'ckpt/{tree.dataset}_geobloom_v{VERSION}_{portion}.pt'
    transfer_path = f'model/tmp/{tree.dataset}_v{VERSION}/' if portion is None else f'model/tmp/{tree.dataset}_v{VERSION}_{portion}/'
    
    scaler = GradScaler()
    if not os.path.exists(transfer_path):
        os.makedirs(transfer_path)

    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    train_beam_width_str = '-'.join([str(x) for x in train_beam_width] if isinstance(train_beam_width, list) else [
        str(train_beam_width) for _ in range(tree.depth)])
    infer_beam_width_str = '-'.join([str(x) for x in infer_beam_width] if isinstance(infer_beam_width, list) else [
        str(infer_beam_width) for _ in range(tree.depth)])
    
    # Prepare the truth nodes for evaluation
    train_truth_nodes = [[set() for i in range(tree.depth)] for j in range(len(train_dataloader.dataset))]
    dev_truth_nodes = [[set() for i in range(tree.depth)] for j in range(len(dev_dataloader.dataset))]

    for i in range(len(train_dataloader.dataset)):
        for poi_idx in train_dataloader.dataset.truths[i]:
            path = tree.get_truth_path(poi_idx)
            for depth in range(tree.depth):
                train_truth_nodes[i][depth].add(path[depth])
    for i in range(len(dev_dataloader.dataset)):
        for poi_idx in dev_dataloader.dataset.truths[i]:
            path = tree.get_truth_path(poi_idx)
            for depth in range(tree.depth):
                dev_truth_nodes[i][depth].add(path[depth])
    

    def encode_node(node_path):
        node_representations = []
        for depth in range(tree.depth):
            node_dataset = NodeEncodeDataset([tree.levels[depth]])
            node_dataloader = DataLoader(node_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=node_dataset.collate_fn)
            for batch in tqdm(node_dataloader, desc=f'Encoding node representations at depth {depth}'):
                node_representations.append(model.encode_node(batch, depth).mul(127 * 64).round().to(dtype=torch.int32).cpu().numpy())
        node_representations = np.vstack(node_representations).astype(np.int32)
        with open(node_path, 'wb') as f:
            f.write(node_representations.tobytes())

    for epoch in range(max_epochs):
        # We now use C++ inference engine to get the candidates.
        # First, we encode all the nodes into node.bin
        encode_node(transfer_path + f'node_v{VERSION}.bin')
        # Second, save the quantized model parameters
        model.serialize(transfer_path + f'nnue_v{VERSION}.bin')
        # Third, run the C++ inference engine to get the candidates
        # For faster inference we use multiple threads on the C++ side.
        threads = str(8)
        train_params = [f'nnue/v{VERSION}/nnue', tree.dataset, 'pytrain', threads, train_beam_width_str, transfer_path]
        if portion is not None:
            train_params.append(portion)
        subprocess.run(train_params)
        subprocess.run([f'nnue/v{VERSION}/nnue', tree.dataset, 'pydev', threads, infer_beam_width_str, transfer_path])

        # Fourth, deserialize the candidates
        next_train_candidates, train_topk = tree.load_candidates(transfer_path + f'train_nodes.bin', len(train_dataloader.dataset), train_beam_width)
        next_dev_candidates, dev_topk = tree.load_candidates(transfer_path + f'dev_nodes.bin', len(dev_dataloader.dataset), infer_beam_width)

        # Fifth, calculate train & dev recall and ndcg
        train_recalls, train_ndcgs = [], []
        dev_recalls, dev_ndcgs = [], []
        for depth in range(1, tree.depth):
            train_recalls.append([])
            for i in range(len(train_dataloader.dataset)):
                hit = 0
                for node in next_train_candidates[depth-1][i]:
                    if node in train_truth_nodes[i][depth]:
                        hit += 1
                    if hit >= len(train_truth_nodes[i][depth]):
                        break
                train_recalls[-1].append(hit / len(train_truth_nodes[i][depth]))
            train_recalls[-1] = np.mean(train_recalls[-1])

        for i in range(len(train_dataloader.dataset)):
            pred = train_topk[i]
            truth = train_dataloader.dataset.truths[i]
            train_ndcgs.append(fast_ndcg(pred, truth, k=5))

        train_ndcgs = np.mean(train_ndcgs)

        for depth in range(1, tree.depth):
            dev_recalls.append([])
            for i in range(len(dev_dataloader.dataset)):
                hit = 0
                for node in next_dev_candidates[depth-1][i]:
                    if node in dev_truth_nodes[i][depth]:
                        hit += 1
                    if hit >= len(dev_truth_nodes[i][depth]):
                        break
                dev_recalls[-1].append(hit / len(dev_truth_nodes[i][depth]))
            dev_recalls[-1] = np.mean(dev_recalls[-1])

        for i in range(len(dev_dataloader.dataset)):
            pred = dev_topk[i]
            truth = dev_dataloader.dataset.truths[i]
            dev_ndcgs.append(fast_ndcg(pred, truth, k=5))

        dev_ndcgs = np.mean(dev_ndcgs)

        print(f'==================== Epoch {epoch} ====================')
        print(f'Train recall: {train_recalls}, Train NDCG @ 5: {train_ndcgs}')
        print(f'Dev recall: {dev_recalls}, Dev NDCG @ 5: {dev_ndcgs}')
        print(f'Previous max metrics: {max_metrics}')
        # update the max metrics
        for depth in range(tree.depth - 1):
            if dev_recalls[depth] > max_metrics[depth]:
                max_metrics[depth] = dev_recalls[depth]

        save = False
        if max_metrics[-1] < dev_ndcgs:
            max_metrics[-1] = dev_ndcgs
            save = True
            # If trained, save the model
        
        print(f'Current max metrics: {max_metrics}')

        if save:
            torch.save(model.state_dict(), ckpt_path)
            print(f'Model saved to {ckpt_path}')

        train_candidates = train_candidates[:1] + next_train_candidates

        model.train()
        # When training, we mix all the depth together to prevent catastrophic forgetting
        train_sample_by_depth = []
        train_target_by_depth = []
        print('Preparing training samples...')
        for depth in range(tree.depth):
            # construct training samples for all depths
            if depth > 0 or not init_train_resample:
                train_target = prepare_target(tree, depth, infer_train_dataloader, train_candidates[depth], ensure_truth=True)
            if depth == 0 and init_train_resample:
                train_sample = []
                for train_candidate_list in train_candidates[depth]:
                    sampled_candidates = random.sample(train_candidate_list, current_train_beam_width)
                    train_sample.append(sampled_candidates)
                train_target = prepare_target(tree, depth, infer_train_dataloader, train_sample, ensure_truth=True)
            else:
                train_sample = train_candidates[depth]
            train_sample_by_depth.append(train_sample)
            train_target_by_depth.append(train_target)

        train_bar = tqdm(train_dataloader, desc=f'Mixed training on {tree.depth} depths')
        retrieve_losses = []
        rank_losses = []
        last_retrieve_loss = 0
        last_rank_loss = 0
        for batch in train_bar:
            query_idxs, query_bloom_filter, query_loc, _ = batch
            query_bloom_filter = query_bloom_filter.to_dense()
            for depth in range(tree.depth):
                train_target = train_target_by_depth[depth]
                # If it is the first depth, sampling not required, and the batch size is equal to the train batch size,
                # We can use the pre-computed bloom filters to save time.
                if depth == 0 and not init_train_resample and len(query_idxs) == train_batch_size:
                    node_bfs, node_locs, node_radius = init_train_bloom_filters, init_train_node_locs, init_train_node_radius
                else:
                    train_sample = train_sample_by_depth[depth]
                    node_bfs, node_locs, node_radius = collate_query_candidates(tree, query_idxs, train_sample)
                node_bfs = node_bfs.to_dense()
                target = train_target.index_select(dim=0, index=torch.tensor(query_idxs, dtype=torch.int64, device='cuda'))
                target = target.to_dense()
                # v18: optimize the context score via lambdaRank
                with torch.inference_mode():
                    select_mask = target == 1
                    truth_bloom_filter = []
                    for i in range(target.shape[0]):
                        selected_nodes = node_bfs[i][select_mask[i]]
                        truth_bloom_filter.append(torch.sum(selected_nodes, dim=0))
                    truth_bloom_filter = torch.stack(truth_bloom_filter)
                    context_truth = (query_bloom_filter - truth_bloom_filter).relu()
                    # The context truth only contains those words that are not in the ground truth but in the query.
                    # We need to find the most similar node bloom filter in the slate, and select it as the context.
                    query_bits = query_bloom_filter.sum(dim=-1)
                    easy_mask = (context_truth.sum(dim=-1) < query_bits * 0.5)
                    context_rank_truth = torch.sum(context_truth.unsqueeze(1).expand_as(node_bfs) * node_bfs, dim=-1) # (batch_size, slate_length)
                    easy_rank_truth = torch.sum(query_bloom_filter.unsqueeze(1).expand_as(node_bfs) * node_bfs, dim=-1)
                    context_rank_truth[easy_mask] = easy_rank_truth[easy_mask]  
                    node_dist = GeoBloom.node_pairwise_distances(node_locs)
                    node_dist[target.unsqueeze(-2).expand_as(node_dist) == 0] = float('inf')
                    min_dist, _ = torch.min(node_dist, dim=-1)
                    dist_mask = min_dist > 1000
                    context_rank_truth[dist_mask] = 0
                    if ensure_context_in_beam:
                        # Add 0.5 to the max context truth to ensure it is in the beam
                        # On GeoGLUE this is quite effective, but on Meituan this makes no difference. 
                        col_indices = torch.argmax(context_rank_truth, dim=-1)
                        row_indices = torch.arange(context_rank_truth.shape[0], dtype=torch.int64, device='cuda')
                        target[row_indices, col_indices] += 0.5
                    context_rank_truth[easy_mask] = -1    
                    del node_dist, min_dist, dist_mask, easy_mask, easy_rank_truth, truth_bloom_filter, context_truth, select_mask
                    
                score, context_score = model(query_bloom_filter, node_bfs, query_loc, node_locs, node_radius, depth)
                node_bfs, node_locs, node_radius = None, None, None
                loss_fn = rank_loss if depth == tree.depth - 1 else retrieve_loss
                loss = loss_fn(score, target) + 0.1 * context_loss(context_score, context_rank_truth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.clip_weights()
                loss_value = loss.item()
                if depth == tree.depth - 1:
                    last_rank_loss = loss_value
                    rank_losses.append(loss_value)
                else:
                    last_retrieve_loss = loss_value
                    retrieve_losses.append(loss_value)
                train_bar.set_postfix({'retrieve_loss': last_retrieve_loss, 'rank_loss': last_rank_loss})

        print(f'Epoch={epoch}, retrieve loss={np.mean(retrieve_losses)}, rank loss={np.mean(rank_losses)}')

    print(f'Max metrics: {max_metrics}')

    # We now quantize the best model and encode the node for further testing.
    model.load_state_dict(torch.load(ckpt_path))
    encode_node(f'data_bin/{tree.dataset}/node_v{VERSION}.bin')
    model.serialize(f'data_bin/{tree.dataset}/nnue_v{VERSION}.bin')
    print(f'Model serialized to data_bin/{tree.dataset}/nnue_v{VERSION}.bin')


@torch.no_grad()
def infer(model: GeoBloom, tree: BloomFilterTree, infer_beam_width, dataloader):
    # NOTE: This function is only for debugging. Use the C++ inference engine instead.
    model.eval()
    model.quantize_test()
    candidates = [[] for _ in range(tree.depth)]
    candidates[0] = [tree.init_candidates] * len(dataloader.dataset)
    batch_size = dataloader.batch_size
    init_bloom_filters_sparse, init_node_locs, init_node_radius = torch.stack([tree.init_bloom_filter] * batch_size), tree.init_loc.unsqueeze(0).repeat(batch_size, 1, 1), tree.init_radius.unsqueeze(0).repeat(batch_size, 1, 1)

    final_metrics = [0] * tree.depth
    model.eval()
    results = []
    for depth in range(tree.depth):
        next_depth = depth + 1 if depth < tree.depth - 1 else -1
        current_infer_beam_width = infer_beam_width[next_depth] if isinstance(infer_beam_width, list) else infer_beam_width
        if depth == 0:
            init_bloom_filters = init_bloom_filters_sparse.to_dense()
        next_candidates = [[] for _ in range(len(dataloader.dataset))]
        metrics = []
        for batch in tqdm(dataloader, desc=f'Searching at depth = {depth}'):
            query_idxs, query_bloom_filter, query_loc, truths = batch
            if depth == 0:
                if len(query_idxs) == batch_size:
                    node_bfs, node_locs, node_radius = init_bloom_filters, init_node_locs, init_node_radius
                else:
                    node_bfs, node_locs, node_radius = torch.stack([tree.init_bloom_filter] * len(query_idxs)).to_dense(), tree.init_loc.unsqueeze(0).repeat(len(query_idxs), 1, 1), tree.init_radius.unsqueeze(0).repeat(len(query_idxs), 1, 1)
            else:
                node_bfs, node_locs, node_radius = collate_query_candidates(tree, query_idxs, candidates[depth])
            query_bloom_filter = query_bloom_filter.to_dense()
            node_bfs = node_bfs.to_dense()
            score, context_score = model(query_bloom_filter, node_bfs, query_loc, node_locs, node_radius, depth)
            node_bfs, node_locs, node_radius = None, None, None
            sorted_indices = torch.argsort(score, dim=-1, descending=True).cpu().numpy()
            for i, query_id in enumerate(query_idxs):
                metric_depth = depth + 1 if depth < tree.depth - 1 else -1
                if depth == tree.depth - 1:
                    predict = [candidates[depth][query_id][idx].poi_idx for idx in sorted_indices[i]]
                    metrics.append(fast_ndcg(predict, truths[i], k=5))
                    results.append(predict)
                else:
                    truth_nodes = set()
                    for poi_idx in truths[i]:
                        truth_nodes.add(tree.get_truth_path(poi_idx)[metric_depth])
                    hit = 0
                    for idx in sorted_indices[i]:
                        for child in candidates[depth][query_id][idx].child:
                            next_candidates[query_id].append(child)
                            if child in truth_nodes:
                                hit += 1
                            if len(next_candidates[query_id]) >= current_infer_beam_width:
                                break
                        if len(next_candidates[query_id]) >= current_infer_beam_width:
                            break
                    metrics.append(hit / len(truth_nodes))

        final_metrics[depth] = np.mean(metrics)
        print('Current metrics: ', final_metrics)
        
        init_bloom_filters = None
        if depth == tree.depth - 1:
            break
        candidates[depth + 1] = next_candidates

    print(f'Metrics: {final_metrics}')

    results = np.array(results).astype(np.uint32)
    return results

                
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GeoGLUE_clean')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--portion', type=str, default='1')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    dataset = args.dataset
    task = args.task
    portion = args.portion
    portion = None if portion == '1' else portion
    max_epochs = args.epochs

    ckpt_path = f'ckpt/{dataset}_geobloom_v{VERSION}.pt' if portion is None else f'ckpt/{dataset}_geobloom_v{VERSION}_{portion}.pt'
    nnue_path = f'data_bin/{dataset}/nnue_v{VERSION}.bin'

    task_list = ['train', 'pytest', 'quantize', 'node', 'continue', 'size_test']

    assert task in task_list, f'The task should be in {task_list}'

    if task == 'quantize' or task == 'size_test':
        # load the model ready for quantization
        ckpt = torch.load(ckpt_path)
        # Identify the tensor - replace 'layer_name.weight' with the actual key
        if '_orig_mod.a.weight' in ckpt:
            depth = ckpt['_orig_mod.a.weight'].shape[1]
            model = GeoBloom(depth=depth).cuda()
            model = torch.compile(model)
        elif 'a.weight' in ckpt:
            depth = ckpt['a.weight'].shape[1]
            model = GeoBloom(depth=depth).cuda()
        else:
            print('Cannot find the depth of the model in the checkpoint.')
            exit()
        model.load_state_dict(ckpt)
        model.serialize(nnue_path if task != 'size_test' else f'data_bin/{dataset}/nnue_v{VERSION}_size_test.bin')
        print(f'Quantized model saved to {nnue_path}')
        if task == 'quantize':
            exit()

    # Create the data module and prepare the datasets
    batch_size = 4 if 'GeoGLUE' in dataset else 32
    poi_dataset = POIDataset(dataset, batch_size=batch_size, num_workers=0, load_query=task!='node', portion=portion)

    train_beam_width = {
        'MeituanBeijing': 200,
        'MeituanShanghai': 200,
        'GeoGLUE': [1000, 1000, 1000, 1000],
        'GeoGLUE_clean': [1000, 500, 500, 500],
    }

    infer_beam_width = {
        'MeituanBeijing': [400, 400, 400, 400],
        'MeituanShanghai': [400, 400, 400, 400],
        'GeoGLUE': [6000, 4000, 4000, 1000],
        'GeoGLUE_clean': [2000, 1000, 1000, 1000],
    }

    # Initialize the model
    tree = BloomFilterTree(dataset, poi_dataset.poi_bloom_filters, poi_dataset.poi_locs)
    model = GeoBloom(depth=tree.depth).cuda()
    model.set_idf_vec(tree.compute_idf_vec())
    model = torch.compile(model)
    if 'GeoGLUE' in dataset:
        learning_rate = 5e-4
    else:
        learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if task == 'train' or task == 'continue':
        if task == 'continue':
            model.load_state_dict(torch.load(ckpt_path))
        # Training loop
        # If your device doesn't support torch.compile, please comment the following line.
        train(model, 
            optimizer, 
            tree,
            train_beam_width[dataset], 
            infer_beam_width[dataset],
            max_epochs = max_epochs,
            train_dataloader=poi_dataset.train_dataloader, 
            infer_train_dataloader=poi_dataset.infer_train_dataloader, 
            dev_dataloader=poi_dataset.dev_dataloader,
            ensure_context_in_beam=dataset == 'GeoGLUE',
            portion=portion)

    elif task == 'pytest':
        # load the current best model
        state_dict = torch.load(ckpt_path)
        uncompiled_model = GeoBloom(depth=tree.depth).cuda()
        # clone the weights
        new_state_dict = {}
        for key in state_dict:
            # if the key starts with _orig_mod, it is the quantized model
            if key.startswith('_orig_mod.'):
                new_state_dict[key[10:]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        uncompiled_model.load_state_dict(new_state_dict)
        # Testing loop
        # load the best model
        top_indices = infer(uncompiled_model, tree, infer_beam_width[dataset], poi_dataset.test_dataloader)
        # np.save(f'result/{dataset}_geobloom_v{VERSION}_top100.npy', top_indices)

    elif task == 'node' or task == 'size_test':
        # load the current best model
        model.load_state_dict(torch.load(ckpt_path))
        node_representations = []
        for depth in range(tree.depth):
            node_dataset = NodeEncodeDataset([tree.levels[depth]])
            node_dataloader = DataLoader(node_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=node_dataset.collate_fn)
            for batch in tqdm(node_dataloader, desc=f'Encoding node representations at depth {depth}'):
                node_representations.append(model.encode_node(batch, depth).mul(127 * 64).round().to(dtype=torch.int32).cpu().numpy())

        if task == 'node':
            # In inference we simply use int32 precomputed embedding to following the NNUE scheme, reducing our workload.
            node_representations = np.vstack(node_representations).astype(np.int32)
            node_path = f'data_bin/{dataset}/node_v{VERSION}.bin' 
        else:
            # In the paper we compare with int16 embeddings. As the embedding dim=32, the size differences are small.
            # We will further adjust the C++ engine to support int16 embeddings.
            node_representations = np.vstack(node_representations).astype(np.int16)
            node_path = f'data_bin/{dataset}/node_v{VERSION}.bin'

        with open(node_path, 'wb') as f:
            f.write(node_representations.tobytes())
        print(f'Node representations saved to {node_path}')


