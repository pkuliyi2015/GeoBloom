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
        3. BloomNNUE writer functions that quantize the model parameters and write them into nnue/{dataset}_v{VERSION}.nnue file.
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
        

        Experimental Results (* means the best result):

            MeituanBeijing: beam 200-300-300-300
            Total search time of all threads: 5.88955s, Query Per Second: 1606.41 (v17), 1247.86 (v18)
            =============== Intermediate Recall Scores ==============
            0.984915        0.939978        0.874999                    (BloomNNUE v19 no context)*
            0.986297        0.935612        0.872766                    (BloomNNUE v19)
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.825786        0.788883        0.643527        0.547722    (BloomNNUE v19 no context)
            0.824922        0.786649        0.644604        0.549730    (BloomNNUE v19)
            0.726300        0.663300        0.505600        0.414500    (Best Baseline TkQ-DPR_D)
            =========================================================

            MeituanShanghai: beam 200-400-300-300
            Total search time of all threads: 12.1486s, Query Per Second: 1046.87(v17), 781.081(v18)
            =============== Intermediate Recall Scores ==============
            0.965070        0.932397        0.884622                    (BloomNNUE v19 no context)*
            0.962212        0.928148        0.882186                    (BloomNNUE v19)
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.826102        0.790903        0.658958        0.567857    (BloomNNUE v19 no context)
            0.825824        0.794026        0.663791        0.572260    (BloomNNUE v19)*
            0.771700        0.734800        0.574200        0.455600    (Best Baseline TkQ-DPR_D)
            =========================================================

            GeoGLUE: beam 6000-4000-4000-1000
            Total search time of all threads: 683.43s, Query Per Second: 29.2641(v17), 24.2718(v18)
            =============== Intermediate Recall Scores ==============
            0.908900        0.839600        0.786750                    (BloomNNUE v19 no context)
            0.919650        0.861850        0.803650                    (BloomNNUE v19)*
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.764550        0.736650        0.610570        0.513950    (BloomNNUE v19 no context)
            0.792250        0.762950        0.634941        0.534300    (BloomNNUE v19)*
            0.735700        0.701200        0.579700        0.484800    (Best Baseline TkQ-DPR_D)
            =========================================================

        The model structure and the performance is good enough for a new paper.
        We will now start the paper writing process.

        DATE: 2024-01-09

    v19_portion: Experiments to test for varying training data portions.

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

from typing import List
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from bloom_filter import make_hashfuncs
from lambdarank import lambdaLoss
from kmeans_tree import build_kmeans_tree

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
NUM_SLICES = 4
NUM_BITS = 8192 
# 4*8192 = -32767 ~ 32768 can be exactly stored with torch.int16, so it saves VRAM
# And we can compare with other methods conveniently as they all uses fp16

# Model definition
class BloomNNUE(nn.Module):
    def __init__(self, depth=4, quantized_one=127.0, weight_scale=64.0, division_factor=16.0, d_threshold=1000.0):
        super(BloomNNUE, self).__init__()
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

        # Initialize all decoder weights to 0.0 at the last layer, i.e., Zero-Projection
        for i in range(depth):
            nn.init.zeros_(self.rank[2 * i + 1].weight)
            nn.init.zeros_(self.context_select[2 * i + 1].weight)
            nn.init.zeros_(self.context_rank[2 * i + 1].weight)
            nn.init.zeros_(self.residual[2 * i + 1].weight)

        nn.init.ones_(self.a.weight)
        nn.init.zeros_(self.b.weight)
        nn.init.ones_(self.c.weight)
        nn.init.zeros_(self.d.weight)

        weight_bound = quantized_one / weight_scale
        self.weight_bound = weight_bound
        self.weight_clipping = [
            {'params' : [self.encoder.weight], 'min_weight' : - (2**15 - 1) // 127, 'max_weight' : (2**15 - 1) // 127 }, # For 16-bit quantization
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
            # In real inference, we follows the NNUE quantization scheme, which yields much smaller model file (~28 MB)
            serialzer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int16).flatten().cpu().numpy()
            serialzer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int8).flatten().cpu().numpy()
            serialzer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int32).flatten().cpu().numpy()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float32)
        else:
            # In size test, we use all 16-bit format to compare with other baselines in fp16 (~40MB)
            # NOTE: This is only for academic comparison. the resulting model is not only larger but also can't correctly.
            serialzer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int16).flatten().cpu().numpy()
            serialzer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int16).flatten().cpu().numpy()
            serialzer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int16).flatten().cpu().numpy()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float16)
            
        buf = bytearray()
        # header: tree depth
        buf.extend(struct.pack('H', self.depth))
        # encoder layer
        buf.extend(serialzer_encoder(self.encoder.weight.T).tobytes())
        buf.extend(serialzer_encoder(self.encoder.bias).tobytes())

        # bottleneck
        for i in range(self.depth):
            buf.extend(serialzer8(self.bottleneck[i].weight[:,:256]).tobytes())

        # all heads
        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(self.depth):
                buf.extend(serialzer8(layer[2 * i].weight).tobytes())
                buf.extend(serialzer32(layer[2 * i].bias).tobytes())
                buf.extend(serialzer8(layer[2 * i + 1].weight).tobytes())

        # the parameter a, b, c, d
        buf.extend(serializer_float(self.a.weight).tobytes())
        buf.extend(serializer_float(self.b.weight).tobytes())
        buf.extend(serializer_float(self.c.weight).tobytes())
        buf.extend(serializer_float(self.d.weight).tobytes())

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
            {'params': self.bottleneck.parameters(), 'lr' : 1e-3},
            {'params': self.a.parameters(), 'lr': 1e-3},
            {'params': self.b.parameters(), 'lr': 1e-3},
            {'params': self.c.parameters(), 'lr': 1e-3},
            {'params': self.d.parameters(), 'lr': 1e-3},
        ]
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
        hidden_rank = sum(hidden_rank) + 1
        del hidden_rank_splits, weight_rank_splits
        intersection = torch.mul(query_bloom_filter.unsqueeze(1), node_bloom_filter)
        text_score = torch.sum(hidden_rank * intersection, dim=-1)
        del hidden_rank

        hidden_context_select = self.context_select_list[2 * depth](hidden)
        hidden_context_select = torch.clamp(hidden_context_select, min=0, max=1)
        hidden_context_select_splits = torch.chunk(hidden_context_select, num_chunks, dim=-1)
        weight_context_select_splits = torch.chunk(self.context_select_list[2 * depth + 1].weight.T, num_chunks, dim=0)
        hidden_context_select = [F.relu(h_split.matmul(w_split) + 1e-6) for h_split, w_split in zip(hidden_context_select_splits, weight_context_select_splits)]
        hidden_context_select = sum(hidden_context_select) + 1
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
    
class POIDataset:
    def __init__(self, dataset, batch_size, num_workers=0, load_query=True, portion=None):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = os.path.join('data', dataset)
        self.data_bin_dir = os.path.join('data_bin', dataset)
        if not os.path.exists(self.data_bin_dir):
            os.makedirs(self.data_bin_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.poi_bloom_filters, self.poi_locs, _ = self.build_or_load('poi')

        if load_query:
            train_bloom_filters, train_locs, train_truths = self.build_or_load('train', portion)
            dev_bloom_filters, dev_locs, dev_truths = self.build_or_load('dev')
            test_bloom_filters, test_locs, test_truths = self.build_or_load('test')

            self.train_dataset = QueryDataset(train_bloom_filters, train_locs, train_truths)
            self.dev_dataset = QueryDataset(dev_bloom_filters, dev_locs, dev_truths)
            self.test_dataset = QueryDataset(test_bloom_filters, test_locs, test_truths)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_query_fn)
            self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_query_fn)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_query_fn)
            self.infer_train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_query_fn)

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
            bloom_filters, locs, truths = self.load_data(raw_file, is_query=split != 'poi')
            self.serialize(bloom_filters, locs, bin_file, truths)
        return bloom_filters, locs, truths

    @staticmethod
    def load_data(file_dir, is_query=True):
        bloom_filters = []
        locs = []
        truths = []
        hash_func_inner, _ = make_hashfuncs(NUM_SLICES, NUM_BITS)

        def hash_func(t):
            hash_list = list(hash_func_inner(t))
            for i in range(1, NUM_SLICES):
                hash_list[i] += i * NUM_BITS
            return set(hash_list)
        
        def ngram_split(text, n=3):
            ngrams = set()
            for k in range(1, n + 1):
                for i in range(len(text) - k + 1):
                    ngrams.add(text[i:i + k])
            return ngrams
        
        with open(file_dir, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading '+ ('query' if is_query else 'POI') + ' data'):
                line = line.strip().split('\t')
                # Avoid mixing the fields in POIs. This helps to reduce the size of POI bloom filters.
                if not is_query:
                    fields = set(line[0].split(','))
                    text = set()
                    for field in fields:
                        text.update(ngram_split(field))
                else:
                    text = ngram_split(line[0])
                bloom_filter = set()
                for t in text:
                    bloom_filter.update(hash_func(t))
                bloom_filters.append(bloom_filter)
                x, y = float(line[1]), float(line[2])
                locs.append([x, y])
                if is_query:
                    truths.append([int(x) for x in line[3].split(',')])
        return bloom_filters, locs, truths
    
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


class GeoKMeansTree:
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
        if dataset == 'MeituanBeijing' or dataset == 'MeituanBeijingZero':
            first_level_width = 50
        elif dataset == 'MeituanShanghai' or dataset == 'MeituanShanghaiZero' or dataset == 'Meituan':
            first_level_width = 200
        elif dataset == 'GeoGLUE' or dataset == 'GeoGLUEZero':
            first_level_width = 4000
        elif dataset == 'GeoGLUE_clean' or dataset == 'GeoGLUE_cleanZero':
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
        self.query_bloom_filters = []
        self.query_locs = []
        # we transfer all the bloom filters into cuda long tensors to save time
        for query_bloom_filter in tqdm(query_bloom_filters, desc='Transfering query bloom filters'):
            self.query_bloom_filters.append(torch.tensor(list(query_bloom_filter), dtype=torch.int16, device='cuda'))
        for loc in tqdm(query_locs, desc='Transfering query locations'):
            self.query_locs.append(torch.tensor(loc, dtype=torch.float32, device='cuda'))
        self.truths = truths
    
    def __len__(self):
        return len(self.query_bloom_filters)
    
    def __getitem__(self, query_idx):
        return query_idx, self.query_bloom_filters[query_idx], self.query_locs[query_idx], self.truths[query_idx]

    
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

def train(model: BloomNNUE, optimizer, tree: GeoKMeansTree, train_beam_width, infer_beam_width, max_epochs, train_dataloader, infer_train_dataloader, dev_dataloader, ensure_context_in_beam=False, portion=None):
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

    scaler = torch.cuda.amp.GradScaler()

    transfer_path = f'model/tmp/{tree.dataset}_v{VERSION}/' if portion is None else f'model/tmp/{tree.dataset}_v{VERSION}_{portion}/'
    if not os.path.exists(transfer_path):
        os.makedirs(transfer_path)

    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
        ckpt_path = f'ckpt/{tree.dataset}_geobloom_v{VERSION}.pt' if portion is None else f'ckpt/{tree.dataset}_geobloom_v{VERSION}_{portion}.pt'

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
        train_params = [f'nnue/nnue_v{VERSION}', tree.dataset, 'pytrain', threads, train_beam_width_str, transfer_path]
        if portion is not None:
            train_params.append(portion)
        subprocess.run(train_params)
        subprocess.run([f'nnue/nnue_v{VERSION}', tree.dataset, 'pydev', threads, infer_beam_width_str, transfer_path])

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
                    node_dist = BloomNNUE.node_pairwise_distances(node_locs)
                    node_dist[target.unsqueeze(-2).expand_as(node_dist) == 0] = float('inf')
                    min_dist, _ = torch.min(node_dist, dim=-1)
                    dist_mask = min_dist > 1000
                    context_rank_truth[dist_mask] = 0
                    del node_dist, min_dist, dist_mask, easy_mask, easy_rank_truth, truth_bloom_filter, context_truth, select_mask

                score, context_score = model(query_bloom_filter, node_bfs, query_loc, node_locs, node_radius, depth)
                node_bfs, node_locs, node_radius = None, None, None
                if ensure_context_in_beam:
                    # Add 0.5 to the max context truth to ensure it is in the beam
                    # On GeoGLUE this is quite effective, but on Meituan this makes no difference. 
                    col_indices = torch.argmax(context_rank_truth, dim=-1)
                    row_indices = torch.arange(context_rank_truth.shape[0], dtype=torch.int64, device='cuda')
                    target[row_indices, col_indices] += 0.5
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
    encode_node(transfer_path + f'node_v{VERSION}.bin')
    model.serialize(transfer_path + f'nnue_v{VERSION}.bin')


@torch.no_grad()
def infer(model: BloomNNUE, tree: GeoKMeansTree, infer_beam_width, dataloader):
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

    dataset = parser.parse_args().dataset
    task = parser.parse_args().task
    portion = parser.parse_args().portion
    portion = None if portion == '1' else portion
    max_epochs = parser.parse_args().epochs

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
            model = BloomNNUE(depth=depth).cuda()
            model = torch.compile(model)
        elif 'a.weight' in ckpt:
            depth = ckpt['a.weight'].shape[1]
            model = BloomNNUE(depth=depth).cuda()
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
        'GeoGLUE_clean': [1000, 1000, 1000, 1000],
    }

    # Initialize the model
    tree = GeoKMeansTree(dataset, poi_dataset.poi_bloom_filters, poi_dataset.poi_locs)
    model = BloomNNUE(depth=tree.depth).cuda()
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
            ensure_context_in_beam='GeoGLUE' in dataset,
            portion=portion)

    elif task == 'pytest':
        # load the current best model
        state_dict = torch.load(ckpt_path)
        uncompiled_model = BloomNNUE(depth=tree.depth).cuda()
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


