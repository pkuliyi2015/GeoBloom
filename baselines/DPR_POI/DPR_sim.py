import argparse
import numpy as np
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MeituanShanghai')
parser.add_argument('--portion', type=str, default='0.1')

dataset = parser.parse_args().dataset
processed_path = f'data/{dataset}'

portion = parser.parse_args().portion
if portion == '1':
    portion_path = 'full'
else:
    portion_path = f'portion_{portion}'

poi_embeddings = np.load(f'baselines/DPR_POI/embeddings/{dataset}/{portion_path}/poi_embeddings.npy')
test_embeddings = np.load(f'baselines/DPR_POI/embeddings/{dataset}/{portion_path}/test_embeddings.npy')

# for each test embedding, find the top k poi embeddings with the highest cosine similarity

import torch

poi_embeddings = torch.tensor(poi_embeddings, dtype=torch.float16).cuda()
test_embeddings = torch.tensor(test_embeddings, dtype=torch.float16).cuda()

top_k_indices = []
batch_size = 16384
for i in trange(test_embeddings.size(0)):
    test_embedding = test_embeddings[i].unsqueeze(0)
    all_cosine_sims = []

    # Process poi_embeddings in batches
    for batch_start in range(0, poi_embeddings.size(0), batch_size):
        batch_end = min(batch_start + batch_size, poi_embeddings.size(0))
        poi_batch = poi_embeddings[batch_start:batch_end]
        
        # Compute cosine similarity for the batch
        cosine_sim_batch = torch.cosine_similarity(poi_batch, test_embedding, dim=1)
        all_cosine_sims.append(cosine_sim_batch)
    
    # Concatenate the results from all batches
    all_cosine_sims = torch.cat(all_cosine_sims, dim=0)
    
    # Find top k indices for the concatenated results
    top_k = torch.topk(all_cosine_sims, 100, largest=True, sorted=True)
    top_k_indices.append(top_k.indices.cpu().numpy())

# save it to the result file
    
top_k_indices = np.array(top_k_indices)

if portion == '1':
    np.save(f'result/{dataset}_DPR_top100.npy', top_k_indices)
else:
    np.save(f'result/{dataset}_DPR_{portion}_top100.npy', top_k_indices)