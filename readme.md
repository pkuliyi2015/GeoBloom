# GeoBloom

- This is the official repository of the paper:  *GeoBloom: Revitalizing Lightweight Models for Geographic Information Retrieval*

## Update
- We found that Anonymous GitHub has a serious bug, which doesn't sync nested folder (nnue/v19/) in our repo. We now manually fixed the bug by adding a placeholder file.

## Quick Start

- With this repository, you can
  - Download the GeoGLUE and GeoGLUE-clean datasets in the paper.
  - Quickly reproduce the unsupervised results of GeoBloom.
  - Train and reproduce the supervised effectiveness of GeoBloom.
  - Compare with baselines BM25-D, BERT, OpenAI-D and DPR-D.
  - Measure the inference time, memory, and disk space usage. 

### Environment Preparation

While our framework can run on various devices, to ensure it works smoothly, we recommend:

- A computer **Linux** system (we use Ubuntu 20.04) with CPUs supporting **AVX-2** instruction set. 
  - Currently, this code repo doesn't support other systems due to our heavy workload.
  - It should be easy to transfer our code to these systems as well.
- Please use **Miniconda or Anaconda**

- We use Python 3.11.5 Lower versions are not tested.
  - conda create -n GeoBloom python==3.11.5


Our methods required:

- PyTorch>=2.1.1 
- tqdm
- scikit-learn==1.4.2
- xxhash==3.4.1

### Dataset Preparation

- Clone this repository.
- Unzip the GeoGLUE_clean.zip in the data folder. 
  - GeoGLUE is too large to be included in this anonymized repo directly.
  - You can either process it on your own or wait for our release of a Google Drive link.


#### Notes

- We will redistribute the **GeoGLUE** dataset following their [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?spm=5176.12282016.0.0.63b47586Niz8D0) license. 
  - The original dataset is from [modelscope.cn](https://www.modelscope.cn/datasets/iic/GeoGLUE/summary)
  - We extract the text, latitude, longitude, and ground truth ids, project them into Euclidean space with pyproj library:
    - proj = pyproj.Transformer.from_crs( 'epsg:4326', 'epsg:2385')
- As GeoGLUE contains over 50% of fake POIs, we align their queries to real POIs, which forms a new dataset GeoGLUE-clean.
  - The text, latitude, and longitude of each query remain unchanged.
  - The ground truth id now point to our real POIs.
  - Those queries that can't find a real POI in our dataset are excluded. 
- Meituan-Beijing and Meituan-Shanghai can't be released due to license constraints.

## **Experiments**

### Unsupervised Effectiveness

We provide an unsupervised version of the GeoBloom that doesn't require training.

#### Steps:

- Preprocess the dataset into Bloom filters and construct the tree:
  - **python model/dataset.py --dataset GeoGLUE_clean**
- Compile the nnue engine:
  - **g++ nnue/v19/nnue_unsupervised.cpp -o nnue/v19/nnue_unsupervised -pthread -mavx2 -O3 -fno-tree-vectorize**
- Retrieve the results:
  - **nnue/v19/nnue_unsupervised GeoGLUE_clean test 8 4000-4000-4000-4000**

Following the steps, you should see something like this:

- GeoGLUE_clean

```
Loading dataset from path: data_bin/GeoGLUE_clean/
Use unsupervised model.
Allocating 434.887 MB for 111331 bloom filters on the tree...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 168.095s, Query Per Second: 72.3222
=============== Intermediate Recall Scores ==============
0.925722        0.880234        0.818952
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.484495        0.427161        0.302846        0.224315
=========================================================
Predictions saved to data_bin/GeoGLUE_clean/test_nodes.bin
```

Note:

- The intermediate recall scores show the recall rates of beam search at each tree depth. It gives insights into the beam width selection.
- We support multi-threading (8 threads in the above example) for your convenience, but the QPS is always evaluated per thread.
- Here, we simply choose a unified beam width (e.g., 4000) and it can be quite "slow". You can set different beam widths at each tree level to obtain faster speed.


### Supervised Effectiveness

The training requires an NVIDIA GPU. We tested on A6000 24GB / NVIDIA V100 32GB, both can run the script without issues.

#### Steps:

- Compile the NNUE engine. You must do it first as the training script relies on this C++ engine.
  - **g++ nnue/v19/nnue.cpp -o nnue/v19/nnue -pthread -mavx2 -O3 -fno-tree-vectorize**
  
- Run the training script
  - **python model/geobloom_v19.py --dataset GeoGLUE_clean --epochs 5**

  It takes around 8 hours on GeoGLUE and 3 hours on GeoGLUE-clean to complete on an NVIDIA A6000 24GB. 
  
- Retrieve the results:

  - **nnue/v19/nnue GeoGLUE_clean test 8 1000-1000-1000-1000**

Following the steps, you should see something like this:

```
Beam width: 1000 1000 1000 1000 Loading dataset from path: data_bin/GeoGLUE_clean/
Allocating 434.887 MB for 111331 bloom filters on the tree...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 65.6473s, Query Per Second: 185.187
=============== Intermediate Recall Scores ==============
0.934935        0.904170        0.872419
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.749692        0.685531        0.515367        0.400921
=========================================================
Predictions saved to data_bin/GeoGLUE_clean/test_nodes.bin
```

### Varying Training Data Portions

Our model improves stably and consistently from 2% to 100% training data usage. 

#### Steps:

- Run the dataset split script.
  - **python model/train_portion.py --dataset GeoGLUE_clean**
- Training the model on that portion.
  - **python model/geobloom.py --dataset GeoGLUE_clean --portion 0.02 --epochs 10**
  - You can select from [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
  - It should be very fast when trained on 2% queries. 
- Retrieve the results:

  - **nnue/v19/nnue GeoGLUE_clean test 8 1000-1000-1000-1000**

Following these steps, you should see something like this:

```
Beam width: 1000 1000 1000 1000 Loading dataset from path: data_bin/GeoGLUE_clean/
Context select head0 untrained
Allocating 434.887 MB for 111331 bloom filters on the tree...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 65.3689s, Query Per Second: 185.975
=============== Intermediate Recall Scores ==============
0.910093        0.870939        0.814428
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.644731        0.576458        0.412220        0.302130
=========================================================
```

**Note: **The latest training will automatically overwrite the quantized model and pre-computed embeddings from previous training. 

To restore from a certain best training, run:

- python model/geobloom.py --dataset GeoGLUE_clean --portion 0.02 --task node

- python model/geobloom.py --dataset GeoGLUE_clean --portion 0.02 --task quantize
- This will output the model weight and node embeddings, by loading the best model previously trained and stored in the ckpt/ folder.

### Inference Time, Memory, and Disk Usage

- You can directly see the inference time from the model output.
- We recommend to use specialized tools (e.g., valgrind) for memory usage.
- The disk usage of our model ise the sum of poi.bin, node_v19.bin, nnue_v19.bin, and tree.bin in the data_bin folder.
  - In NNUE quantization scheme, not every parameter is stored in 16-bit numbers, preventing a fair comparison. 
  - We thereby provide an option, by running.
    - python model/geobloom.py --dataset GeoGLUE_clean --task size_test
  - It converts everything into 16-bit. Then, you can directly compare with PLMs (e.g., DPR_D), which need much larger sizes when stored in 16-bit.

You should see something like this:

```
======== File Sizes in MB ========
Components
Model           40.10
Embeddings      54.24
Bloom Filters   455.02
Tree            3.82
Total - 553.16
==================================
```

### Baselines

- For your convenience, we provide the baselines BM25-D, OpenAI-D, and DPR-D, which are relatively powerful in our tasks.
- You can quickly evaluate all baselines versus our method, but these methods require additional packages (jieba, acceleration, transformers,...)
- We strictly conduct experiments without any cherry-picking, so the result should be very close to that reported in our paper. 
  - You need OpenAI key for the OpenAI baseline, and it will cost around $2. We used the API on March 4th, 2024. Until that time, it was still not as competitive as BM25 on the provided datasets.
  - BM25_D's performance is slightly enhanced with a specialized tokenizer *jieba Chinese text segmentation library* and *jieba.lcut_for_search* function.
  - DPR_D requires negative sampling. We use the BM25_D to generate hard negative samples for it.

## Future Development

- Currently, the training is significantly slower than the inference, as we treat Bloom filters as dense vectors and only rely on PyTorch torch.compile for acceleration. It should be much faster if we implement custom CUDA kernels.

- Sophisticated Bloom filters and index design can be applied to this framework to achieve better space efficiency, making it possible to deploy as offline map apps.


## Contents

```
GeoBloom
├── baselines
| ├── BM25 # BM25, BM25-D with jieba and rank_bm25
| ├── BERT # Huggingface transformers
| ├── OpenAI # OpenAI API
| └── DPR	# Modified from NanoDPR https://github.com/Hannibal046/nanoDPR
├── ckpt # Store the checkpoints previously trained.
├── data  
| ├── GeoGLUE === Will release soon.
| └── GeoGLUE_Cleaned # Please decompress the file from this repo.
├── data_bin === The model will generate binary files here for C++ inference.
├── model
|	├── bloom_filter.py  === The vanilla Bloom filter generator.
|	├── bloom_filter_tree.py  === Organize the Bloom filter into a tree-based index.
|	├── dataset.py  ===  Pytorch dataset. It can also run independently to preprocess the dataset into Bloom filters.
|	├── geobloom_v19.py === Main script, including the model definition and training algorithm.
|	├── kmeans_tree.py === Run KMeans on geographic coordinates to form a hierarchical tree.
|	├── lambdarank.py === The lambda rank loss function.
|	└── train_portion.py === Split the training data into various portions.
├── nnue  === The native inference engine implemented in C++
|	└── v19
|	  ├── nnue_avx2.h  === The header file for essential data structures and NNUE engine.
|		├── nnue.cpp  === The main executable source file.
|	  ├── nnue_avx2_unsupervised.h  === unsupervised version.
|		└── nnue_unsupervised.cpp === unsupervised version.
├── result
|	└── evaluation.py  === A simple script for evaluating the outcomes of baselines
└── readme.md

```

## Thanks for Reading!