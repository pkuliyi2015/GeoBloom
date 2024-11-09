# GeoBloom

- This is the official repository of the paper:  *GeoBloom: Revitalizing Lightweight Models for Geographic Information Retrieval*

## Update
- 2024/11/09:
  - We fixed the bug in bloom_filter_tree.py and the training process. Now the effectiveness is improved comparing to the paper results.
  - We replace the dense training with the index-based training based on a hybrid dense-sparse training strategy at different depths of the tree.
  - We implement a custom CUDA kernel to compute the intersection of Bloom filters, where the query is sparse and node is either sparse (sorted) or dense (float32 or float16). 
  - Now the training is 10x faster (7 minutes / epoch for Beijing, Shanghai, and GeoGLUE_clean; 18 minutes for GeoGLUE. Tested on an NVIDIA V100 32GB GPU). 
  - We update this training time in the paper revision.
- 2024/10/30: We now release the repository on GitHub, including the Bloom filters and the anonymized (noised) locations of Beijing and Shanghai datasets.
- We found that Anonymous GitHub has a serious bug, which doesn't sync nested folder (nnue/v19/) in our repo. We now manually fixed the bug by adding a placeholder file.

## Quick Start

- With this repository, you can
  - Download the GeoGLUE and GeoGLUE-clean datasets in the paper.
  - Download the Bloom filters and the (anonymized) locations of Beijing and Shanghai datasets.
  - Quickly reproduce all effectiveness results in unsupervised and supervised settings.
  - Train from scratch for the supervised effectiveness of GeoBloom.
  - Compare with baselines BM25-D, BERT, OpenAI-D and DPR-D.
  - Measure the inference time, memory, and disk space usage.
  - Analyze the outcomes of GeoBloom by cases.

### Environment Preparation

While our framework can run on various devices, to ensure it works smoothly, we recommend:

- A computer **Linux** system (we use Ubuntu 20.04) with CPUs supporting **AVX-2** instruction set.

  - Currently, this code repo doesn't support other systems due to our heavy workload.
  - It should be easy to transfer our code to these systems as well.
- Please use **Miniconda or Anaconda**
- We use Python 3.11.5. Lower versions are not tested.

  - conda create -n GeoBloom python==3.11.5

Our methods required:

- PyTorch>=2.1.1
- tqdm
- scikit-learn==1.4.2
- xxhash==3.4.1
- torch-scatter==2.1.2
- torch-sparse==0.6.18

### Dataset Preparation

- Clone this repository.
- Decompress the GeoGLUE.7z and GeoGLUE_clean.7z in the data folder.
- The Beijing and Shanghai raw datasets are not included in this repo due to license constraints. However,
  - We provide the Bloom filters and the anonymized locations so that you can quickly reproduce the results of GeoBloom.
  - We also provide two strong baselines, i.e, the OpenAI and DPR embeddings, for your convenience.
  - If you are interested in the raw datasets, please contact us via your institution email to request access under the license constraints.
  - Email: liyi0067@e.ntu.edu.sg, shang006@e.ntu.edu.sg, or 386029724@qq.com (in case of institutional email is not accessible).
- (Optional) We provide the OpenAI embeddings and trained DPR embeddings in Beijing and Shanghai for your convenience. [Google Drive](https://drive.google.com/drive/folders/1GeB7A90cocWvUJGysVxyK1pqyHKcfX5B?usp=drive_link)

#### Notes

- The **GeoGLUE** dataset is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?spm=5176.12282016.0.0.63b47586Niz8D0) license. You should strictly follow the license to use the dataset.
  - The original dataset is from [modelscope.cn](https://www.modelscope.cn/datasets/iic/GeoGLUE/summary)
  - We extract the text, latitude, longitude, and ground truth ids, project them into Euclidean space with pyproj library:
    - proj = pyproj.Transformer.from_crs( 'epsg:4326', 'epsg:2385')
- As GeoGLUE contains over 50% of fake POIs, we align their queries to real POIs, which forms a new dataset GeoGLUE-clean.
  - The text, latitude, and longitude of each query remain unchanged.
  - The ground truth id now point to our real POIs.
  - Those queries that can't find a real POI in our dataset are excluded.
- We provide the generation script of the GeoGLUE-clean dataset in data/geoglue_clean.py. You can run it to generate the dataset with your own OpenAI key. Please be aware that it is slow and expensive (13 hours, $100+).

## **Experiments**

- To replicate GeoBloom, you must compile the NNUE engine first as all the experiments rely on it.

  - **g++ nnue/v19/nnue.cpp -o nnue/v19/nnue -pthread -mavx2 -O3 -fno-tree-vectorize**

### Unsupervised Effectiveness

GeoBloom supports unsupervised effectiveness evaluation without labeled queries.

#### Steps:

- Compile the nnue engine as instructed above.
- For GeoGLUE and GeoGLUE-clean, you need to preprocess the dataset into Bloom filters and construct the tree:

  - **python model/dataset.py --dataset GeoGLUE**
  - **python model/dataset.py --dataset GeoGLUE_clean**
- For Beijing and Shanghai, you don't need to preprocess the dataset.
- Retrieve the results:

  - **nnue/v19/nnue Beijing unsupervised 8 400-400-400-400**
  - **nnue/v19/nnue Shanghai unsupervised 8 400-400-400-400**
  - **nnue/v19/nnue GeoGLUE_clean unsupervised 8 800-800-800-800**
  - **nnue/v19/nnue GeoGLUE unsupervised 8 6000-6000-6000-6000**

Following the steps, you should see something like this:

- Beijing

```
Beam width: 400 400 400 400 Loading dataset from path: data_bin/Beijing/
Use unsupervised model.
Allocating 45.834 MB for 23467 bloom filters on the tree...
Allocating 14.9438 MB for 122420 leaf nodes at depth 0...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 19.771s, Query Per Second: 854.737
=============== Intermediate Recall Scores ==============
0.986885        0.881396        0.808939        0.761202
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.637374        0.593689        0.474783        0.406415
=========================================================
```

- GeoGLUE_clean

```
Beam width: 800 800 800 800 Loading dataset from path: data_bin/GeoGLUE_clean/
Use unsupervised model.
Allocating 277.549 MB for 142105 bloom filters on the tree...
Allocating 94.8846 MB for 777295 leaf nodes at depth 0...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 67.9288s, Query Per Second: 178.967
=============== Intermediate Recall Scores ==============
0.968331        0.890022        0.814099        0.733816
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.529983        0.471251        0.327060        0.236818
=========================================================
```

Note:

- The intermediate recall scores show the recall rates of beam search at each tree depth. It gives insights into the beam width selection.
- We support multi-threading (8 threads in the above example) for your convenience, but the QPS is always evaluated per thread.
- Here, we simply choose a unified beam width (e.g., 4000) and it can be quite "slow". You can set different beam widths at each tree level to obtain faster speed.

### Supervised Effectiveness

The training requires an NVIDIA GPU. We tested on NVIDIA V100 32GB, which can run the script without issues.

#### Steps:
- Compile the nnue engine as instructed above.
- Run the training script. It supports all datasets. For Beijing and Shanghai, we recommend epochs 15. For GeoGLUE and GeoGLUE-clean, we recommend epochs 8.

  - **python model/geobloom_v19.py --dataset Beijing --epochs 15**
  - **python model/geobloom_v19.py --dataset Shanghai --epochs 15**
  - **python model/geobloom_v19.py --dataset GeoGLUE_clean --epochs 8**
  - **python model/geobloom_v19.py --dataset GeoGLUE --epochs 8**
- Retrieve the results:

  - **nnue/v19/nnue Beijing test 8 400-400-400-400**
  - **nnue/v19/nnue GeoGLUE_clean test 8 800-800-800-800**

Following the steps, you should see something like this:

- Beijing

```
Beam width: 400 400 400 400 Loading dataset from path: data_bin/Beijing/
Allocating 45.834 MB for 23467 bloom filters on the tree...
Allocating 14.9438 MB for 122420 leaf nodes at depth 0...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 19.7889s, Query Per Second: 853.965
=============== Intermediate Recall Scores ==============
0.992959        0.943726        0.890166        0.859498
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.781922        0.734642        0.595412        0.512516
=========================================================
Predictions saved to data_bin/Beijing/test_nodes.bin
```

- GeoGLUE_clean

```
Beam width: 1000 1000 1000 1000 Loading dataset from path: data_bin/GeoGLUE_clean/
Allocating 217.443 MB for 111331 bloom filters on the tree...
Infering test candidates...
Searching 10000th query...
Total search time of all threads: 62.0868s, Query Per Second: 195.806
=============== Intermediate Recall Scores ==============
0.994078        0.946039        0.913712        0.886896
====================== Evaluation =======================
Recall@20        Recall@10       NDCG@5          NDCG@1
0.772559        0.717776        0.550100        0.427161
=========================================================
Predictions saved to data_bin/GeoGLUE_clean/test_nodes.bin
```

- The above QPS is obtained from an old Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz.
- If you use a more powerful CPU, like an i9-10900X @ 3.70GHz in the paper, the QPS will be at least twice more.

### Varying Training Data Portions

Our model improves stably and consistently from 2% to 100% training data usage. We currently only support GeoGLUE and GeoGLUE-clean, but is simple (and straightforward) to extend to Beijing and Shanghai by shuffle and slicing the queries in model/dataset.py, line 74.

#### Steps:

- Run the dataset split script.
  - **python model/train_portion.py --dataset GeoGLUE_clean**
- Training the model on that portion.
  - **python model/geobloom.py --dataset GeoGLUE_clean --portion 0.02 --epochs 8**
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
- We use VmHWM to measure the memory usage. Simply run
  - **nnue/v19/nnue Beijing memory**
  - You should see something like this:

```
Initial VmHWM: 1 MB
POI dataset Memory Usage: 23.6836 MB
NNUE model Memory Usage: 21.7539 MB
Allocating 45.834 MB for 23467 bloom filters on the tree...
Allocating 14.9438 MB for 122420 leaf nodes at depth 0...
Bloom Filter Tree Memory Usage: 67.5469 MB
Node embeddings Memory Usage: 17.7891 MB
Final VmHWM: 132 MB
```

- The disk usage of our model ise the sum of poi.bin, node_v19.bin, nnue_v19.bin, and tree.bin in the data_bin folder.
  - In NNUE quantization scheme, not every parameter is stored in 16-bit numbers, preventing a fair comparison.
  - We thereby provide an option, by running.
    - python model/geobloom.py --dataset GeoGLUE_clean --task size_test
  - It converts everything into 16-bit. Then, you can directly compare with PLMs (e.g., DPR_D), which need much larger sizes when stored in 16-bit.

You should see something like this:

```
======== File Sizes in MB ========
Components
Model           20.10
Embeddings      54.24
Bloom Filters   119.62
Tree            3.82
Total - 197.77
==================================
```

### Baselines

- Due to page limit, we omit the alpha values of Baselines in the paper. Here is the table for your convenience.

| Method     | Meituan-Beijing | Meituan-Shanghai | GeoGLUE | GeoGLUE-clean |
|------------|------------------|------------------|---------|---------------|
| BM25-D     | 0.4              | 0.4              | 0.9     | 0.9           |
| SIF-D      | 0.4              | 0.4              | 0.9     | 0.9           |
| BERT-D     | 0.4              | 0.4              | 0.8     | 0.8           |
| OpenAI-D   | 0.3              | 0.3              | 0.8     | 0.8           |
| DRMM-D     | 0.7              | 0.7              | 0.9     | 0.9           |
| ARC-I-D    | 0.1              | 0.1              | 0.8     | 0.8           |
| DPR-D      | 0.3              | 0.3              | 0.9     | 0.9           |

- For your convenience, we provide the baselines BM25-D, OpenAI-D, and DPR-D, which are relatively powerful in our tasks.
- You can quickly evaluate all baselines versus our method, but these methods require additional packages (jieba, acceleration, transformers,...)
- For unreleased Beijing and Shanghai, we provide the representations of OpenAI and DPR. Please use the following Google Drive link to download them.
  - https://drive.google.com/drive/folders/1GeB7A90cocWvUJGysVxyK1pqyHKcfX5B?usp=drive_link
  - Please use the deserialize() function in dataset.py to load the noised locations for the two datasets.

- We strictly conduct experiments without any cherry-picking. But our GeoBloom's result should be higher than that reported in our paper due to our recent fix in the FP32 serialization.
  - You need OpenAI key for the OpenAI baseline, and it will cost around $2. We used the API on March 4th, 2024. Until that time, it was still not as competitive as BM25 on the provided datasets.
  - DPR_D requires negative sampling. We use the BM25_D to generate hard negative samples for it.

## Future Development

- Currently, the training is significantly slower than the inference, as we treat Bloom filters as dense vectors and only rely on PyTorch torch.compile for acceleration. It should be much faster if we implement custom CUDA kernels.
- Sophisticated Bloom filters and index design can be applied to this framework to achieve better space efficiency, making it possible to deploy as offline map apps.

## Contents

```
GeoBloom
├── baselines
│   ├── BM25/          # BM25, BM25-D with jieba and rank_bm25
│   ├── BERT/          # Huggingface transformers
│   ├── OpenAI/        # OpenAI API
│   └── DPR/           # Modified from NanoDPR https://github.com/Hannibal046/nanoDPR
├── ckpt/              # Store the checkpoints previously trained
├── cuda/              # Custom CUDA kernels for fast intersection computing
├── data/
│   ├── GeoGLUE.7z    # Please decompress the file from this repo
│   ├── GeoGLUE_clean.7z # Please decompress the file from this repo
│   └── geoglue_clean.py # Script to generate clean dataset with OpenAI
├── data_bin/          # The model will generate binary files here for C++ inference
│   ├── Beijing/       # Binary files for Beijing dataset
│   ├── Shanghai/      # Binary files for Shanghai dataset
│   ├── GeoGLUE/      # Binary files for GeoGLUE dataset
│   └── GeoGLUE_clean/ # Binary files for GeoGLUE-clean dataset
├── model/
│   ├── bloom_filter.py        # The vanilla Bloom filter generator
│   ├── bloom_filter_tree.py   # Organize the Bloom filter into a tree-based index
│   ├── dataset.py            # PyTorch dataset, can preprocess dataset into Bloom filters
│   ├── geobloom_v19.py       # Main script with model definition and training
│   ├── kmeans_tree.py        # Run KMeans on coordinates to form hierarchical tree
│   ├── lambdarank.py         # The lambda rank loss function
│   └── train_portion.py      # Split training data into various portions
├── nnue/                     # Native inference engine in C++
│   └── v19/
│       ├── nnue_avx2.h              # Header for data structures and NNUE engine
│       └── nnue.cpp                 # Main executable source file
├── result/
│   ├── analyze.py            # Easily analyze the outcomes of GeoBloom
│   └── evaluation.py         # Script for evaluating baseline outcomes
└── README.md                 # Project documentation and instructions

```

## Thanks for Reading!
