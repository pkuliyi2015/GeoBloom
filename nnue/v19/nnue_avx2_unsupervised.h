#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

#define USE_CONTEXT
using namespace std;

constexpr double INF = numeric_limits<double>::infinity();
constexpr int MAX_AVX2_REGISTERS = 16; // We can use 16 AVX2 registers at most
constexpr int NUM_SLICES = 4;
constexpr int NUM_BITS = 8192;
constexpr int BF_SIZE =  NUM_BITS * NUM_SLICES / 32;
constexpr int MAX_TOP_CONTEXT = 30;

// Utility Functions
// The following code is migrated from https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md

inline __m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i sum128lo = _mm256_castsi256_si128(sum0);
    __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
};

inline __m128i m256_haddx4_no_bias(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i sum128lo = _mm256_castsi256_si128(sum0);
    __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(sum128lo, sum128hi);
};


void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
    #if defined (USE_VNNI)
        // This does exactly the same thing as explained below but in one instruction.
        acc = _mm256_dpbusd_epi32(acc, a, b);
    #else
        // Multiply a * b and accumulate neighbouring outputs into int16 values
        __m256i product0 = _mm256_maddubs_epi16(a, b);

        // Multiply product0 by 1 (idempotent) and accumulate neighbouring outputs into int32 values
        __m256i one = _mm256_set1_epi16(1);
        product0 = _mm256_madd_epi16(product0, one);

        // Add to the main int32 accumulator.
        acc = _mm256_add_epi32(acc, product0);
    #endif
};


void linear32(
    const int8_t * weights,  // the layer to use. We have two: L_1, L_2
    const int32_t * biases,
    int32_t*           output, // the already allocated storage for the result
    const int8_t*      input   // the input, which is the output of the previous ClippedReLU layer
) {
    constexpr int num_inputs = 32;
    constexpr int num_outputs = 32;
    constexpr int register_width = 256 / 8;
    const int num_in_chunks = num_inputs / register_width;
    const int num_out_chunks = num_outputs / 4;

    for (int i = 0; i < num_out_chunks; ++i) {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2D array.
        const int offset0 = (i * 4 + 0) * num_inputs;
        const int offset1 = (i * 4 + 1) * num_inputs;
        const int offset2 = (i * 4 + 2) * num_inputs;
        const int offset3 = (i * 4 + 3) * num_inputs;

        // Accumulation starts from 0, we add the bias only at the end.
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (int j = 0; j < num_in_chunks; ++j) {
            // We unroll by 4 so that we can reuse this value, reducing the number of
            // memory operations required.
            const __m256i in = _mm256_load_si256((__m256i *)&input[j * register_width]);

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            // For definition see below.
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256((__m256i *)&weights[offset0 + j * register_width]));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256((__m256i *)&weights[offset1 + j * register_width]));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256((__m256i *)&weights[offset2 + j * register_width]));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256((__m256i *)&weights[offset3 + j * register_width]));
        }

        const __m128i bias = _mm_load_si128((__m128i *) & biases[i * 4]);
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        // For the definition see below.
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
        // Here we account for the weights scaling.
        outval = _mm_srai_epi32(outval, 6);
        _mm_store_si128((__m128i *)&output[i * 4], outval);
    }
}


__m256i __crelu32(  // no need to have any layer structure, we just need the number of elements // the already allocated storage for the result
    const int32_t* input   // the input, which is the output of the previous linear layer
) {
    constexpr int size = 32;
    constexpr int in_register_width = 256 / 32;
    constexpr int out_register_width = 256 / 8;
    const int num_out_chunks = size / out_register_width;
    const __m256i zero    = _mm256_setzero_si256();
    const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i in0 =
        _mm256_packs_epi32(
            _mm256_load_si256((__m256i *)&input[0]),
            _mm256_load_si256((__m256i *)&input[in_register_width])
        );
    const __m256i in1 =
        _mm256_packs_epi32(
            _mm256_load_si256((__m256i *)&input[(2) * in_register_width]),
            _mm256_load_si256((__m256i *)&input[(3) * in_register_width])
        );

    const __m256i result =
        _mm256_permutevar8x32_epi32(
            _mm256_max_epi8(
                _mm256_packs_epi16(in0, in1),
                zero
            ),
            control
        );

    return result;
}

void crelu32(int8_t*  output, // the already allocated storage for the result
    const int32_t* input){
    _mm256_store_si256((__m256i *) output, __crelu32(input));
}


// The following code is originally written.
// Utility function to calculate mean and standard deviation
void calc_mean_std(int len, int* arr, double& mean, double& std) {
    // Calculate mean. NOTE: Use 0.0 instead of 0 to do implicit type conversion and avoid overflow
    mean = static_cast<double>(accumulate(arr, arr + len, static_cast<long long>(0))) / len;
    // Calculate standard deviation
    double sq_sum = inner_product(arr, arr + len, arr, 0.0,
        [](double a, double b) { return a + b; },
        [mean](int a, int b) { return (a - mean) * (b - mean); });
    std = sqrt(sq_sum / (len - 1));
    // Handle the case where std is 0
    if (std == 0) {
        std = 1;
    }
}

// Sigmoid function
inline double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// ReLU function
inline double relu(double x) {
    return max(0.0, x);
}

class BloomFilter;

class Dataset {
public:
    vector<vector<uint16_t>> bloom_filters;
    vector<pair<double, double>> locations;
    vector<vector<uint32_t>> truths;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
    bool has_truth_flag = false;

    bool load(const string& file_dir) {
        ifstream file(file_dir, ios::binary);

        if (!file) {
            cerr << "Cannot open file: " << file_dir << endl;
            return false;
        }

        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        uint16_t has_truth;
        file.read(reinterpret_cast<char*>(&has_truth), sizeof(has_truth));
        has_truth_flag = has_truth == 1;

        bloom_filters.resize(num_rows);
        vector<uint16_t> bloom_filter_lengths(num_rows);
        file.read(reinterpret_cast<char*>(bloom_filter_lengths.data()), num_rows * sizeof(uint16_t));

        for (uint32_t i = 0; i < num_rows; ++i) {
            bloom_filters[i].resize(bloom_filter_lengths[i]);
            file.read(reinterpret_cast<char*>(bloom_filters[i].data()), bloom_filter_lengths[i] * sizeof(uint16_t));
            // sort the bloom filter for fast intersection calculation at leaf nodes
            sort(bloom_filters[i].begin(), bloom_filters[i].end());
        }

        locations.resize(num_rows);
        file.read(reinterpret_cast<char*>(locations.data()), num_rows * sizeof(double) * 2);

        if (has_truth_flag) {
            truths.resize(num_rows);
            for (uint32_t i = 0; i < num_rows; ++i) {
                uint16_t truth_len;
                file.read(reinterpret_cast<char*>(&truth_len), sizeof(truth_len));
                truths[i].resize(truth_len);
                file.read(reinterpret_cast<char*>(truths[i].data()), truth_len * sizeof(uint32_t));
            }
        }
        file.close();

        return true;
    }
};

// NOTE: We shouldn't use bitset as it is too slow comparing to our implementation
class BloomFilter {
public:
    alignas(64) uint32_t * bits = nullptr;

    BloomFilter() {}

    BloomFilter(uint32_t * bits) {
        this->bits = bits;
    }

    BloomFilter operator|= (const BloomFilter &other) {
        // OR operation in avx2
        for (int i = 0; i < BF_SIZE; i+=8) {
            __m256i data = _mm256_load_si256((__m256i *) (this->bits + i));
            __m256i result = _mm256_or_si256(data, _mm256_load_si256((__m256i *) (other.bits + i)));
            _mm256_store_si256((__m256i *) (this->bits + i), result);
        }
        return *this;
    }

    BloomFilter operator^= (const BloomFilter &other) {
        // XOR operation in avx2
        for (int i = 0; i <  BF_SIZE; i+=8) {
            __m256i data = _mm256_loadu_si256((__m256i *) (this->bits + i));
            __m256i result = _mm256_xor_si256(data, _mm256_loadu_si256((__m256i *) (other.bits + i)));
            _mm256_storeu_si256((__m256i *) (this->bits + i), result);
        }
        return *this;
    }

    inline void set(uint16_t idx) {
        bits[idx >> 5] |= (1 << (idx & 31));
    }

    inline uint64_t test(uint16_t idx) {
        return bits[idx >> 5] & (1 << (idx & 31));
    }

    inline uint64_t fast_test(int slice, uint32_t bit){
        return bits[slice] & bit;
    }

    inline int fast_loop_test(int query_length, int * query_slices, uint32_t * query_bits) {
        int result = 0;
        int query_len = query_length;
        for(int i=0; i < query_len; i++){
            if(this->bits[query_slices[i]] & query_bits[i]){
                result++;
            }
        }
        return result;
    }

    inline int fast_avx2_test(int query_length, int * query_slices, uint32_t * query_bits) {
        // NOTE: It seems that the AVX2 version is slower than the fast_loop_test version
        int result = 0;
        __m256i zero = _mm256_setzero_si256();
        for(int i=0; i < query_length; i+=8){
            // We need to load 32 bytes at once from query_slices[i] to query_slices[i+31]
            __m256i indices = _mm256_load_si256((__m256i *)(query_slices + i));
            // Gather 32-bit elements from bits based on indices
            __m256i data = _mm256_i32gather_epi32((int *)bits, indices, 4);
            __m256i query = _mm256_load_si256((__m256i *) (query_bits + i));
            __m256i result_vec = _mm256_and_si256(data, query);
            __m256i cmp_mask = _mm256_cmpeq_epi8(result_vec, zero);
            // Create a mask of the comparison results
            uint32_t mask = _mm256_movemask_epi8(cmp_mask);
            // Count non-zero bytes
            result += 32 - __builtin_popcount(mask); // Each bit in mask corresponds to a byte in result_vec
        }
        return result;
    }

};


class TreeNode {
    // A normal BERT 768 embedding requires 768 * 32 = 24576 bits
    // In this 4-bit version, our representaion is slightly larger (NUM_SLICES * NUM_BITS + 2048 bits)
    // but we may use smaller bloom filter for further memory saving.
public:
    BloomFilter bloom_filter;
    vector<uint16_t> * bloom_filter_compressed = nullptr;
    double location[2];
    double radius;
    uint32_t node_idx;
    TreeNode* parent;
    vector<TreeNode*> * children;
    double boundary[4]; // min_lat, max_lat, min_lon, max_lon

    // The following embedding is computed  in the searching process
    int32_t * embedding = nullptr;

    TreeNode() {}
    ~TreeNode() {
        if (this->children != nullptr) {
            delete this->children;
        }
    }

    void set_leaf_node(vector<uint16_t>& bloom_filter, pair<double, double>& location, uint32_t node_idx) {
        this->bloom_filter_compressed = &bloom_filter;
        double lat = location.first;
        double lon = location.second;
        this->location[0] = lat;
        this->location[1] = lon;
        this->node_idx = node_idx;
        this->boundary[0] = lat;
        this->boundary[1] = lat;
        this->boundary[2] = lon;
        this->boundary[3] = lon;
    }

    void set_branch_node(uint32_t * bits, uint32_t node_idx) {
        this->bloom_filter.bits = bits;
        this->node_idx = node_idx;
        this->boundary[0] = INF;
        this->boundary[1] = -INF;
        this->boundary[2] = INF;
        this->boundary[3] = -INF;
    }


    void add_child(TreeNode & child) {
        if (this->children == nullptr) {
            this->children = new vector<TreeNode*>();
        }
        this->children->push_back(&child);
        if (child.bloom_filter_compressed != nullptr) {
            for (uint16_t b: *child.bloom_filter_compressed) {
                this->bloom_filter.set(b);
            }
        } else {
            this->bloom_filter |= child.bloom_filter;
        }
        child.parent = this;
        this->boundary[0] = min(this->boundary[0], child.boundary[0]);
        this->boundary[1] = max(this->boundary[1], child.boundary[1]);
        this->boundary[2] = min(this->boundary[2], child.boundary[2]);
        this->boundary[3] = max(this->boundary[3], child.boundary[3]);
    }

    void compute_radius() {
        double lat_diff = this->boundary[1] - this->boundary[0];
        double lon_diff = this->boundary[3] - this->boundary[2];
        this->radius = sqrt(lat_diff * lat_diff + lon_diff * lon_diff) / 2.0;
        this->location[0] = (this->boundary[0] + this->boundary[1]) / 2.0;
        this->location[1] = (this->boundary[2] + this->boundary[3]) / 2.0;
    }
};

class NNUEBottleNeck {
public:
    alignas(64) int8_t weight [32][256];

    void load(ifstream& file) {
        file.read(reinterpret_cast<char*>(weight), sizeof(weight));
    }

    void unsupervised(){
    }
};



class NNUEReweighter {
public:
    alignas(64) int8_t l1_weight [32][32];
    alignas(64) int32_t l1_bias [32];
    alignas(64) int8_t l2_weight [NUM_SLICES * NUM_BITS][32];

    void load(ifstream& file) {
        file.read(reinterpret_cast<char*>(l1_weight), sizeof(l1_weight));
        file.read(reinterpret_cast<char*>(l1_bias), sizeof(l1_bias));
        file.read(reinterpret_cast<char*>(l2_weight), sizeof(l2_weight));
    }

    bool empty(){
        bool all_zeros = true;
        for (int i = 0; i < NUM_SLICES * NUM_BITS; ++i) {
            for (int j = 0; j < 32; ++j) {
                if (l2_weight[i][j] != 0) {
                    all_zeros = false;
                    break;
                }
            }
        }
        return all_zeros;
    }

    void unsupervised(){
        for (int i = 0; i < NUM_SLICES * NUM_BITS; ++i) {
            for (int j = 0; j < 32; ++j) {
                l2_weight[i][j] = 0;
            }
        }
    }
};

class NNUEResidual {
public:
    alignas(64) int8_t l1_weight [32][32];
    alignas(64) int32_t l1_bias [32];
    alignas(64) int8_t l2_weight [32];

    void load(ifstream& file) {
        file.read(reinterpret_cast<char*>(l1_weight), sizeof(l1_weight));
        file.read(reinterpret_cast<char*>(l1_bias), sizeof(l1_bias));
        file.read(reinterpret_cast<char*>(l2_weight), sizeof(l2_weight));
    }

    bool empty(){
        bool all_zeros = true;
        for (int i = 0; i < 32; ++i) {
            if (l2_weight[i] != 0) {
                all_zeros = false;
                break;
            }
        }
        return all_zeros;
    }

    void unsupervised(){
        for (int i = 0; i < 32; ++i) {
            l2_weight[i] = 0;
        }
    }
};


class NNUE {
public:
    alignas(64) uint16_t depth;
    alignas(64) int32_t encoder_weight[NUM_SLICES * NUM_BITS][256];
    alignas(64) int32_t encoder_bias[256];

    alignas(64) NNUEBottleNeck * bottleneck = nullptr;
    
    alignas(64) NNUEReweighter * rank = nullptr;
    alignas(64) NNUEReweighter * context_select = nullptr;
    alignas(64) NNUEReweighter * context_rank = nullptr;
    alignas(64) NNUEResidual * residual = nullptr;

    ~NNUE() {
        if (rank != nullptr) {
            free(rank);
        }
        if (context_select != nullptr) {
            free(context_select);
        }
        if (context_rank != nullptr) {
            free(context_rank);
        }
        if (residual != nullptr) {
            free(residual);
        }
    }

    vector<float> a, b, c, d;

    bool load(const string& file_dir) {
        ifstream file(file_dir, ios::binary);
        if (!file) {
            cerr << "Cannot open file: " << file_dir << endl;
            exit(1);
            return false;
        }
        file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
        a.resize(depth);
        b.resize(depth);
        c.resize(depth);
        d.resize(depth);

        file.read(reinterpret_cast<char*>(encoder_weight), sizeof(encoder_weight));
        file.read(reinterpret_cast<char*>(encoder_bias), sizeof(encoder_bias));

        bottleneck = (NNUEBottleNeck *)aligned_alloc(64, sizeof(NNUEBottleNeck) * depth);
        for (int i = 0; i < depth; ++i) {
            bottleneck[i].load(file);
        }

        // memory allocation & weight loading
        rank = (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        for (int i = 0; i < depth; ++i) {
            rank[i].load(file);
            if (rank[i].empty()) {
                cout << "Rank head" << i << " untrained" << endl;
            }
        }
        context_select = (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        for (int i = 0; i < depth; ++i) {
            context_select[i].load(file);
            if (context_select[i].empty()) {
                cout << "Context select head" << i << " untrained" << endl;
            }
        }
        context_rank =  (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        for (int i = 0; i < depth; ++i) {
            context_rank[i].load(file);
            if (context_rank[i].empty()) {
                cout << "Context rank head" << i << " untrained" << endl;
            }
        }
        residual = (NNUEResidual *)aligned_alloc(64, sizeof(NNUEResidual) * depth);
        for (int i = 0; i < depth; ++i) {
            residual[i].load(file);
            if (residual[i].empty()) {
                cout << "Residual head" << i << " untrained" << endl;
            }
        }

        file.read(reinterpret_cast<char*>(a.data()), depth * sizeof(float));
        file.read(reinterpret_cast<char*>(b.data()), depth * sizeof(float));
        file.read(reinterpret_cast<char*>(c.data()), depth * sizeof(float));
        file.read(reinterpret_cast<char*>(d.data()), depth * sizeof(float));

        file.close();
        return true;
    }

    void unsupervised(){
        cout << "Use unsupervised model." << endl;
        depth = 4;
        bottleneck = (NNUEBottleNeck *)aligned_alloc(64, sizeof(NNUEBottleNeck) * depth);
        rank = (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        context_select = (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        context_rank =  (NNUEReweighter *)aligned_alloc(64, sizeof(NNUEReweighter) * depth);
        residual = (NNUEResidual *)aligned_alloc(64, sizeof(NNUEResidual) * depth);

        for (int i = 0; i < depth; ++i) {
            bottleneck[i].unsupervised();
            rank[i].unsupervised();
            context_select[i].unsupervised();
            context_rank[i].unsupervised();
            residual[i].unsupervised();
        }

        a.resize(depth);
        b.resize(depth);
        c.resize(depth);
        d.resize(depth);

        for (int i = 0; i < depth; ++i) {
            a[i] = 1;
            b[i] = 0;
            c[i] = 1;
            d[i] = 1;
        }
    }

    inline void encode_query_avx2(vector<uint16_t> &bloom_filter, int32_t * query_embedding) {
        // This is the AVX2 version of the query encoder
        // The weights are 256 int32_t. Every time we process 8 numbers
        // But there are only 16 registers in AVX2, so we need to loop twice
        constexpr int register_width = 256 / 32;
        constexpr int ideal_register_num = 256 / register_width;
        static_assert(ideal_register_num % MAX_AVX2_REGISTERS == 0, "The ideal register number should be divisible by MAX_AVX2_REGISTERS");
        constexpr int loop_num = ideal_register_num / MAX_AVX2_REGISTERS;
        constexpr int num_chunks = MAX_AVX2_REGISTERS;
        alignas (64) int32_t buffer[256];
        int normalization = bloom_filter.size(); // In our case, it always > 0
        alignas (64) int bound = normalization * 127;
        __m256i regs[num_chunks];
        
        for (int loop=0; loop<256; loop+=128) {
            for (int i = 0; i < num_chunks; ++i) {
                regs[i] = _mm256_load_si256((__m256i *) & encoder_bias[i * 8 + loop]);
            }
            for (uint16_t b: bloom_filter) {
                for (int i = 0; i < num_chunks; ++i) {
                    regs[i] = _mm256_add_epi32(regs[i], _mm256_load_si256((__m256i *) &encoder_weight[b][i * 8 + loop]));
                }
            }
            // clipping to [0, bound]
            for (int i = 0; i < num_chunks; ++i) {
                regs[i] = _mm256_min_epi32(regs[i], _mm256_set1_epi32(bound));
                regs[i] = _mm256_max_epi32(regs[i], _mm256_set1_epi32(0));
            }
            for (int i = 0; i < num_chunks; ++i) {
                _mm256_store_si256((__m256i *) & buffer[i * 8 + loop], regs[i]);
            }
        }
        // The bit normalization will bring significant precision loss if we do it right after the
        // encoder. So we do it after the l1 layer matmul.
        // We do looped multiplication instead of avx2, which greatly mitigate precision loss.
        // v19: The query_embedding is now different across different layers.
        // We use a 2D array to store the query embedding for each layer.
        for (int d = 0; d < this->depth; d++){
            for (int i = 0; i < 32; i++) {
                int32_t sum = 0;
                for (int j = 0; j < 256; j++) {
                    sum += bottleneck[d].weight[i][j] * buffer[j];
                }
                query_embedding[d * 32 + i] = sum / normalization;
            }
        }
        // NOTE: We don't need to add the bias here as it has been added to the node embedding.
    }


    inline int l2_avx2(
        const int32_t * query_embedding, 
        const int32_t * node_embedding, 
        const int depth,
        int8_t * rank_hidden,
        int8_t * context_select_hidden,
        int8_t * context_rank_hidden
    ){
        alignas (64) int32_t buffer[32];
        constexpr int num_per_groups = 256 / 32;
        constexpr int groups = 32 / num_per_groups;
        for (int i = 0; i < groups; ++i) {
            __m256i data = _mm256_load_si256((__m256i *) (query_embedding + i * num_per_groups));
            __m256i result = _mm256_add_epi32(data, _mm256_load_si256((__m256i *) (node_embedding + i * num_per_groups)));
            // do weight scaling here
            result = _mm256_srai_epi32(result, 6);
            _mm256_store_si256((__m256i *) (buffer + i * 8), result);
        }
        // This hidden will be shared by the rank, context select, context rank and residual head.
        alignas (64) int8_t shared_hidden[32];
        crelu32(shared_hidden, buffer);
        // Compute the rank l2.
        linear32(&rank[depth].l1_weight[0][0], rank[depth].l1_bias, buffer, shared_hidden);
        crelu32(rank_hidden, buffer);

        // Compute the context select l2.
        linear32(&context_select[depth].l1_weight[0][0], context_select[depth].l1_bias, buffer, shared_hidden);
        crelu32(context_select_hidden, buffer);

        // Compute the context rank l2.
        linear32(&context_rank[depth].l1_weight[0][0], context_rank[depth].l1_bias, buffer, shared_hidden);
        crelu32(context_rank_hidden, buffer);

        // Finally compute the residual l2.
        linear32(&residual[depth].l1_weight[0][0], residual[depth].l1_bias, buffer, shared_hidden);
        __m256i hidden_res = __crelu32(buffer);
        __m256i res_weight = _mm256_load_si256((__m256i *) &residual[depth].l2_weight[0]);
        __m256i res_out = _mm256_maddubs_epi16(hidden_res, res_weight);
        res_out = _mm256_madd_epi16(res_out, _mm256_set1_epi16(1));
        // We now get 8 int32_t. Sum them up to get the final residual score
        _mm256_store_si256((__m256i *) buffer, res_out);
        int res_score = accumulate(buffer, buffer + 8, 0);

        return res_score;
    }

    inline void l3_avx2(
        const TreeNode * node, 
        const int query_length, 
        const uint16_t * query_numbers, 
        const int depth,
        const int8_t * rank_hidden,
        const int8_t * context_select_hidden,
        int & intersection_score,
        int & context_select_score,
        int & unmatched_bits_length,
        uint16_t * unmatched_bits
    ) {
        // v18: This function now calculates the intersection score and the context score simultaneously,
        // and write the unmatched bits to a given buffer.
        int common_bits = 0;
        // __m256i hidden_vec = _mm256_load_si256((__m256i *) rank_hidden);
        // __m256i positive_sum = _mm256_setzero_si256();
        // __m256i negative_sum = _mm256_setzero_si256();
        // __m256i context_select_hidden_vec = _mm256_load_si256((__m256i *) context_select_hidden);
        // __m256i context_positive_sum = _mm256_setzero_si256();
        // int8_t (& rank_weight)[32768][32] = rank[depth].l2_weight;
        // int8_t (& context_select_weight)[32768][32] = context_select[depth].l2_weight;
        // We need to loop through the query numbers to calculate the intersection score
        // If the node bloom filter is not compressed, we directly use the bits
        if (node->bloom_filter.bits != nullptr){
            uint32_t * bits = node->bloom_filter.bits;
            for(int i = 0; i < query_length; i++){
                uint16_t query_number = query_numbers[i];
                if (bits[query_number >> 5] &  (1 << (query_number & 31))) {
                    // __m256i weight_vec = _mm256_load_si256((__m256i *) &rank_weight[query_number][0]);
                    // __m256i score_vec = _mm256_maddubs_epi16(hidden_vec, weight_vec);
                    // score_vec = _mm256_madd_epi16(score_vec, _mm256_set1_epi16(1));
                    // // extract the 8 int32_t from score_vec for leaky relu
                    // positive_sum = _mm256_add_epi32(positive_sum, _mm256_max_epi32(score_vec, _mm256_set1_epi32(0)));
                    // negative_sum = _mm256_add_epi32(negative_sum, _mm256_min_epi32(score_vec, _mm256_set1_epi32(0)));
                    // __m256i context_weight_vec = _mm256_load_si256((__m256i *) &(context_select_weight[query_number][0]));
                    // __m256i context_score_vec = _mm256_maddubs_epi16(context_select_hidden_vec, context_weight_vec);
                    // context_score_vec = _mm256_madd_epi16(context_score_vec, _mm256_set1_epi16(1));
                    // context_positive_sum = _mm256_add_epi32(context_positive_sum, _mm256_max_epi32(context_score_vec, _mm256_set1_epi32(0)));
                    // We don't need to add the negative sum here as the context score is activated by relu
                    common_bits++;
                }else{
                    *unmatched_bits = query_number;
                    unmatched_bits++;
                }
            }
        } else {
            // The node bloom filter is compressed, we need to use the compressed version
            // It is an ordered vector of uint16_t, and the query numbers are also ordered.
            vector<uint16_t> & bloom_filter = *node->bloom_filter_compressed;
            int bloom_filter_len = bloom_filter.size();
            // Hence, we can use two pointers to iterate through the bloom filter and the query numbers
            int i = 0, j = 0;
            while (i < bloom_filter_len && j < query_length){
                uint16_t query_number = query_numbers[j];
                if (bloom_filter[i] == query_number){
                    // __m256i weight_vec = _mm256_load_si256((__m256i *) &rank_weight[query_number][0]);
                    // __m256i score_vec = _mm256_maddubs_epi16(hidden_vec, weight_vec);
                    // score_vec = _mm256_madd_epi16(score_vec, _mm256_set1_epi16(1));
                    // // extract the 8 int32_t from score_vec for leaky relu
                    // positive_sum = _mm256_add_epi32(positive_sum, _mm256_max_epi32(score_vec, _mm256_set1_epi32(0)));
                    // negative_sum = _mm256_add_epi32(negative_sum, _mm256_min_epi32(score_vec, _mm256_set1_epi32(0)));
                    // __m256i context_weight_vec = _mm256_load_si256((__m256i *) &(context_select_weight[query_number][0]));
                    // __m256i context_score_vec = _mm256_maddubs_epi16(context_select_hidden_vec, context_weight_vec);
                    // context_score_vec = _mm256_madd_epi16(context_score_vec, _mm256_set1_epi16(1));
                    // context_positive_sum = _mm256_add_epi32(context_positive_sum, _mm256_max_epi32(context_score_vec, _mm256_set1_epi32(0)));
                    // We don't need to add the negative sum here as the context score is activated by relu
                    common_bits++;
                    i++;
                    j++;
                }else if (query_number < bloom_filter[i]){
                    *unmatched_bits = query_number;
                    unmatched_bits++;
                    j++;
                }else{
                    i++;
                }
            }
        }
        // calculate and write the unmatched bits length
        unmatched_bits_length = query_length - common_bits;
        // leaky relu >> 4 -> negative slope = 1/16
        // negative_sum = _mm256_srai_epi32(negative_sum, 4);
        // positive_sum = _mm256_add_epi32(positive_sum, negative_sum);
        // alignas (64) int32_t buffer[8];
        // _mm256_store_si256((__m256i *) buffer, positive_sum);
        common_bits *= 127 * 64;
        intersection_score = common_bits; // + accumulate(buffer, buffer + 8, 0);
        // Context select. As it is activated by relu, we don't need to do bit shift
        // _mm256_store_si256((__m256i *) buffer, context_positive_sum);
        context_select_score = common_bits; // + accumulate(buffer, buffer + 8, 0);
    }

    inline void eval_intersect(
        const int query_length, 
        const uint16_t * query_numbers, 
        const int32_t * query_embedding, 
        const TreeNode * node,
        const int depth,
        int & intersection_score, 
        int & context_select_score,
        int8_t * context_rank_hidden,
        int & unmatched_bits_length,
        uint16_t * unmatched_bits,
        bool is_leaf
    ) {
        alignas (64) int8_t rank_hidden[32];
        alignas (64) int8_t context_select_hidden[32];
        int res_score = l2_avx2(query_embedding, node->embedding, depth, rank_hidden, context_select_hidden, context_rank_hidden);
        l3_avx2(node, query_length, query_numbers, depth, rank_hidden, context_select_hidden, intersection_score, context_select_score, unmatched_bits_length, unmatched_bits);
        intersection_score += res_score * 16;
    }

    inline void eval_context( 
        const TreeNode * node, 
        const int query_length, 
        const uint16_t * query_numbers, 
        const int8_t * context_rank_hidden,
        const int depth,
        int & context_score,
        bool is_leaf
    ) {
        // int8_t (&weight)[32768][32] = context_rank[depth].l2_weight;
        // // v18: This function calculates the context score only.
        // __m256i hidden_vec = _mm256_load_si256((__m256i *) context_rank_hidden);
        // __m256i positive_sum = _mm256_setzero_si256();
        // __m256i negative_sum = _mm256_setzero_si256();

        // // If the node bloom filter is not compressed, we directly use the bits
        // if (node->bloom_filter.bits != nullptr){
        //     uint32_t * bits = node->bloom_filter.bits;
        //     for(int i = 0; i < query_length; i++){
        //         uint16_t query_number = query_numbers[i];
        //         if (bits[query_number >> 5] &  (1 << (query_number & 31))) {
        //             __m256i weight_vec = _mm256_load_si256((__m256i *) &weight[query_number][0]);
        //             __m256i score_vec = _mm256_maddubs_epi16(hidden_vec, weight_vec);
        //             score_vec = _mm256_madd_epi16(score_vec, _mm256_set1_epi16(1));
        //             // extract the 8 int32_t from score_vec for leaky relu
        //             positive_sum = _mm256_add_epi32(positive_sum, _mm256_max_epi32(score_vec, _mm256_set1_epi32(0)));
        //             negative_sum = _mm256_add_epi32(negative_sum, _mm256_min_epi32(score_vec, _mm256_set1_epi32(0)));
        //         }
        //     }
        // } else {
        //     // The node bloom filter is compressed, we need to use the compressed version
        //     // It is an ordered vector of uint16_t, and the query numbers are also ordered.
        //     vector<uint16_t> & bloom_filter = *node->bloom_filter_compressed;
        //     int bloom_filter_len = bloom_filter.size();
        //     // Hence, we can use two pointers to iterate through the bloom filter and the query numbers
        //     int i = 0, j = 0;
        //     while (i < bloom_filter_len && j < query_length){
        //         uint16_t query_number = query_numbers[j];
        //         if (bloom_filter[i] == query_number){
        //              __m256i weight_vec = _mm256_load_si256((__m256i *) &weight[query_number][0]);
        //             __m256i score_vec = _mm256_maddubs_epi16(hidden_vec, weight_vec);
        //             score_vec = _mm256_madd_epi16(score_vec, _mm256_set1_epi16(1));
        //             // extract the 8 int32_t from score_vec for leaky relu
        //             positive_sum = _mm256_add_epi32(positive_sum, _mm256_max_epi32(score_vec, _mm256_set1_epi32(0)));
        //             negative_sum = _mm256_add_epi32(negative_sum, _mm256_min_epi32(score_vec, _mm256_set1_epi32(0)));
        //             i++;
        //             j++;
        //         }else if (query_number < bloom_filter[i]){
        //             j++;
        //         }else{
        //             i++;
        //         }
        //     }
        // }
        // leaky relu >> 4 -> negative slope = 1/16
        // negative_sum = _mm256_srai_epi32(negative_sum, 4);
        // positive_sum = _mm256_add_epi32(positive_sum, negative_sum);
        // alignas (64) int32_t buffer[8];
        // _mm256_store_si256((__m256i *) buffer, positive_sum);
        // context_score = accumulate(buffer, buffer + 8, 0);
        context_score = 0;
    }

    inline void distance_mix(int len, int * desc_sim, double * dist_sim, double * output, int depth){
        double a = this->a[depth];
        double b = this->b[depth];
        double c = this->c[depth];
        double d = this->d[depth];
        double desc_sim_mean, desc_sim_std;
        calc_mean_std(len, desc_sim, desc_sim_mean, desc_sim_std);
        for(int i=0; i < len; i++){
            output[i] = (c - sigmoid((a * (desc_sim[i] - desc_sim_mean)/desc_sim_std + b))) * (dist_sim[i] - d);
        }
    }

};

inline void heapify(double scores[], uint32_t indices[], int n, int i) {
    while (true) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && scores[left] > scores[largest])
            largest = left;

        if (right < n && scores[right] > scores[largest])
            largest = right;

        if (largest != i) {
            double tmp_score = scores[i];
            scores[i] = scores[largest];
            scores[largest] = tmp_score;
            uint32_t tmp_index = indices[i];
            indices[i] = indices[largest];
            indices[largest] = tmp_index;
            i = largest;
        } else {
            break;
        }
    }
}

inline void build_heap(double scores[], uint32_t indices[], int n) {
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(scores, indices, n, i);
}

inline uint32_t heappop(double scores[], uint32_t indices[], int& n) {
    // if (n <= 0) This should never happen
    //     return -1; // or throw an exception

    // Move the last element to root and reduce the size of heap
    uint32_t rootIndex = indices[0];
    n--;
    if (n > 0) {
        scores[0] = scores[n];
        indices[0] = indices[n];
        heapify(scores, indices, n, 0);
    }
    return rootIndex;
}

class sort_buffer {
public:
    uint32_t * node_idxs;
    uint32_t * next_node_idxs;
    double * scores;
    int * next_desc_sim;
    double * next_dist_sim;

    // v18: The following buffer is for context selection.
    // They work independently from the above buffers.
    int8_t * hiddens;
    int * unmatched_lengths;
    uint16_t * unmatched_numbers;
    int * context_select_scores;


    int buffer_size;
    int max_query_length;

    sort_buffer(int buffer_size, int max_query_length){
        this->buffer_size = buffer_size;
        this->max_query_length = max_query_length;
        node_idxs = new uint32_t[buffer_size];
        next_node_idxs = new uint32_t[buffer_size];
        scores = new double[buffer_size];
        next_desc_sim = new int[buffer_size];
        next_dist_sim = new double[buffer_size];

        hiddens = (int8_t *) aligned_alloc(64, sizeof(int8_t) * buffer_size * 32);
        unmatched_lengths = new int[buffer_size];
        unmatched_numbers = new uint16_t[buffer_size * max_query_length];
        context_select_scores = new int[buffer_size];
    }
    ~sort_buffer(){
        delete[] node_idxs;
        delete[] next_node_idxs;
        delete[] scores;
        delete[] next_desc_sim;
        delete[] next_dist_sim;

        free(hiddens);
        delete[] unmatched_lengths;
        delete[] unmatched_numbers;
        delete[] context_select_scores;
    }

    inline void step(){
        uint32_t * tmp_node_idxs = node_idxs;
        node_idxs = next_node_idxs;
        next_node_idxs = tmp_node_idxs;
    }
};

class Tree {
public:
    alignas (64) uint32_t * bf_bits; // For fast memory access
    alignas (64) int32_t * embeddings = nullptr;
    vector<TreeNode*> levels;
    vector<uint32_t> sizes;
    vector<uint32_t*> bf_headers; // The bf header of each level
    vector<int32_t*> emb_headers; // The embedding header of each level

    // initialize the sort helper from 0 to 19999
    Tree(Dataset &poi, vector<vector<vector<uint32_t>>> &levels, int keep_depth){
        // allocate memory for all the bloom filters
        int branch_rows = 0;
        this->sizes.push_back(poi.num_rows);
        for (int depth = 1; depth < keep_depth; ++depth) {
            branch_rows += levels[depth-1].size();
            sizes.push_back(levels[depth-1].size());
        }
        cout << "Allocating " << float(sizeof(uint32_t) * BF_SIZE * branch_rows ) / 1024 / 1024 << " MB for "<< branch_rows << " bloom filters on the tree..." << endl;
        this->bf_bits = (uint32_t *) aligned_alloc(64, sizeof(uint32_t) * branch_rows * BF_SIZE);
        memset(this->bf_bits, 0, sizeof(uint32_t) * branch_rows * BF_SIZE);

        // construct the tree
        // cout << "Constructing  " << float(sizeof(TreeNode) * poi.num_rows) / 1024 / 1024 << " MB for "<< poi.num_rows << " leaf nodes at depth 0..." << endl;
        TreeNode* leaf_nodes = (TreeNode *) aligned_alloc(64, sizeof(TreeNode) * poi.num_rows);
        memset(leaf_nodes, 0, sizeof(TreeNode) * poi.num_rows);

        for (uint32_t i = 0; i < poi.num_rows; ++i) {
            leaf_nodes[i].set_leaf_node(poi.bloom_filters[i], poi.locations[i], i);
        }
        this->levels.push_back(leaf_nodes);


        uint32_t * ptr = this->bf_bits;
        this->bf_headers.push_back(ptr);
        for (int depth = 1; depth < keep_depth; ++depth) {
            int num_clusters = levels[depth-1].size();
            this->bf_headers.push_back(ptr);
            // cout << "Allocating " << float(sizeof(TreeNode) * num_clusters) / 1024 / 1024 << " MB for "<< num_clusters << " branch nodes at depth " << depth << "..." << endl;
            TreeNode* branch_nodes = (TreeNode *) aligned_alloc(64, sizeof(TreeNode) * num_clusters);
            memset(branch_nodes, 0, sizeof(TreeNode) * num_clusters);
            for (uint32_t cluster = 0; cluster < levels[depth-1].size(); ++cluster) {
                branch_nodes[cluster].set_branch_node(ptr, cluster);
                for (uint32_t leaf_idx : levels[depth-1][cluster]) {
                    branch_nodes[cluster].add_child(leaf_nodes[leaf_idx]);
                }
                branch_nodes[cluster].compute_radius();
                ptr += BF_SIZE;
            }
            this->levels.push_back(branch_nodes);
            leaf_nodes = branch_nodes;
        }
        // finally, we reverse the order of the levels
        reverse(this->levels.begin(), this->levels.end());
        reverse(this->sizes.begin(), this->sizes.end());
        reverse(this->bf_headers.begin(), this->bf_headers.end());

        // DEBUG: Check if the tree nodes' bloom filters point to the correct location
        for (int depth = 0; depth < keep_depth - 1; ++depth) {
            for (int i = 0; i < this->sizes[depth]; ++i) {
                TreeNode * node = this->levels[depth] + i;
                if (node->bloom_filter.bits != this->bf_headers[depth] + node->node_idx * BF_SIZE) {
                    cerr << "Error constructing the tree. The bloom filter of node " << i << " at depth " << depth << " is not pointing to the correct location." << endl;
                    exit(1);
                }
            }
        }
    }

    ~Tree() {
        for (auto level : this->levels) {
            free(level);
        }
        if (this->embeddings != nullptr) {
            free(this->embeddings);
        }
        free(this->bf_bits);
    }

    void load_embeddings(const string &file_dir) {
        // unsupervised mode doesn't require embeddings.
        // We initialize a fake embedding for testing.
        int total_sizes = 0;
        for (auto i : sizes){
            total_sizes += i;
        }
        this->embeddings = (int32_t *) aligned_alloc(64, 32*sizeof(int32_t) * total_sizes);
        if (this->embeddings == nullptr) {
            cerr << "Memory allocation failed." << endl;
                        exit(1);
        }

        int num_loads = 0;
        this->emb_headers.resize(this->sizes.size());
        int32_t *ptr = this->embeddings;
        for (int i = 0; i < this->sizes.size(); ++i) {
            int level_size = this->sizes[i];
            this->emb_headers[i] = ptr;
            for (int j = 0; j < level_size; ++j) {
                this->levels[i][j].embedding = ptr;
                ptr += 32;
                num_loads++;
            }
        }
    }

    void prune_bits(){
        // This function is to prune the common bits at each level (except the leaf level)
        // This works for all dataset, and is especially effective for GeoGLUE dataset as it is heavily anonymized.
        vector<uint16_t> common_bit_idxs;
        uint32_t * bits = new uint32_t[BF_SIZE];
        // set the common bits to all zeros;
        memset(bits, 0, BF_SIZE * sizeof(uint32_t));
        BloomFilter bf(bits);
        for (int i = 0; i < NUM_SLICES * NUM_BITS; i++){
            bool common = true;
            for(int j = 0; j < sizes[0]; j++){
                if(!(levels[0]+j)->bloom_filter.test(i)){
                    common = false;
                    break;
                }
            }
            if(common){
                bf.set(i);
                common_bit_idxs.push_back(i);
            }
        }
        // cout << "Common bits at level 0: " << common_bit_idxs.size() << endl;
        // prune the common bits at each level
        for (int j = 0; j < sizes[0]; j++){
            (levels[0]+j)->bloom_filter ^= bf;
        }

        vector<uint16_t> next_common_bit_idxs;
        for(int depth=1; depth < sizes.size()-1; depth++){
            memset(bits, 0, BF_SIZE * sizeof(uint32_t));
            for(uint16_t idx: common_bit_idxs){
                bool common = true;
                for(int j = 0; j < sizes[depth]; j++){
                    if(!(levels[depth]+j)->bloom_filter.test(idx)){
                        common = false;
                        break;
                    }
                }
                if(common){
                    bf.set(idx);
                    next_common_bit_idxs.push_back(idx);
                }
            }
            // cout << "Common bits at level " << depth << ": " << next_common_bit_idxs.size() << endl;
            common_bit_idxs = next_common_bit_idxs;
            next_common_bit_idxs.clear();
            for (int j = 0; j < sizes[depth]; j++){
                (levels[depth]+j)->bloom_filter ^= bf;
            }
        }
        delete[] bits;
    }
    
    vector<uint32_t> get_path(uint32_t leaf_idx){
        // This function is to get the path from the leaf node to the root node
        vector<uint32_t> path;
        TreeNode * node = &levels[levels.size()-1][leaf_idx];
        while(node->parent != nullptr){
            path.push_back(node->node_idx);
            node = node->parent;
        }
        reverse(path.begin(), path.end());
        return path;
    }

    vector<uint32_t> beam_search_nnue(NNUE & nnue, vector<uint16_t> query_bloom_filter, double lat, double lon, vector<uint32_t> & beam_width, sort_buffer & buffer, int topk){
        // This function is to perform efficient beam search with NNUE
        // We first sort the query bloom filter.
        sort(query_bloom_filter.begin(), query_bloom_filter.end());
        int query_length = query_bloom_filter.size();
        uint16_t * query_numbers = query_bloom_filter.data();
        // prepare the query embedding
        alignas (64) int32_t * query_embedding = (int32_t *) aligned_alloc(64, sizeof(int32_t) * 32 * nnue.depth);
        nnue.encode_query_avx2(query_bloom_filter, query_embedding);
        uint32_t write_ptr = 0;
        // We run the evaluation loop twice.
        // The first loop: evaluate the desc_sim, dist_sim, and the context selection scores
        for(int i=0; i < sizes[0]; i++){
            TreeNode * child = &levels[0][i];
            nnue.eval_intersect(
                query_length, 
                query_numbers, 
                query_embedding + 32 * 0, 
                child, 
                0,
                buffer.next_desc_sim[write_ptr],
                buffer.context_select_scores[write_ptr],
                buffer.hiddens + write_ptr * 32,
                buffer.unmatched_lengths[write_ptr],
                buffer.unmatched_numbers + write_ptr * buffer.max_query_length,
                false
            );
            double lat_diff = lat - child->location[0];
            double lon_diff = lon - child->location[1];
            double dist = max(0.0, sqrt(lat_diff * lat_diff + lon_diff * lon_diff) - child->radius);
            buffer.next_dist_sim[write_ptr] = -log(dist + 1);
            buffer.node_idxs[write_ptr] = i;
            write_ptr++;
        }
        // The second loop: context selection
        // Sort the context node indices by the context selection scores via std::sort
        #ifdef USE_CONTEXT
        int top_context = write_ptr > MAX_TOP_CONTEXT? MAX_TOP_CONTEXT: write_ptr;
        vector<int> in_beam_idxs(write_ptr);
        iota(in_beam_idxs.begin(), in_beam_idxs.end(), 0);
        partial_sort(in_beam_idxs.begin(), in_beam_idxs.begin() + top_context, in_beam_idxs.end(), [&](int a, int b) {
            return buffer.context_select_scores[a] > buffer.context_select_scores[b];
        });
        // nested loop to find context for each node
        // if founded, evaluate the context score
        for (int i = 0; i < write_ptr; ++i) {
            TreeNode * node = &levels[0][buffer.node_idxs[i]];
            int unmatched_length = buffer.unmatched_lengths[i];
            if (unmatched_length == 0) {
                continue;
            }
            for (int j = 0; j < top_context; ++j) {
                int in_beam_idx = in_beam_idxs[j];
                TreeNode * context_node = &levels[0][buffer.node_idxs[in_beam_idx]];
                // skip the node itself as it won't provide more bits as context
                if (context_node->node_idx == node->node_idx) {
                    continue;
                }
                // skip nodes that are 1000 meters away
                double lat_diff = node->location[0] - context_node->location[0];
                double lon_diff = node->location[1] - context_node->location[1];
                double dist_sq = lat_diff * lat_diff + lon_diff * lon_diff;
                if (dist_sq > 1000 * 1000) {
                    continue;
                }
                // evaluate the context score
                int context_score;
                nnue.eval_context(
                    context_node,
                    unmatched_length,
                    buffer.unmatched_numbers + i * buffer.max_query_length,
                    buffer.hiddens + i * 32,
                    0,
                    context_score,
                    false
                );
                buffer.next_desc_sim[i] += context_score;
                break;
            }
        }
        #endif
        nnue.distance_mix(write_ptr, buffer.next_desc_sim, buffer.next_dist_sim, buffer.scores, 0);
        build_heap(buffer.scores, buffer.node_idxs, write_ptr);
        int candidate_size = min(beam_width[0], write_ptr);
        // Here comes the most time consuming part
        for(int depth=1; depth < levels.size(); depth++){
            int next_candidate_size = min(beam_width[depth], sizes[depth]);
            write_ptr = 0;
            bool is_leaf = depth == levels.size() - 1;
            // Similar to the process above.
            while(candidate_size > 0){
                uint32_t node_idx = heappop(buffer.scores, buffer.node_idxs, candidate_size);
                TreeNode * node = &this->levels[depth-1][node_idx];
                for(TreeNode* child: *(node->children)){
                    nnue.eval_intersect(
                        query_length, 
                        query_numbers, 
                        query_embedding + 32 * depth,
                        child,
                        depth, 
                        buffer.next_desc_sim[write_ptr],
                        buffer.context_select_scores[write_ptr],
                        buffer.hiddens + write_ptr * 32,
                        buffer.unmatched_lengths[write_ptr],
                        buffer.unmatched_numbers + write_ptr * buffer.max_query_length,
                        is_leaf
                    );
                    double lat_diff = lat - child->location[0];
                    double lon_diff = lon - child->location[1];
                    double dist = max(0.0, sqrt(lat_diff * lat_diff + lon_diff * lon_diff) - child->radius) + 1;
                    buffer.next_dist_sim[write_ptr] = -log(dist);
                    buffer.next_node_idxs[write_ptr] = child->node_idx;
                    write_ptr++;
                    if (write_ptr >= next_candidate_size){
                        break;
                    }
                }
                if (write_ptr >= next_candidate_size){
                    break;
                }
            }
            #ifdef USE_CONTEXT
            int top_context = write_ptr > MAX_TOP_CONTEXT? MAX_TOP_CONTEXT: write_ptr;
            in_beam_idxs.resize(write_ptr);
            iota(in_beam_idxs.begin(), in_beam_idxs.end(), 0);
            partial_sort(in_beam_idxs.begin(), in_beam_idxs.begin() + top_context, in_beam_idxs.end(), [&](int a, int b) {
                return buffer.context_select_scores[a] > buffer.context_select_scores[b];
            });
            // nested loop to find context for each node
            // if founded, evaluate the context score
            for (int i = 0; i < write_ptr; ++i) {
                TreeNode * node = &levels[depth][buffer.next_node_idxs[i]];
                int unmatched_length = buffer.unmatched_lengths[i];
                if (unmatched_length == 0) {
                    continue;
                }
                for (int j = 0; j < top_context; ++j) {
                    int in_beam_idx = in_beam_idxs[j];
                    TreeNode * context_node = &levels[depth][buffer.next_node_idxs[in_beam_idx]];
                    if (context_node->node_idx == node->node_idx) {
                        continue;
                    }
                    double lat_diff = node->location[0] - context_node->location[0];
                    double lon_diff = node->location[1] - context_node->location[1];
                    double dist_sq = lat_diff * lat_diff + lon_diff * lon_diff;
                    if (dist_sq > 1000 * 1000) {
                        continue;
                    }
                    int context_score;
                    nnue.eval_context(
                        context_node, 
                        unmatched_length,
                        buffer.unmatched_numbers + i * buffer.max_query_length,
                        buffer.hiddens + i * 32,
                        depth,
                        context_score,
                        is_leaf
                    );
                    buffer.next_desc_sim[i] += context_score;
                    break;
                }
            }
            #endif
            nnue.distance_mix(write_ptr, buffer.next_desc_sim, buffer.next_dist_sim, buffer.scores, depth);
            build_heap(buffer.scores, buffer.next_node_idxs, write_ptr);
            buffer.step();
            candidate_size = write_ptr;
        }
        topk = min(topk, candidate_size);
        vector<uint32_t> results(topk);
        for(int i=0; i < topk; i++){
            results[i] = heappop(buffer.scores, buffer.node_idxs, candidate_size);
        }
        return results;
    }

    vector<vector<uint32_t>> beam_search_nnue_intermediates(NNUE & nnue, vector<uint16_t> query_bloom_filter, double lat, double lon, vector<uint32_t> & beam_width, sort_buffer & buffer, int topk){
        // This function will returns the intermediate node indices for 2nd, 3rd, ... level
        // together with a top-k ranking of the leaf nodes.
        // It is used to assist the PyTorch training and retrieval, reranking analysis.
        // We first sort the query bloom filter.
        sort(query_bloom_filter.begin(), query_bloom_filter.end());
        int query_length = query_bloom_filter.size();
        uint16_t * query_numbers = query_bloom_filter.data();
        // Pre-compute the query embedding
        alignas (64) int32_t * query_embedding = (int32_t *) aligned_alloc(64, sizeof(int32_t) * 32 * nnue.depth);
        nnue.encode_query_avx2(query_bloom_filter, query_embedding);
        uint32_t write_ptr = 0;
        // We run the evaluation loop twice.
        // The first loop: evaluate the desc_sim, dist_sim, and the context selection scores
        for(int i=0; i < sizes[0]; i++){
            TreeNode * child = &levels[0][i];
            nnue.eval_intersect(
                query_length, 
                query_numbers, 
                query_embedding + 32 * 0,
                child, 
                0,
                buffer.next_desc_sim[write_ptr],
                buffer.context_select_scores[write_ptr],
                buffer.hiddens + write_ptr * 32,
                buffer.unmatched_lengths[write_ptr],
                buffer.unmatched_numbers + write_ptr * buffer.max_query_length,
                false
            );
            double lat_diff = lat - child->location[0];
            double lon_diff = lon - child->location[1];
            double dist = max(0.0, sqrt(lat_diff * lat_diff + lon_diff * lon_diff) - child->radius)+ 1;
            buffer.next_dist_sim[write_ptr] = -log(dist);
            buffer.node_idxs[write_ptr] = i;
            write_ptr++;
        }
        // The second loop: context selection
        // Sort the context node indices by the context selection scores via std::sort
        #ifdef USE_CONTEXT
        int top_context = write_ptr > MAX_TOP_CONTEXT? MAX_TOP_CONTEXT: write_ptr;
        vector<int> in_beam_idxs(write_ptr);
        iota(in_beam_idxs.begin(), in_beam_idxs.end(), 0);
        partial_sort(in_beam_idxs.begin(), in_beam_idxs.begin() + top_context, in_beam_idxs.end(), [&](int a, int b) {
            return buffer.context_select_scores[a] > buffer.context_select_scores[b];
        });
        // nested loop to find context for each node
        // if founded, evaluate the context score
        for (int i = 0; i < write_ptr; ++i) {
            TreeNode * node = &levels[0][buffer.node_idxs[i]];
            int unmatched_length = buffer.unmatched_lengths[i];
            if (unmatched_length == 0) {
                continue;
            }
            for (int j = 0; j < top_context; ++j) {
                int in_beam_idx = in_beam_idxs[j];
                TreeNode * context_node = &levels[0][buffer.node_idxs[in_beam_idx]];
                // skip the node itself
                if (context_node->node_idx == node->node_idx) {
                    continue;
                }
                // skip nodes that are 1000 meters away
                double lat_diff = node->location[0] - context_node->location[0];
                double lon_diff = node->location[1] - context_node->location[1];
                double dist_sq = lat_diff * lat_diff + lon_diff * lon_diff;
                if (dist_sq > 1000 * 1000) {
                    continue;
                }
                // evaluate the context score
                int context_score;
                nnue.eval_context(
                    context_node, 
                    unmatched_length,
                    buffer.unmatched_numbers + i * buffer.max_query_length,
                    buffer.hiddens + i * 32,
                    0,
                    context_score,
                    false
                );
                buffer.next_desc_sim[i] += context_score;
                break;
            }
        }
        #endif
        nnue.distance_mix(write_ptr, buffer.next_desc_sim, buffer.next_dist_sim, buffer.scores, 0);
        // vector<double> score_debug(write_ptr);
        // memcpy(score_debug.data(), buffer.scores, write_ptr * sizeof(double));
        build_heap(buffer.scores, buffer.node_idxs, write_ptr);
        int candidate_size = min(beam_width[0], write_ptr);
        // Prepare the result list
        vector<vector<uint32_t>> results(levels.size());
        for(int depth=1; depth < levels.size(); depth++){
            int next_candidate_size = min(beam_width[depth], sizes[depth]);
            write_ptr = 0;
            bool is_leaf = depth == levels.size() - 1;
            while(candidate_size > 0){
                uint32_t node_idx = heappop(buffer.scores, buffer.node_idxs, candidate_size);
                TreeNode * node = &this->levels[depth-1][node_idx];
                for(TreeNode* child: *(node->children)){
                    nnue.eval_intersect(
                        query_length, 
                        query_numbers, 
                        query_embedding + 32 * depth,
                        child,
                        depth,
                        buffer.next_desc_sim[write_ptr],
                        buffer.context_select_scores[write_ptr],
                        buffer.hiddens + write_ptr * 32,
                        buffer.unmatched_lengths[write_ptr],
                        buffer.unmatched_numbers + write_ptr * buffer.max_query_length,
                        is_leaf
                    );
                    double lat_diff = lat - child->location[0];
                    double lon_diff = lon - child->location[1];
                    double dist = max(0.0, sqrt(lat_diff * lat_diff + lon_diff * lon_diff) - child->radius) + 1;
                    buffer.next_dist_sim[write_ptr] = -log(dist);
                    buffer.next_node_idxs[write_ptr] = child->node_idx;
                    write_ptr++;
                    if (write_ptr >= next_candidate_size){
                        break;
                    }
                }
                if (write_ptr >= next_candidate_size){
                    break;
                }
            }
            #ifdef USE_CONTEXT
            int top_context = write_ptr > MAX_TOP_CONTEXT? MAX_TOP_CONTEXT: write_ptr;
            in_beam_idxs.resize(write_ptr);
            iota(in_beam_idxs.begin(), in_beam_idxs.end(), 0);
            partial_sort(in_beam_idxs.begin(), in_beam_idxs.begin() + top_context, in_beam_idxs.end(), [&](int a, int b) {
                return buffer.context_select_scores[a] > buffer.context_select_scores[b];
            });
            // nested loop to find context for each node
            // if founded, evaluate the context score
            for (int i = 0; i < write_ptr; ++i) {
                TreeNode * node = &levels[depth][buffer.next_node_idxs[i]];
                int unmatched_length = buffer.unmatched_lengths[i];
                if (unmatched_length == 0) {
                    continue;
                }
                int original_score = buffer.next_desc_sim[i];
                for (int j = 0; j < top_context; ++j) {
                    int in_beam_idx = in_beam_idxs[j];
                    int context_select_score = buffer.context_select_scores[in_beam_idx];
                    TreeNode * context_node = &levels[depth][buffer.next_node_idxs[in_beam_idx]];
                    if (context_node->node_idx == node->node_idx) {
                        continue;
                    }
                    double lat_diff = node->location[0] - context_node->location[0];
                    double lon_diff = node->location[1] - context_node->location[1];
                    double dist_sq = lat_diff * lat_diff + lon_diff * lon_diff;
                    if (dist_sq > 1000 * 1000) {
                        continue;
                    }
                    int context_score;
                    nnue.eval_context(
                        context_node, 
                        unmatched_length,
                        buffer.unmatched_numbers + i * buffer.max_query_length,
                        buffer.hiddens + i * 32,
                        depth,
                        context_score,
                        is_leaf
                    );
                    buffer.next_desc_sim[i] += context_score;
                    break;
                }
            }
            #endif
            results[depth-1].resize(write_ptr);
            memcpy(results[depth-1].data(), buffer.next_node_idxs, write_ptr * sizeof(uint32_t));
            nnue.distance_mix(write_ptr, buffer.next_desc_sim, buffer.next_dist_sim, buffer.scores, depth);
            build_heap(buffer.scores, buffer.next_node_idxs, write_ptr);
            buffer.step();
            candidate_size = write_ptr;
        }
        topk = min(topk, candidate_size);
        for (int i = 0; i < topk; ++i) {
            results[levels.size()-1].push_back(heappop(buffer.scores, buffer.node_idxs, candidate_size));
        }
        return results;
    }
};
    