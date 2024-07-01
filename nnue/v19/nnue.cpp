// -------------------------------------------------
// GeoBloom search engine
// Corresponding to python geobloom_v{VERSION}.py
// Will do modularized design in the future
// -------------------------------------------------

#include "nnue_avx2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <ctime>
#include <algorithm>
#include <thread>
#include <atomic>

using namespace std;

const uint32_t VERSION = 19;

bool load_tree(const string& file_dir, vector<vector<vector<uint32_t>>> &levels) {
    ifstream file(file_dir, ios::binary);

    if (!file) {
        cerr << "Cannot open file: " << file_dir << endl;
        return false; // Return an empty structure if file opening fails
    }

    uint32_t num_levels;
    file.read(reinterpret_cast<char*>(&num_levels), sizeof(num_levels));

    for (uint32_t i = 0; i < num_levels; ++i) {
        uint32_t num_clusters;
        file.read(reinterpret_cast<char*>(&num_clusters), sizeof(num_clusters));
        vector<vector<uint32_t>> clusters(num_clusters);

        for (uint32_t j = 0; j < num_clusters; ++j) {
            uint32_t num_nodes;
            file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
            clusters[j].resize(num_nodes);

            for (uint32_t k = 0; k < num_nodes; ++k) {
                file.read(reinterpret_cast<char*>(&clusters[j][k]), sizeof(uint32_t));
            }
        }

        levels.push_back(move(clusters));
    }
    return true;
}

float ndcg(vector<uint32_t>& truth, vector<uint32_t>& prediction, size_t k=5) {
    float dcg = 0;
    for (uint32_t i = 0; i < min(prediction.size(), k); ++i) {
        if (find(truth.begin(), truth.end(), prediction[i]) != truth.end()) {
            dcg += 1 / log2(i + 2);
        }
    }
    float idcg = 0;
    for (uint32_t i = 0; i < min(truth.size(), k); ++i) {
        idcg += 1 / log2(i + 2);
    }
    return dcg / idcg;
}

float recall(vector<uint32_t>& truth, vector<uint32_t>& prediction, size_t k) {
    float num_correct = 0;
    for (uint32_t i = 0; i < min(prediction.size(), k); ++i) {
        if (find(truth.begin(), truth.end(), prediction[i]) != truth.end()) {
            num_correct += 1;
        }
    }
    return num_correct / min(truth.size(), k);
}

float recall_intermediate(set<uint32_t>& truth, vector<uint32_t>& prediction) {
    float num_correct = 0;
    for (uint32_t prediction_id: prediction) {
        if (truth.find(prediction_id) != truth.end()) {
            num_correct += 1;
        }
    }
    return num_correct / truth.size();
}

void evaluate(Dataset & dataset, vector<uint32_t> * predictions){

    float ndcg1 = 0, ndcg5 = 0, recall10=0, recall20=0;
    for (int i = 0; i < dataset.num_rows; ++i) {
        if (predictions[i].size() == 0) {
            continue;
        }
        ndcg1 += ndcg(dataset.truths[i], predictions[i], 1);
        ndcg5 += ndcg(dataset.truths[i], predictions[i], 5);
        recall10 += recall(dataset.truths[i], predictions[i], 10);
        recall20 += recall(dataset.truths[i], predictions[i], 20);
    }
    ndcg1 /= dataset.num_rows;
    ndcg5 /= dataset.num_rows;
    recall10 /= dataset.num_rows;
    recall20 /= dataset.num_rows;
    cout.precision(6);
    // Fixed to 6 decimal places
    cout << "====================== Evaluation =======================" << endl;
    cout << "Recall@20 \t Recall@10  \t NDCG@5  \t NDCG@1" << endl;
    cout << fixed << recall20 << "\t" << recall10 << "\t" << ndcg5 << "\t" << ndcg1 << endl;
    cout << "=========================================================" << endl;

}

void thread_search(NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, vector<uint32_t> * predictions, int start, int end, int buffer_size, atomic<int>& counter) {
    sort_buffer buffer(buffer_size, 1024);
    for (int i = start; i < end; ++i) {
        vector<uint32_t> result = tree->beam_search_nnue(*nnue, dataset.bloom_filters[i], dataset.locations[i].first, dataset.locations[i].second, beam_width, buffer, 200);
        predictions[i] = result;
        int processed = counter.fetch_add(1) + 1;
        if (processed % 1000 == 0) {
            cout << "Searching " << processed << "th query..." << endl;
        }
    }
}


void single_thread_search(NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, vector<uint32_t> * predictions) {
    int buffer_size = tree->sizes[0];
    for (int i=1; i < tree->sizes.size(); i++){
        int current_buffer_size = min(beam_width[i], tree->sizes[i]);
        buffer_size = max(buffer_size, current_buffer_size);
    }
    atomic<int> counter(0);
    thread_search(nnue, tree, dataset, beam_width, predictions, 0, dataset.num_rows, buffer_size, counter);
}


void multi_thread_search(int num_threads, NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, vector<uint32_t> * predictions) {
    int buffer_size = tree->sizes[0];
    for (int i = 1; i < tree->sizes.size(); i++) {
        int current_buffer_size = min(beam_width[i], tree->sizes[i]);
        buffer_size = max(buffer_size, current_buffer_size);
    }
    vector<thread> threads(num_threads);
    int chunk_size = dataset.num_rows / num_threads;
    // Start threads
    atomic<int> counter(0);
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i != num_threads - 1) ? (start + chunk_size) : dataset.num_rows;
        threads[i] = thread(thread_search, nnue, tree, ref(dataset), ref(beam_width), predictions, start, end, buffer_size, ref(counter));
    }
    // Wait for all threads to finish
    for (auto &t : threads) {
        t.join();
    }
}

void single_thread_speed_test(NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width) {
    vector<uint32_t> * predictions = new vector<uint32_t>[dataset.num_rows];
    auto start_search = clock();
    single_thread_search(nnue, tree, dataset, beam_width, predictions);
    auto end_search = clock();

    cout << "Search time: " << (double)(end_search - start_search) / CLOCKS_PER_SEC << "s" << endl;
    cout << "Query Per Second: " << dataset.num_rows / ((double)(end_search - start_search) / CLOCKS_PER_SEC) << endl;
    // Compute the ndcg score
    evaluate(dataset, predictions);
    delete[] predictions;
}

void multi_thread_speed_test(int num_threads, NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, const string & output_path = "") {
    vector<uint32_t> * predictions = new vector<uint32_t>[dataset.num_rows];
    auto start_search = clock();
    multi_thread_search(num_threads, nnue, tree, dataset, beam_width, predictions);
    auto end_search = clock();

    cout << "Total search time of all threads: " << (double)(end_search - start_search) / CLOCKS_PER_SEC << "s" << endl;
    cout << "Query Per Second: " << dataset.num_rows / ((double)(end_search - start_search) / CLOCKS_PER_SEC) << endl;
    // Compute the ndcg score
    evaluate(dataset, predictions);
    if(output_path != ""){
        // Write the predictions to file. Just write everything in binary in a row
        ofstream file(output_path, ios::binary);
        if (!file) {
            cerr << "Cannot open file: " << output_path << endl;
            return;
        }
        for (int i = 0; i < dataset.num_rows; i++){
            file.write(reinterpret_cast<char*>(&predictions[i][0]), predictions[i].size() * sizeof(uint32_t));
        }
        file.close();
        cout << "Predictions saved to " << output_path << endl;
    }
    delete[] predictions;
}

void thread_search_intermediates(NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, vector<vector<uint32_t>> * predictions, int start, int end, int buffer_size, atomic<int>& counter) {
    sort_buffer buffer(buffer_size, 1024);
    for (int i = start; i < end; ++i) {
        vector<vector<uint32_t>> result = tree->beam_search_nnue_intermediates(*nnue, dataset.bloom_filters[i], dataset.locations[i].first, dataset.locations[i].second, beam_width, buffer, 20);
        predictions[i] = result;

        int processed = counter.fetch_add(1) + 1;
        if (processed % 10000 == 0) {
            cout << "Searching " << processed << "th query..." << endl;
        }
    }
}


void search_intermediates(int num_threads, NNUE * nnue, Tree* tree, Dataset & dataset, vector<uint32_t> & beam_width, const string & output_path) {
    vector<vector<uint32_t>> * predictions = new vector<vector<uint32_t>>[dataset.num_rows];
    int buffer_size = tree->sizes[0];
    for (int i = 1; i < tree->sizes.size(); i++) {
        int current_buffer_size = min(beam_width[i], tree->sizes[i]);
        buffer_size = max(buffer_size, current_buffer_size);
    }
    vector<thread> threads(num_threads);
    int chunk_size = dataset.num_rows / num_threads;
    // Start threads
    atomic<int> counter(0);
    auto start_search = clock();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i != num_threads - 1) ? (start + chunk_size) : dataset.num_rows;
        threads[i] = thread(thread_search_intermediates, nnue, tree, ref(dataset), ref(beam_width), predictions, start, end, buffer_size, ref(counter));
    }
    // Wait for all threads to finish
    for (auto &t : threads) {
        t.join();
    }
    auto end_search = clock();

    cout << "Total search time of all threads: " << (double)(end_search - start_search) / CLOCKS_PER_SEC
         << "s, Query Per Second: " << dataset.num_rows / ((double)(end_search - start_search) / CLOCKS_PER_SEC) << endl;

    float * recall_scores = new float[tree->sizes.size() - 1];
    memset(recall_scores, 0, sizeof(float) * (tree->sizes.size() - 1));
    // Compute the recall score
    for (int i = 0; i < dataset.num_rows; ++i) {
        if (predictions[i].size() == 0) {
            continue;
        }
        vector<set<uint32_t>> truth_by_depth;
        for (uint32_t truth: dataset.truths[i]){
            vector<uint32_t> truth_path = tree->get_path(truth);
            for (int depth = 0; depth < truth_path.size(); depth++){
                if (truth_by_depth.size() <= depth){
                    truth_by_depth.push_back(set<uint32_t>());
                }
                truth_by_depth[depth].insert(truth_path[depth]);
            }
        }
        for (int depth = 0; depth < predictions[i].size()-1; depth++){
            recall_scores[depth] += recall_intermediate(truth_by_depth[depth], predictions[i][depth]);
        }
    }
    for (int depth = 1; depth < tree->sizes.size(); depth++){
        recall_scores[depth - 1] /= dataset.num_rows;
    }

    float ndcg_score = 0;
    for (int i = 0; i < dataset.num_rows; ++i) {
        if (predictions[i].size() == 0) {
            continue;
        }
        ndcg_score += ndcg(dataset.truths[i], predictions[i][predictions[i].size() - 1]);
    }
    ndcg_score /= dataset.num_rows;

    // Print statistics
    cout << "=============== Intermediate Recall Scores =============="<< endl;
    cout.precision(6);
    for (int depth = 1; depth < tree->sizes.size(); depth++){
        cout << fixed << recall_scores[depth - 1] << "\t";
    }
    cout << endl;
    vector<uint32_t> * top_predictions = new vector<uint32_t>[dataset.num_rows];
    for (int i = 0; i < dataset.num_rows; ++i) {
        if (predictions[i].size() == 0) {
            continue;
        }
        top_predictions[i] = predictions[i][predictions[i].size() - 1];
    }
    evaluate(dataset, top_predictions);
    // Write the predictions to file. Just write everything in binary in a row
    ofstream file(output_path, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << output_path << endl;
        return;
    }
    for (int i = 0; i < dataset.num_rows; i++){
        for (int depth = 0; depth < predictions[i].size(); depth++){
            file.write(reinterpret_cast<char*>(&predictions[i][depth][0]), predictions[i][depth].size() * sizeof(uint32_t));
        }
    }
    file.close();
    cout << "Predictions saved to " << output_path << endl;
    delete[] predictions;
    delete[] top_predictions;
}


// Use argv to pass the dataset name
int main(int argc, char *argv[]) {
    // Get the dataset name
    string dataset_name;
    string task;
    bool has_portion = false;
    string portion;
    int num_threads = 16;
    vector<uint32_t> beam_width;
    string transfer_path;
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <dataset_name> <task> <num_threads> <beamwidth1-beamwidth2-...> <transfer_path>" << endl;
        dataset_name = "MeituanBeijing";
        task = "test";
        beam_width = {200, 300, 300, 300};
        transfer_path = "data_bin/" + dataset_name + "/";
    }
    else {
        dataset_name = argv[1];
        task = argv[2];
        if (argc >= 4) {
            num_threads = stoi(argv[3]);
            // Parse the beam width
            // If no beam width is specified, use the default beam width
            if (argc >= 5) {
                stringstream ss(argv[4]);
                uint32_t width;
                while (ss >> width) {
                    beam_width.push_back(width);
                    if (ss.peek() == '-') {
                        ss.ignore();
                    }
                }
                printf("Beam width: ");
                for (uint32_t width: beam_width){
                    printf("%d ", width);
                }
            } else {
                if(dataset_name == "GeoGLUE"){
                    beam_width = {6000, 6000, 4000, 1000};
                }else if (dataset_name == "MeituanBeijing"){
                    beam_width = {200, 300, 300, 300};
                }else if (dataset_name == "MeituanShanghai"){
                    beam_width = {200, 300, 300, 300};
                }
            }
            // Parse the output path here. Task 1 does not need output path; Task 2 and 3 need output path
            if (argc >= 6) {
                transfer_path = argv[5];
                if (transfer_path.back() != '/') {
                    transfer_path += "/";
                }
                if (argc == 7){
                    has_portion = true;
                    portion = argv[6];
                }
            }else{
                transfer_path = "data_bin/" + dataset_name + "/";
            }
        }
    }

    // Load the dataset. We assume the dataset is already serialized in the data_bin folder
    string dataset_path = "data_bin/" + dataset_name + "/";
    cout << "Loading dataset from path: " << dataset_path << endl;
    // The dataset won't change once serialized.
    string train_path = dataset_path + "train.bin";
    if (has_portion){
        train_path = dataset_path + "portion/" + "train_" + portion + ".bin";
    } else {
        train_path = dataset_path + "train.bin";
    }
    string dev_path = dataset_path + "dev.bin";
    string test_path = dataset_path + "test.bin";
    string poi_path = dataset_path + "poi.bin";
    string tree_path = dataset_path + "tree.bin";

    // The model and the node representations will change with training.
    string node_path = transfer_path + "node_v" + to_string(VERSION) + ".bin";
    string nnue_path = transfer_path + "nnue_v" + to_string(VERSION) + ".bin";

    // Load the dataset
    Dataset poi, train, dev, test;
    poi.load(poi_path);

    // For all tasks, load the tree and nnue
    vector<vector<vector<uint32_t>>> levels;
    load_tree(tree_path, levels);
    
    NNUE * nnue = (NNUE *) aligned_alloc(64, sizeof(NNUE));
    nnue->load(nnue_path);
    // Construct the tree
    Tree * tree = new Tree(poi, levels, nnue->depth);
    tree->load_embeddings(node_path);
    tree->prune_bits();
    // Task 1: Speed test
    if (task == "speed"){
        test.load(test_path);
        cout << "Testing speed on " << test.num_rows << " queries..." << endl;
        multi_thread_speed_test(num_threads, nnue, tree, test, beam_width);
    }else if(task == "pydev"){
        dev.load(dev_path);
        cout << "Preparing dev candidates..." << endl;
        search_intermediates(num_threads, nnue, tree, dev, beam_width, transfer_path + "dev_nodes.bin");
    }else if(task == "pytrain"){
        train.load(train_path);
        cout << "Preparing train candidates..." << endl;
        search_intermediates(num_threads, nnue, tree, train, beam_width, transfer_path + "train_nodes.bin");
    }else if(task == "test"){
        test.load(test_path);
        cout << "Infering test candidates..." << endl;
        search_intermediates(num_threads, nnue, tree, test, beam_width, transfer_path + "test_nodes.bin");
    }else if (task == "all"){
        train.load(train_path);
        dev.load(dev_path);
        test.load(test_path);
        cout << "Searching train candidates..." << endl;
        multi_thread_speed_test(num_threads, nnue, tree, train, beam_width, dataset_path + "train_top200.bin");
        cout << "Searching dev candidates..." << endl;
        multi_thread_speed_test(num_threads, nnue, tree, dev, beam_width, dataset_path + "dev_top200.bin");
        cout << "Searching test candidates..." << endl;
        multi_thread_speed_test(num_threads, nnue, tree, test, beam_width, dataset_path + "test_top200.bin");
    }else if (task == "pause"){
        // pause the program so that you can measure its memory usage via htop.
        printf("Paused. Press any button to exit.");
        char c = getchar();
    }
    
    delete tree;
    free(nnue);
    return 0;
}