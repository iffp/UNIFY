// search_wrapper.cpp - UNIFY query execution wrapper for FANNS benchmarking
// This wrapper performs range-filtered ANN queries using UNIFY HSIG index

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <set>
#include <chrono>
#include <thread>
#include <atomic>
#include <omp.h>

#include "../hannlib/api.h"
#include "../include/fanns_survey_helpers.cpp"

// Global atomic for peak thread count
std::atomic<int> peak_threads(0);

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    // Restrict to single thread for query execution
    omp_set_num_threads(1);

	// Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

	// Parameters
    if (argc != 13) {
        cerr << "Usage: " << argv[0] << " --query_path <query.fvecs> "
             << "--query_ranges_file <query_ranges.csv> "
             << "--groundtruth_file <groundtruth.ivecs> "
             << "--index_file <index_path> "
             << "--ef_search <ef_search>"
			 << "--k <k>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  --query_path         - Query vectors in .fvecs format\n";
        cerr << "  --query_ranges_file  - Query ranges (low-high per line, CSV)\n";
        cerr << "  --groundtruth_file   - Groundtruth in .ivecs format\n";
        cerr << "  --index_file         - Path to the saved index\n";
        cerr << "  --ef_search          - Search ef parameter\n";
        return 1;
    }

    // Store arguments
    string query_path, query_ranges_file, groundtruth_file, index_file;
    int ef_search = -1;
	int k = -1;

    for (int i = 1; i < argc; i += 2) {
        string arg = argv[i];
        if (arg == "--query_path") query_path = argv[i + 1];
        else if (arg == "--query_ranges_file") query_ranges_file = argv[i + 1];
        else if (arg == "--groundtruth_file") groundtruth_file = argv[i + 1];
        else if (arg == "--index_file") index_file = argv[i + 1];
        else if (arg == "--ef_search") ef_search = stoi(argv[i + 1]);
		else if (arg == "--k") k = stoi(argv[i + 1]);
		else {
			cerr << "Unknown argument: " << arg << endl;
			return 1;
		}
    }

    // Validate inputs
    if (query_path.empty() || query_ranges_file.empty() || 
        groundtruth_file.empty() || index_file.empty()) {
        cerr << "Error: Missing required arguments\n";
        return 1;
    }
    if (ef_search <= 0) {
        cerr << "Error: ef_search must be a positive integer\n";
        return 1;
    }

    // Load queries
    vector<vector<float>> queries = read_fvecs(query_path);
    int num_queries = queries.size();
    int dim = queries.empty() ? 0 : queries[0].size();

    // Load query ranges (format: "low-high" per line, e.g., "10-50")
    vector<pair<int, int>> query_ranges = read_two_ints_per_line(query_ranges_file);
    if (query_ranges.size() != num_queries) {
        cerr << "Error: Number of query ranges (" << query_ranges.size() 
             << ") != number of queries (" << num_queries << ")\n";
        return 1;
    }

    // Load groundtruth
    vector<vector<int>> groundtruth = read_ivecs(groundtruth_file);
    if (groundtruth.size() != num_queries) {
        cerr << "Error: Number of groundtruth entries (" << groundtruth.size() 
             << ") != number of queries (" << num_queries << ")\n";
        return 1;
    }

    // Truncate ground-truth to at most k items
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

    // Load the index
    hannlib::L2Space space(dim);
    // Load with max_elements = 0 means auto-detect from index file
    hannlib::ScalarHSIG<float> index(&space, index_file, false, 0);

    // Set search parameters
    index.set_ef(ef_search);

    // ========== QUERY EXECUTION (TIMED) ==========
    // Store results for later recall calculation
    vector<vector<int>> query_results(num_queries);
    auto start_time = high_resolution_clock::now();
    // Execute queries
	int64_t low, high;
    for (int i = 0; i < num_queries; i++) {
        low = query_ranges[i].first;
        high = query_ranges[i].second;
        
        // Perform hybrid search (range-filtered ANN)
        auto result = index.OptimizedHybridSearch(
            queries[i].data(), 
            k, 
            make_pair(low, high)
        );
        
        // Extract IDs from result priority queue
        query_results[i].reserve(k);
        while (!result.empty()) {
            query_results[i].push_back(result.top().second);
            result.pop();
        }
    }

    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    done = true;
    monitor.join();

	// Compute time and QPS
    double query_time_sec = duration_cast<duration<double>>(end_time - start_time).count();
    double qps = num_queries / query_time_sec;

    // Compute recall
	size_t match_count = 0;
	size_t total_count = 0;
	for (size_t q = 0; q < num_queries; q++) {
		int n_valid_neighbors = min(k, (int)groundtruth[q].size());
		vector<int> groundtruth_q = groundtruth[q];
		vector<int> nearest_neigbors_q;
		for (size_t i = 0; i < query_results[q].size(); i++) {
			nearest_neigbors_q.push_back(query_results[q][i]);
		}
		sort(groundtruth_q.begin(), groundtruth_q.end());
		sort(nearest_neigbors_q.begin(), nearest_neigbors_q.end());
		vector<int> intersection;
		set_intersection(groundtruth_q.begin(), groundtruth_q.end(), nearest_neigbors_q.begin(), nearest_neigbors_q.end(), back_inserter(intersection));
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}

	double recall = (double)match_count / total_count;

    // ========== OUTPUT RESULTS ==========
    cout << "Query time (s): " << query_time_sec << endl;
    cout << "Peak thread count: " << peak_threads.load()-1 << endl;
    cout << "QPS: " << qps << endl;
    cout << "Recall: " << recall << endl;
    
    // Memory footprint
    peak_memory_footprint();

    return 0;
}
