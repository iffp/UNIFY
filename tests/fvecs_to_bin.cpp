#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Read fvecs format
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

// Write .bin format (UNIFY format - just format conversion)
void write_bin(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for writing\n";
        exit(1);
    }
    
    int num_points = data.size();
    int dim = data.empty() ? 0 : data[0].size();
    
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    
    for (const auto& vec : data) {
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    
    file.close();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.fvecs> <output.bin>\n";
        std::cerr << "  Converts .fvecs to .bin format (no sorting)\n";
        return 1;
    }

    std::string input_fvecs = argv[1];
    std::string output_bin = argv[2];

    std::cout << "Reading vectors from " << input_fvecs << "...\n";
    std::vector<std::vector<float>> vectors = read_fvecs(input_fvecs);
    
    std::cout << "Writing vectors to " << output_bin << "...\n";
    write_bin(output_bin, vectors);
    
    std::cout << "Conversion completed successfully!\n";
    std::cout << "  Vectors: " << vectors.size() << "\n";
    std::cout << "  Dimension: " << (vectors.empty() ? 0 : vectors[0].size()) << "\n";
    
    return 0;
}
