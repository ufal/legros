/**
 merge contexts
 */

#include <memory>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include "CLI11.hpp"

void process_buffer(const std::vector<std::vector<std::string>> &buffer,
    std::ofstream &fout, size_t num_inputs, int num_cols) {

  std::vector<std::string> output_lines(buffer.size());

  #pragma omp parallel for
  for(int i = 0; i < buffer.size(); ++i) {

    // sum i-th rows in all files (indexed by j); columns indexed by k.
    std::vector<int> column_sums(num_cols, 0);

    for(int j = 0; j < num_inputs; ++j) {
      std::istringstream line(buffer[i][j]);

      for(int k = 0; k < num_cols; ++k) {
        int n;
        line >> n;

        column_sums[k] += n;
      }
    }

    std::ostringstream oss;
    for(const auto &e : column_sums) {
      oss << e << " ";
    }

    oss << "\n";
    output_lines[i] = oss.str();
  }

  for(int i = 0; i < buffer.size(); ++i) {
    fout << output_lines[i];
  }
}


int main(int argc, char* argv[]) {
  CLI::App app{"Merge contexts."};

  std::vector<std::string> input_files;
  std::string output_file;
  size_t buffer_size = 5000;
  int num_cols;

  app.add_option("output",
    output_file, "Output matrix data file. Will have same shape as inputs.")
    ->required()
    ->check(CLI::NonexistentPath);

  app.add_option("input",
    input_files, "List of input files, products of get_substring_contexts. Should have same shape.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("--num-cols",
    num_cols, "Number of columns (that's word vocabulary size (eg 200000)).")
    ->required();

  app.add_option("--buffer-size",
    buffer_size, "Buffer size.");

  CLI11_PARSE(app, argc, argv);

  size_t num_inputs = input_files.size();
  if(num_inputs < 2) {
    std::cerr << "need more than one input" << std::endl;
    return 1;
  }

  std::vector<std::vector<std::string>> buffer(buffer_size, std::vector<std::string>(num_inputs));
  std::vector<std::unique_ptr<std::ifstream>> fins;
  std::ofstream fout(output_file);

  for(int i = 0; i < num_inputs; ++i) {
    fins.push_back(std::make_unique<std::ifstream>(std::ifstream(input_files[i])));
  }

  int lineno = 0;
  int buffer_pos = 0;

  while(std::getline(*fins.at(0), buffer[buffer_pos][0])) {
    for(int i = 1; i < num_inputs; ++i) {
      if(!std::getline(*fins[i], buffer[buffer_pos][i])) {
        std::cerr << "\n" << i + 1 <<  "-th file (" << input_files[i] << ") ends prematurely." << std::endl;
        return 2;
      }
    }

    ++lineno;
    ++buffer_pos;
    std::cerr << "Lineno: " << lineno << "\r";

    // full buffer -> process
    if(buffer_pos == buffer_size) {
      process_buffer(buffer, fout, num_inputs, num_cols);
      buffer_pos = 0;
    }
  }

  // process the rest of the buffer
  if(buffer_pos > 0) {
    process_buffer(buffer, fout, num_inputs, num_cols);
  }

  for(int i = 0; i < num_inputs; ++i) {
    fins[i]->close();
  }
  fout.close();

  return 0;
}