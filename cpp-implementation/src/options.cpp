#include "options.h"

struct opt opt;

void get_options(CLI::App& app, int argc, char* argv[]) {

  app.add_option("subword_vocabulary",
    opt.subword_vocab_file, "Subword vocabulary, subword per line.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("word_vocabulary",
    opt.word_vocab_file, "Word vocabulary, word per line.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("input",
    opt.training_data_file, "Tokenized text.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("fasttext",
    opt.fasttext_output_matrix, "Pseudo-inverse of the fasttext output matrix.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("output",
    opt.output_file, "Matrix data.")
    ->required()
    ->check(CLI::NonexistentPath);

  app.add_option("--allowed-substrings",
    opt.allowed_substrings, "List of words accompanied with allowed substrings.")
    ->check(CLI::ExistingFile);

  app.add_option("--max-subword",
    opt.max_subword, "Maximum subword length.");

  app.add_option("--window-size",
    opt.window_size, "Window size.");

  app.add_option("--fasttext-dim",
    opt.fasttext_dim, "Dimension of the fasttext embeddings.");

  app.add_option("--buffer-size",
    opt.buffer_size, "Buffer size.");

  app.add_option("--shard-size",
    opt.shard_size, "Shard size for matrix multiplication.");

}