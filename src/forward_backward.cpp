#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "CLI11.hpp"
#include "vocabs.h"
#include <Eigen/Dense>
#include "unigram_model.h"

struct opt {
    std::string embeddings_file;
    std::string subword_vocab_file;
    std::string inverse_embeddings_file;
    std::string saved_model_file;
    std::string load_model_file;
    float base_logprob;
    int word_count;
    int epochs = 1;
} opt;

void get_options(CLI::App& app, int argc, char* argv[]) {
    app.add_option("embeddings_file",
        opt.embeddings_file, "Word embeddings.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("subword_vocab_file",
        opt.subword_vocab_file, "List of subwords")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("pseudo_inverse_embeddings",
        opt.inverse_embeddings_file, "File with a pseudo-inverse matrix of word embeddings")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("saved_model_file",
        opt.saved_model_file, "File to save the model to")
        ->required()
        ->check(CLI::NonexistentPath);

    app.add_option("--load-model",
        opt.load_model_file, "Load model from file")
        ->check(CLI::ExistingFile);

    app.add_option("--epochs",
        opt.epochs, "Number of epochs");

    app.add_option("--base-logprob",
        opt.base_logprob, "Logprob of unseen subwords aka smoothing");
}

int main(int argc, char* argv[]) {

    CLI::App app{
        "Byte-based Forward-backward EM estimation of subword embeddings."};
    get_options(app, argc, argv);
    CLI11_PARSE(app, argc, argv);

    std::cerr << "Loading subword vocab: " << opt.subword_vocab_file << std::endl;
    Vocab subword_vocab(opt.subword_vocab_file);

    std::cerr << "Loading embedding matrix from " << opt.embeddings_file
              << std::endl;
    Embeddings words(opt.embeddings_file);

    // std::cerr << "Top-left corner of the embedding matrix:" << std::endl;
    // std::cerr << words.emb.block(0,0,5,5) << std::endl;

    // std::cerr << "Bottom-right corner of the embedding matrix:" << std::endl;
    // std::cerr << words.emb.bottomRightCorner(5, 5) << std::endl;

    int embedding_dim = words.embedding_dim;
    int word_count = words.size();

    std::cerr << "Loading the pseudo-inverse embedding matrix from " << opt.inverse_embeddings_file << std::endl;
    std::ifstream inv_embedding_fh(opt.inverse_embeddings_file);
    Eigen::MatrixXf inverse_emb(embedding_dim, word_count);

    int i = 0;
    for(std::string line; std::getline(inv_embedding_fh, line); ++i) {
        std::stringstream ss(line);
        for(int j = 0; j < word_count; ++j) {
            ss >> inverse_emb(i, j);
        }
    }

    // std::cerr << "Top-left corner of the inverse embedding matrix:" << std::endl;
    // std::cerr << inverse_emb.block(0,0,5,5) << std::endl;ů

    // std::cerr << "Bottom-right corner of the inverse embedding matrix:" << std::endl;
    // std::cerr << inverse_emb.bottomRightCorner(5, 5) << std::endl;

    std::vector<std::string> test_words({"včelař", "hokejista", "podpatek", "náramný", "veličenstvo"});
    UnigramModel model(words, subword_vocab, inverse_emb);

    if(!opt.load_model_file.empty()) {
        std::cerr << "Loading model from " << opt.load_model_file << std::endl;
        model.load(opt.load_model_file);
    }

    for(int i = 0; i < opt.epochs; ++i) {
        std::cerr << "Iteration " << i + 1 << std::endl;
        model.estimate_parameters(/*epochs=*/ 1, opt.base_logprob);

        for(auto word: test_words) {
            std::cerr << "TEST " << word;
            std::vector<std::string> segm;
            model.viterbi_decode(segm, word);

            for(auto it = segm.rbegin(); it != segm.rend(); ++it) {
                std::cerr << " " << *it;
            }
            std::cerr << std::endl;
        }
    }

    model.save(opt.saved_model_file);

    return 0;
}

