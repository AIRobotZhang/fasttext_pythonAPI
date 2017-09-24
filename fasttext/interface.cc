/* An interface for fastText */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "interface.h"
#include "cpp/src/fasttext.h"
#include "cpp/src/dictionary.h"
#include "cpp/src/real.h"

namespace interface {
    FastTextModel::FastTextModel(){}

    /* mimic private function FastText::checkModel */
    bool FastTextModel::checkModel(std::istream& in) 
    {
        int32_t magic;
        int32_t version;
        in.read((char*)&(magic), sizeof(int32_t));
        if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
            return false;
        }
        in.read((char*)&(version), sizeof(int32_t));
        if (version != FASTTEXT_VERSION) {
            return false;
        }
        return true;
    }

    void FastTextModel::loadModel(std::string filename)
    {
        // Load model
        std::shared_ptr<fasttext::FastText> ft = std::make_shared<fasttext::FastText>();
        ft->loadModel(filename);
        _fasttext = ft;

        // We need to re-check the model to read the args and the dictionary
        std::ifstream ifs(filename, std::ifstream::binary);
        if (!checkModel(ifs)) {
            std::cerr << "interface.cc: Model file has wrong file format!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Load model parameters
        std::shared_ptr<fasttext::Args> args = std::make_shared<fasttext::Args>();
        args->load(ifs);
        bucket = args->bucket;
        cutoff = args->cutoff;
        dim = args->dim;
        dsub = args->dsub;
        epoch = args->epoch;
        labelPrefix = args->label;
        lr = args->lr;
        lrUpdateRate = args->lrUpdateRate;
        maxn = args->maxn;
        minCount = args->minCount;
        minCountLabel = args->minCountLabel;
        minn = args->minn;
        neg = args->neg;
        qnorm = args->qnorm;
        qout = args->qout;
        retrain = args->retrain;
        t = args->t;
        wordNgrams = args->wordNgrams;
        ws = args->ws;

        if(args->loss == fasttext::loss_name::ns) {
            lossName = "ns";
        }
        if(args->loss == fasttext::loss_name::hs) {
            lossName = "hs";
        }
        if(args->loss == fasttext::loss_name::softmax) {
            lossName = "softmax";
        }
        if(args->model == fasttext::model_name::cbow) {
            modelName = "cbow";
        }
        if(args->model == fasttext::model_name::sg) {
            modelName = "skipgram";
        }
        if(args->model == fasttext::model_name::sup) {
            modelName = "supervised";
        }

        // Load dictionary
        std::shared_ptr<fasttext::Dictionary> dict = std::make_shared<fasttext::Dictionary>(args);
        dict->load(ifs);
        _dict = dict;
    }

    int32_t FastTextModel::dictGetNLabels()
    {
        return _dict->nlabels();
    }

    std::string FastTextModel::dictGetLabel(int32_t i)
    {
        return _dict->getLabel(i);
    }

    /* Interface for ./fasttext predict */
    std::vector<std::string> 
    FastTextModel::predict(std::string text, int32_t k)
    {
        /* Convert string into input stream */
        std::istringstream in(text);

        /* Run the prediction */
        std::vector<std::pair<fasttext::real,std::string>> predictions;
        _fasttext->predict(in, k, predictions);

        /* Forward the label */
        std::vector<std::string> labels;
        for(auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            labels.push_back(it->second);
        }
        return labels;
    }

    /* Interface for ./fasttext predict-prob */
    std::vector<std::vector<std::string>>
    FastTextModel::predictProb(std::string text, int32_t k)
    {
        /* Convert string into input stream */
        std::istringstream in(text);

        /* Run the prediction */
        std::vector<std::pair<fasttext::real,std::string>> predictions;
        _fasttext->predict(in, k, predictions);

        /* Forward the results */
        std::vector<std::vector<std::string>> results;
        for(auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            std::vector<std::string> result;
            result.push_back(it->second);

            /* We use string stream here instead of to_string, to make sure
             * that the string is consistent with std::cout from fasttext(1) */
            std::ostringstream probability_stream;
            probability_stream << exp(it->first);
            result.push_back(probability_stream.str());

            results.push_back(result);
        }
        return results;
    }

    /* Interface for ./fasttext test */
    std::string FastTextModel::test(std::string filename, int32_t k)
    {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            std::cerr << "interface.cc: Test file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        /* Get the current output stream */
        std::streambuf* old_ofs = std::cout.rdbuf();
        /* Initialize new string stream; to save the test output */
        std::stringstream buffer;
        /* Redirect the std::cout to buffer */
        std::cout.rdbuf(buffer.rdbuf());
        /* Run the test */
        _fasttext->test(ifs, k);
        /* Restore the output stream */
        std::cout.rdbuf(old_ofs);
        /* Convert buffer to plain string */
        std::string test_output = buffer.str();
        return test_output;
    }
}
