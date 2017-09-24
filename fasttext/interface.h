/* This is wrapper for fasttext(1) */
#ifndef FASTTEXT_INTERFACE_H
#define FASTTEXT_INTERFACE_H

#include <string>
#include <memory>

#include "cpp/src/fasttext.h"
#include "cpp/src/dictionary.h"

/* We create this to mimic cpp/src/fasttext.h to gain access on the private 
 * data like arguments and dictionary */
namespace interface {
    class FastTextModel {
        private:
            std::shared_ptr<fasttext::FastText> _fasttext;
            std::shared_ptr<fasttext::Dictionary> _dict;

            bool checkModel(std::istream& in);
        public:
            FastTextModel();

            bool qnorm;
            bool qout;
            bool retrain;
            double lr;
            double t;
            int bucket;
            int dim;
            int epoch;
            int lrUpdateRate;
            int maxn;
            int minCount;
            int minCountLabel;
            int minn;
            int neg;
            int wordNgrams;
            int ws;
            size_t cutoff;
            size_t dsub;
            std::string labelPrefix;
            std::string lossName;
            std::string modelName;

            void loadModel(std::string filename);
            int32_t dictGetNLabels();
            std::string dictGetLabel(int32_t i);
            std::vector<std::string> predict(std::string text, int32_t k);
            std::vector<std::vector<std::string>> predictProb(std::string text,
                int32_t k);
            std::string test(std::string filename, int32_t k);
    };
}

#endif

