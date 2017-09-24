# fastText C++ interface
# This file is act as a bridge between interface.{h,cc} with the fasttext.pyx
from libcpp.string cimport string
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "interface.h" namespace "interface":
    cdef cppclass FastTextModel:
        FastTextModel()
        
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
        string labelPrefix;
        string lossName;
        string modelName;

        void loadModel(string filename)
        int32_t dictGetNLabels()
        string dictGetLabel(int32_t i)

        vector[string] predict(string text, int32_t k)
        vector[vector[string]] predictProb(string text, int32_t k)
        string test(string filename, int32_t k)
