#!/bin/bash

# desampling
python desample.py 1
python sentence_cnn.py glove200 200 desample none > desample_1to1_glove200.txt
python desample.py 4
python sentence_cnn.py glove200 200 desample none > desample_1to4_glove200.txt
python desample.py 8
python sentence_cnn.py glove200 200 desample none > desample_1to8_glove200.txt

# oversampling
python oversample.py 9
python sentence_cnn.py glove200 200 oversample none > oversample_times9_glove200.txt
python oversample.py 7
python sentence_cnn.py glove200 200 oversample none > oversample_times7_glove200.txt
python oversample.py 4
python sentence_cnn.py glove200 200 oversample none > oversample_times4_glove200.txt
python oversample.py 2
python sentence_cnn.py glove200 200 oversample none > oversample_times2_glove200.txt

# SMOTE
python sentence_cnn.py glove200 200 none SMOTE > SMOTE_glove200.txt

# Class weights
python sentence_cnn.py glove200 200 weights none 1 5 > weights_1to5_glove200.txt
python sentence_cnn.py glove200 200 weights none 1 10 > weights_1to10_glove200.txt
python sentence_cnn.py glove200 200 weights none 1 20 > weights_1to20_glove200.txt
python sentence_cnn.py glove200 200 weights none 2 20 > weights_2to20_glove200.txt

# Embeddings
python sentence_cnn.py glove50 50 none none > glove50.txt
python sentence_cnn.py glove100 100 none none > glove100.txt
python sentence_cnn.py glove200 200 none none > glove200.txt
python sentence_cnn.py glove300 300 none none > glove300.txt
python sentence_cnn.py glove200twitter 200 none none > glove200twitter.txt
python sentence_cnn.py glove100twitter 100 none none > glove100twitter.txt
python sentence_cnn.py glove50twitter 50 none none > glove50twitter.txt
python sentence_cnn.py mimic200 200 none none > mimic200.txt
python sentence_cnn.py mimic300 300 none none > mimic300.txt
python sentence_cnn.py mimic400 400 none none > mimic300.txt
python sentence_cnn.py wikivectors 200 none none > wikivectors200.txt
