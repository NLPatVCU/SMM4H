#!/bin/bash

# oversample

python oversample.py 7
python sentence_cnn.py glove100twitter 100 oversample none > oversample_times7_glove100twitter.txt

python oversample.py 7
python sentence_cnn.py glove50twitter 50 oversample none > oversample_times7_glove50twitter.txt


# desample

python desample.py 4
python sentence_cnn.py glove50twitter 50 desample none > desample_1to4_glove50twitter.txt

python desample.py 4
python sentence_cnn.py glove100twitter 100 desample none > desample_1to4_glove100twitter.txt


# weights
python sentence_cnn.py glove100twitter 100 weights none 1 10 > weights_1to10_glove100twitter.txt

python sentence_cnn.py glove50twitter 50 weights none 1 10 > weights_1to10_glove50twitter.txt
