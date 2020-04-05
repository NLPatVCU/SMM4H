#!/bin/bash
# python sentence_cnn_working.py glove200twitter 200 no_desample no_class_weights > glove200twitter.txt
# python sentence_cnn_working.py glove100twitter 100 no_desample no_class_weights  > glove100twitter.txt
# python sentence_cnn_working.py glove50twitter 50 no_desample no_class_weights > glove50twitter.txt
# python sentence_cnn_working.py glove50 50 no_desample no_class_weights > glove50.txt
python sentence_cnn_working.py glove100 100 no_desample no_class_weights  > glove100.txt
python sentence_cnn_working.py glove200 200 no_desample no_class_weights  > glove200.txt
python sentence_cnn_working.py glove200 200 no_desample no_class_weights  > glove200.txt
python sentence_cnn_working.py mimic200 200 no_desample no_class_weights  > mimic200.txt
python sentence_cnn_working.py mimic300 300 no_desample no_class_weights  > mimic300.txt
python sentence_cnn_working.py mimic400 400 no_desample no_class_weights  > mimic400.txt
