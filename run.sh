#!/bin/bash
python sentence_cnn_working.py glove50 50 no_desample class_weights no_imb_algo > glove50weights.txt
python sentence_cnn_working.py glove50 50 no_desample class_weights imb_algo SMOTE > glove50SMOTE.txt
python sentence_cnn_working.py glove50 50 no_desample class_weights imb_algo ADAYSN > glove50ADASYN.txt
python sentence_cnn_working.py glove50 50 no_desample class_weights imb_algo RandomOver > glove50RandomOver.txt
python sentence_cnn_working.py glove50 50 no_desample class_weights imb_algo RandomUnder > glove50RandomUnder.txt
