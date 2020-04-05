#!/bin/bash
# my desampling algo
# python desample.py 1
# python sentence_cnn_working.py glove50 50 desample no_class_weights no_imb > glove50desample1.txt
# python desample.py 2
# python sentence_cnn_working.py glove50 50 desample no_class_weights no_imb > glove50desample2.txt
# python desample.py 4
# python sentence_cnn_working.py glove50 50 desample no_class_weights no_imb > glove50desample4.txt
# python desample.py 6
# python sentence_cnn_working.py glove50 50 desample no_class_weights no_imb > glove50desample6.txt
# python desample.py 8
# python sentence_cnn_working.py glove50 50 desample no_class_weights no_imb > glove50desample8.txt

# python sentence_cnn_working.py glove50 50 no_desample class_weights no_imb_algo > glove50weights.txt
# python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo SMOTE > glove50SMOTE.txt
python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo ADAYSN > glove50ADASYN.txt
# python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo RandomOver > glove50RandomOver.txt
python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo RandomUnder > glove50RandomUnder.txt
# python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo SMOTEENN > glove50SMOTEENN.txt
# python sentence_cnn_working.py glove50 50 no_desample no_class_weights imb_algo SMOTETomek > glove50SMOTETomek.txt
