#!/bin/bash
# python3 sentence_cnn.py fasttext 100 none none > fasttexttwitter.txt
# python3 sentence_cnn.py word2vectwitter 400 none none > word2vectwitter.txt
# python3 desample.py 4
# python3 sentence_cnn.py word2vectwitter 400 desample none > word2vectwitterdesample4.txt

# python oversample.py 7
# python3 sentence_cnn.py word2vectwitter 400 oversample none > word2vectwitteroversample7.txt

python3 sentence_cnn.py word2vectwitter 400 weights none  2 20 > word2vectwitterweights2_20.txt

# python3 sentence_cnn.py word2vectwitter 400 none SMOTE > word2vectwitterSMOTE.txt

python3 sentence_cnn.py word2vectwitter 400 none ADASYN > word2vectwitterADASYN.txt
