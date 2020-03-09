import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.model import evaluate


def read_from_file(file):
    """
    Reads external files and insert the content to a list. It also removes whitespace
    characters like `\n` at the end of each line

    :param file: name of the input file.
    :return : content of the file in list format
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


class Data_Model:
    def __init___(self, x_data, y_data, x_val, y_val, max_words=5000, SMOTE_flag=False):
        self.x_data = read_from_file(x_data)
        self.y_data = read_from_file(y_data)
        self.x_val = read_from_file(x_val)
        self.y_val = read_from_file(y_val)
        self.SMOTE_flag = SMOTE_flag
        self.max_words = max_words

        self.y_train_data, self.labels, self.num_classes = self.binarize_labels(self.y_data)
        self.y_validation_data = self.binarize_labels(self.y_val)[0]
        self.x_train_data = self.tokenize(self.x_data)
        self.x_validation_data = self.tokenize(self.x_val)

    def binarize_labels(self, labels):
        binarizer = LabelBinarizer()
        binarizer.fit(labels)
        labels = binarizer.classes_
        print(labels)
        num_classes = len(labels)
        binary_Y = [int(label) for label in labels]
        return binary_Y, labels, num_classes

    def tokenize(self, sentences):
        tokenizer = Tokenizer(num_words=max_words, lower=True)
        tokenizer.fit_on_texts(sentences)
        X_data_tokenized = get_features(sentences)
        return tokenizer, X_data_tokenized

    def create_embedding(self):
        word_index = tokenizer.word_index
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            return embedding_matrix

    def read_embedding(self):
