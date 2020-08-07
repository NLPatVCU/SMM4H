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
    def __init__(self, x_data, y_data, x_val, y_val, embedding_path, embedding_demension = 50, max_words=10000, max_len=300, SMOTE_flag=False):
        self.x_data = read_from_file(x_data)
        self.y_data = read_from_file(y_data)
        self.x_val = read_from_file(x_val)
        self.y_val = read_from_file(y_val)
        self.SMOTE_flag = SMOTE_flag
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_path = embedding_path
        self.embedding_demension = embedding_demension

        self.y_train_data, self.labels, self.num_classes = self.binarize_labels(self.y_data)
        self.y_validation_data = self.binarize_labels(self.y_val)[0]
        self.tokenizer, self.x_train_data = self.tokenize(self.x_data)
        self.x_validation_data = self.tokenize(self.x_val)[1]

        self.embeddings_index = self.read_embeddings_from_file(self.embedding_path)
        self.embedding = self.create_embedding()

    def binarize_labels(self, labels):
        binarizer = LabelBinarizer()
        binarizer.fit(labels)
        labels = binarizer.classes_
        print(labels)
        num_classes = len(labels)
        binary_Y = [int(label) for label in labels]
        return binary_Y, labels, num_classes

    def tokenize(self, sentences):
        tokenizer = Tokenizer(num_words=self.max_words, lower=False)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        X_data_tokenized = pad_sequences(sequences, maxlen=self.max_len)
        return tokenizer, X_data_tokenized

    def create_embedding(self):
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((self.max_words, self.embedding_demension))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if i < self.max_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            return embedding_matrix

    def read_embeddings_from_file(self, path):
        """
        Function to read external embedding files to build an index mapping words (as strings)
        to their vector representation (as number vectors).
        :return dictionary: word vectors
        """
        print("Reading external embedding file ......")
        if not os.path.isfile(path):
            raise FileNotFoundError("Not a valid file path")

        embeddings_index = {}
        with open(path) as f:
            next(f)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
        return embeddings_index

o1 = Data_Model("../data/train/tweets", "../data/train/labels", "../data/validation/tweets_val", "../data/validation/labels_val", "../../embeddings/glove.6B.50d.txt")
