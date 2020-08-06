import sys
import io
import os
import pandas as pd
import sys
import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer


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

class Model:
    def __init__(self, Xdata_train, Ydata_train, Xdata_val, Ydata_val, maxwords, maxlen, data_test=None):

        self.maxwords = maxwords
        self.maxlen = maxlen

        self.X_data, self.binary_Y, self.word_index, self.labels = self.process_train(read_from_file(Xdata_train), read_from_file(Ydata_train))
        self.X_data_val, self.binary_Y_val = self.process_val(read_from_file(Xdata_val), read_from_file(Ydata_val))

        if data_test != None:
            self.X_data_test = self.process_test(read_from_file(data_test))

    def process_train(self, Xdata_train, Ydata_train):
        """
        Reads in X data and formats it correctly
        :return X_data: X train data prepared for model
        :return binary_Y: labels prepared for model
        :return word_index: word index of X_data ready for init_embedding function
        :return labels: list of what labels are in binary_Y
        """
        df_data = pd.DataFrame(Xdata_train, columns=['tweet'])
        df_label = pd.DataFrame(Ydata_train, columns=['label'])
        df_data.reset_index(drop=True, inplace=True)
        df_label.reset_index(drop=True, inplace=True)
        df = pd.concat((df_data, df_label), axis=1)

        tokenizer = Tokenizer(num_words=self.maxwords, lower=True)
        tokenizer.fit_on_texts(df['tweet'])
        X_data = self.get_features(df['tweet'], tokenizer)
        word_index = tokenizer.word_index

        binarizer = LabelBinarizer()
        binarizer.fit(df['label'])
        labels = binarizer.classes_
        print(labels)
        num_classes = len(labels)
        binary_y = binarizer.transform(df['label'])
        binary_Y = []
        for label_arr in binary_y:
            for label in label_arr:
                binary_Y.append(label)
        binary_Y = np.array(binary_Y)

        return X_data, binary_Y, word_index, labels


    def process_val(self, Xdata_val, Ydata_val):
        # for validation data
        """
        Prepares validation data for model
        :return X_data_val: X validation data ready for model
        :return binary_y_val: Y validation data ready for model
        """
        df_data_val = pd.DataFrame(Xdata_val, columns=['tweet'])
        df_label_val = pd.DataFrame(Ydata_val, columns=['label'])
        df_data_val.reset_index(drop=True, inplace=True)
        df_label_val.reset_index(drop=True, inplace=True)
        df_val = pd.concat((df_data_val, df_label_val), axis=1)


        tokenizer_val = Tokenizer(num_words=self.maxwords, lower=True)
        print(df_val['tweet'])
        tokenizer_val.fit_on_texts(df_val['tweet'])
        X_data_val = self.get_features(df_val['tweet'], tokenizer_val)

        binarizer_val = LabelBinarizer()
        binary_y_val = binarizer_val.fit(df_val['label'].astype(str))

        binary_y_val = binarizer_val.transform(df_val['label'])
        binary_Y_val = []
        for label_arr in binary_y_val:
            for label in label_arr:
                binary_Y_val.append(label)
        binary_Y_val = np.array(binary_Y_val)

        return X_data_val, binary_Y_val

    # for test data
    def process_test(self):
        """
        Prepares test data for model
        :return X_data_test: X test data prepared for model
        """
        df_test = pd.DataFrame(self.data_test, columns=['tweet'])
        tokenizer_test = Tokenizer(num_words=self.maxwords, lower=True)
        print(df_test['tweet'])
        tokenizer_test.fit_on_texts(df_test['tweet'])
        X_data_test = self.get_features(df_test['tweet'], tokenizer_test)
        return X_data_test

    def get_features(self, text_series, tokenizer):
        """
        Transforms text data to feature_vectors that can be used in the ml model.
        tokenizer must be available.
        :param text_series: text to create sequences from
        :param tokenizer: scikit learn tokenizer that has been fitted to text
        :return: padded sequences
        """
        sequences = tokenizer.texts_to_sequences(text_series)
        return pad_sequences(sequences, maxlen=self.maxlen)
