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


class Model:
    def __init__(self, Xdata_train, Ydata_train, Xdata_val, Ydata_val, maxwords, maxlen, test=False, data_test=None):
        """
        Prepares data for CNN

        :param Xdata_train: preprocessed X train data
        :type Xdata_train: List
        :param Ydata_train: preprocessed Y train data
        :type Ydata_train: List
        :param Xdata_val: preprocessed X validation data
        :type Xdata_val: List
        :param Ydata_val: preprocessed Y validation data
        :type Ydata_val: List
        :param maxwords: maximum words to use
        :type maxwords: Int
        :param maxlen: maximum input length for tweet
        :type maxlen: Int
        :param test: test data flag
        :type test: Bool
        :param data_test: preprocessed test data
        :type data_test: List
        """

        self.maxwords = maxwords
        self.maxlen = maxlen

        # process train & validation data
        self.X_data, self.binary_Y, self.word_index, self.labels, self.tokenizer = self.process_train(Xdata_train, Ydata_train)
        self.X_data_val, self.binary_Y_val = self.process_val(Xdata_val, Ydata_val, self.tokenizer)

        # process test data if it exists
        if test:
            self.X_data_test = self.process_test(data_test)

    def process_train(self, Xdata_train, Ydata_train):
        """
        Reads in X data and formats it correctly.

        :param Xdata_train: CSV file of X train data read in via the read_from_file function
        :param Ydata_train: CSV file of Y train data read in via the read_from_file function
        :return: X & Y train data, word_index, labels
        """

        # creates data frame
        df_data = pd.DataFrame(Xdata_train, columns=['tweet'])
        df_label = pd.DataFrame(Ydata_train, columns=['label'])
        df_data.reset_index(drop=True, inplace=True)
        df_label.reset_index(drop=True, inplace=True)
        df = pd.concat((df_data, df_label), axis=1)

        # creates tokenizer
        tokenizer = Tokenizer(num_words=self.maxwords, lower=True)
        tokenizer.fit_on_texts(df['tweet'])
        X_data = self.get_features(df['tweet'], tokenizer)
        word_index = tokenizer.word_index

        # binarizes Y data
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

        return X_data, binary_Y, word_index, labels, tokenizer


    def process_val(self, x_data_val, y_data_val, tok):
        """
        Prepares validation data for model.

        :param Xdata_val: x validation data
        :type Xdata_val: List
        :param Ydata_val: y validation data
        :type Ydata_val: List
        :return: X&Y validation data ready for model
        """
        df_data_val = pd.DataFrame(x_data_val, columns=['tweet'])
        df_label_val = pd.DataFrame(y_data_val, columns=['label'])
        df_data_val.reset_index(drop=True, inplace=True)
        df_label_val.reset_index(drop=True, inplace=True)
        df_val = pd.concat((df_data_val, df_label_val), axis=1)



        # tokenizer_val = Tokenizer(num_words=self.maxwords, lower=True)
        print(df_val['tweet'])
        # tokenizer_val.fit_on_texts(df_val['tweet'])
        X_data_val = self.get_features(df_val['tweet'], tok)

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
    def process_test(self, data_test):
        """
        Prepares test data for model.

        :param data_test: x test data
        :param data_test: List
        :return: X test data prepared for model
        :rtype: List
        """
        # creates data frame
        df_test = pd.DataFrame(data_test, columns=['tweet'])

        # tokenizer
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
