import sys
import io
import os
import pandas as pd
import sys
import keras
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
from imblearn.over_sampling import SMOTE, ADASYN
from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras_wc_embd import get_dicts_generator, get_embedding_layer, get_embedding_weights_from_file
import chars2vec
import talos

class CNN:
    def __init__(self, x_train, y_train, embedding_matrix, x_val, y_val, labels, dim, maxlen, maxwords, filter_length, cross_val, weights, weight_ratios, test, x_test):
        """
        Runs CNN.

        :param x_train: X train data pre-processed
        :param y_train: Y train data pre-processed
        :param embedding_matrix: embedding ready for embedding layer
        :param x_val: X validation data pre-processed
        :param y_val: Y validation data pre-processed
        :param labels: list of unique labels
        :param dim: dimension of word embedding
        :param maxlen: maximum input length of a tweet
        :param maxwords: maximum words
        :param filter_length: length of filter
        :param cross_val: flag for cross validation
        :param x_test: test data
        """

        self.labels = labels
        self.dim = dim
        self.maxlen = maxlen
        self.maxwords = maxwords
        self.filter_length = filter_length
        self.cross_val = cross_val
        self.weights = weights

        if test:
            self.x_test = x_test
            self.test = True
        else:
            self.test = False

        if self.weights:
            self.weight_ratios = weight_ratios

        self.x_train = x_train
        self.y_train = y_train
        self.embedding_matrix = embedding_matrix
        self.x_val = x_val
        self.y_val = y_val

        if self.cross_val:
            self.cv()
        else:
            self.train_test()


    def predict(self, model, x_test, y_test, encoder_classes):
        """
        Takes the predictions as input and returns the indices of the maximum values along an axis using numpy argmax function as true labels.
        Then evaluates it against the trained model

        :param model: trained model
        :param x_test: test data
        :param y_test: test true labels
        :param encoder_classes:
        :return: predicted and true labels
        """
        pred = model.predict(x_test)
        y_true = y_test
        y_pred_ind = np.argmax(pred, axis=1)
        y_pred = [encoder_classes[i] for i in y_pred_ind]
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print("Accuracy :", test_acc)
        print("Loss : ", test_loss)

        return y_pred, y_true

    def cv_evaluation_fold(self, y_pred, y_true, labels):
        """
        Evaluation metrics for emicroach fold

        :param y_pred: predicted labels
        :param y_true: true labels
        :param labels: list of the classes
        :return: fold stats
        """
        fold_statistics = {}
        for label in labels:
            fold_statistics[label] = {}
            f1 = f1_score(y_true, y_pred, average='micro', labels=[label])
            precision = precision_score(y_true, y_pred, average='micro', labels=[label])
            recall = recall_score(y_true, y_pred, average='micro', labels=[label])
            fold_statistics[label]['precision'] = precision
            fold_statistics[label]['recall'] = recall
            fold_statistics[label]['f1'] = f1

        # add averages
        fold_statistics['system'] = {}
        f1 = f1_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        fold_statistics['system']['precision'] = precision
        fold_statistics['system']['recall'] = recall
        fold_statistics['system']['f1'] = f1

        return fold_statistics

    def prediction_to_label(self, prediction):
        """
        Turns the prediction into a label.

        :param prediction: prediction for X data
        :return: labels in dictionary form
        """
        tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
        return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))

    def fit_Model(self, model, x_train, y_train):
        """
        fit the defined model to train on the data

        :param model: trained model
        :param x_train: training data
        :param y_train: training labels
        :return: model and loss & accuracy stats
        """
        if self.weights:
            class_weight= {0:self.weight_ratios[0], 1:self.weight_ratios[1]}
            history = model.fit(x_train, y_train, epochs=20,
                                batch_size=512, class_weight=class_weight)
            loss = history.history['loss']
            acc = history.history['accuracy']
        else:
            history = model.fit(x_train, y_train, epochs=20,
                                batch_size=512)
            print(history.history.keys())
            loss = history.history['loss']
            acc = history.history['accuracy']


        return model, loss, acc

    def test_data(self, model):
        pred = model.predict(self.x_test)
        y_pred_ind = np.argmax(pred, axis=1)
        pred_labels = [self.labels[i] for i in y_pred_ind]

        dataset = pd.read_csv("../../data/test/test.tsv", sep='\t')
        tweets_id = dataset['tweet_id'].tolist()
        tweets = dataset['tweet'].tolist()
        d = {'tweet_id':tweets_id, 'tweets':tweets, 'Class': pred_labels}

        df = pd.DataFrame(data=d)
        df.to_csv("../../data/test/test_new.tsv", sep='\t')

    def cv(self):
        """
        This function does the cross validation.

        :param x_train_data: processed X train data.
        :param y_train_data: processed Y train data.
        :param embedding_matrix: embedding perpared for embedding layer.
        :param x_val: processed X validation data.
        :param y_val: processed Y validation data.
        :param labels: list of labels.
        param X_data_test: processed test data.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        skf.get_n_splits(self.x_train, self.y_train)

        fold = 1
        originalclass = []
        predictedclass = []


        for train_index, test_index in skf.split(self.x_train, self.y_train):

            x_train, x_test = self.x_train[train_index], self.x_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            print("Training Fold %i" % fold)
            print(len(x_train), len(x_test))
            filter_length = 64

            model = Sequential()
            model.add(Embedding(self.maxwords, self.dim, weights=[self.embedding_matrix], input_length=self.maxlen))
            model.add(Conv1D(self.filter_length, 1, activation='relu'))
            model.add(MaxPool1D(5))
            model.add(Conv1D(self.filter_length, 1, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(self.filter_length, activation='relu'))
            model.add(Dense(2, activation='sigmoid'))
            model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            cv_model, loss, acc = self.fit_Model(model, x_train, y_train)
            y_pred, y_true = self.predict(cv_model, x_test, y_test, self.labels)
            y_true = [str(lab) for lab in y_true]
            originalclass.extend(y_true)
            predictedclass.extend(y_pred)

            print("--------------------------- Results ------------------------------------")
            print(classification_report(y_true, y_pred, labels=self.labels))
            print(confusion_matrix(y_true, y_pred))
            fold_statistics = self.cv_evaluation_fold(y_pred, y_true, labels=self.labels)

            fold += 1

        y_pred_val, y_true_val = self.predict(cv_model, self.x_val, self.y_val, self.labels)
        y_true_val = [str(lab) for lab in y_true_val]
        print("--------------------------- Results ------------------------------------")
        print(classification_report(y_true_val, y_pred_val, labels=self.labels))
        print(confusion_matrix(y_true_val, y_pred_val))
        if self.test:
            self.test_data()


    def train_test(self):
        model = Sequential()
        model.add(Embedding(self.maxwords, self.dim, weights=[self.embedding_matrix], input_length=self.maxlen))
        model.add(Conv1D(self.filter_length, 1, activation='relu'))
        model.add(MaxPool1D(5))
        model.add(Conv1D(self.filter_length, 1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.filter_length, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(self.x_train, self.y_train,
                            epochs=20,
                            batch_size=512)

        y_pred_val, y_true_val = self.predict(model, self.x_val, self.y_val, self.labels)
        y_true_val = [str(lab) for lab in y_true_val]
        # y_pred = np.array(model.predict(x_val))

        print(classification_report(y_true_val, y_pred_val, target_names=self.labels))
        print(confusion_matrix(y_true_val, y_pred_val))

        if self.test:
            self.test_data(model)
