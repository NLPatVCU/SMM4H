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
import talos
sys.path.append(".")
from model import Model
from embedding import MakeEmbedding

class CNN:
    def __init__(self, x_train, y_train, embedding_matrix, x_val, y_val, labels, dim, maxlen, maxwords, filter_length, cross_val=True, char=False, char_embedding_matrix=None, x_test=None):
        self.labels = labels
        self.dim = dim
        self.maxlen = maxlen
        self.maxwords = maxwords
        self.filter_length = filter_length
        self.cross_val = cross_val
        self.char = char
        self.char_embedding_matrix = char_embedding_matrix

        if x_test != None:
            self.x_test = X_test
            self.test = True
        else:
            self.test = False

        self.x_train = x_train
        self.y_train = y_train
        self.embedding_matrix = embedding_matrix
        self.x_val = x_val
        self.y_val = y_val

        if self.cross_val:
            self.cv(self.x_train, self.y_train, self.embedding_matrix, self.x_val, self.y_val, self.labels, self.char_embedding_matrix)
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
        if False:
            class_weight= {0:int(sys.argv[5]), 1:int(sys.argv[6])}
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

    def cv(self, x_train_data, y_train_data, embedding_matrix, x_val, y_val, labels, char_embedding_matrix=None, X_data_test=None):
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
        skf.get_n_splits(x_train_data, y_train_data)

        fold = 1
        originalclass = []
        predictedclass = []


        for train_index, test_index in skf.split(x_train_data, y_train_data):

            x_train, x_test = x_train_data[train_index], x_train_data[test_index]
            y_train, y_test = y_train_data[train_index], y_train_data[test_index]
            print("Training Fold %i" % fold)
            print(len(x_train), len(x_test))
            filter_length = 64

            if self.char:
                chars = Sequential()
                chars.add(Embedding(5000, 50, weights=[char_embedding_matrix], input_length=self.maxlen))
                words = Sequential()
                words.add(Embedding(self.maxwords, self.dim, weights=[embedding_matrix], input_length=self.maxlen))

                model = Sequential()
                model.add(Concatenate(axis=-1)([chars, words]))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(MaxPool1D(5))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(2, activation='sigmoid'))
                model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            else:

                model = Sequential()
                model.add(Embedding(self.maxwords, self.dim, weights=[embedding_matrix], input_length=self.maxlen))
                model.add(Conv1D(self.filter_length, 1, activation='relu'))
                model.add(MaxPool1D(5))
                model.add(Conv1D(self.filter_length, 1, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(2, activation='sigmoid'))
                model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            cv_model, loss, acc = self.fit_Model(model, x_train, y_train)
            y_pred, y_true = self.predict(cv_model, x_test, y_test, labels)
            y_true = [str(lab) for lab in y_true]
            originalclass.extend(y_true)
            predictedclass.extend(y_pred)

            print("--------------------------- Results ------------------------------------")
            print(classification_report(y_true, y_pred, labels=labels))
            print(confusion_matrix(y_true, y_pred))
            fold_statistics = self.cv_evaluation_fold(y_pred, y_true, labels=labels)

            fold += 1

        y_pred_val, y_true_val = self.predict(cv_model, x_val, y_val, labels)
        y_true_val = [str(lab) for lab in y_true_val]
        print("--------------------------- Results ------------------------------------")
        print(classification_report(y_true_val, y_pred_val, labels=labels))
        print(confusion_matrix(y_true_val, y_pred_val))
        if self.test:
            print(len(X_data_test))
            pred = model.predict(X_data_test)
            y_pred_ind = np.argmax(pred, axis=1)
            pred_labels = [labels[i] for i in y_pred_ind]
            dataset = pd.read_csv("data/test/test.tsv", sep='\t')
            tweets_id = dataset['tweet_id'].tolist()
            tweets = dataset['tweet'].tolist()
            print(len(tweets_id))
            print(len(tweets))
            print(len(pred_labels))
            d = {'tweet_id':tweets_id, 'tweets':tweets, 'Class': pred_labels}
            df = pd.DataFrame(data=d)
            df.to_csv('weights1to10_glovetwitter50_CV_TEST.tsv', sep='\t')

    def train_test(self):
        # train - test split
        filter_length = 32
        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen))
        model.add(Conv1D(filter_length, 1, activation='relu'))
        model.add(MaxPool1D(5))
        model.add(Conv1D(filter_length, 1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(x_train_data, y_train_data,
                            epochs=20,
                            batch_size=512)

        y_pred_val, y_true_val = evaluate.predict(model, x_val, y_val, labels)
        y_true_val = [str(lab) for lab in y_true_val]
        # y_pred = np.array(model.predict(x_val))

        print(classification_report(y_true_val, y_pred_val, target_names=labels))
        print(len(X_data_test))
        pred = model.predict(X_data_test)
        y_pred_ind = np.argmax(pred, axis=1)
        pred_labels = [labels[i] for i in y_pred_ind]
        dataset = pd.read_csv("data/test/test.tsv", sep='\t')
        tweets_id = dataset['tweet_id'].tolist()
        tweets = dataset['tweet'].tolist()
        print(len(tweets_id))
        print(len(tweets))
        print(len(pred_labels))
        d = {'tweet_id':tweets_id, 'tweets':tweets, 'Class': pred_labels}
        df = pd.DataFrame(data=d)
        df.to_csv('weights1to10_glovetwitter50_traintest_TEST.tsv', sep='\t')



model = Model("../data/train/tweets_none", "../data/train/labels_none", "../data/validation/tweets_val_none", "../data/validation/labels_val_none", 5000, 300)
makembedding = MakeEmbedding(model.word_index, "../embeddings/twitter50.txt", 50, 5000, char=True)
cnn = CNN(model.X_data, model.binary_Y, makembedding.embedding_matrix, model.X_data_val, model.binary_Y_val, model.labels, 50, 300, 5000, 32, True, True, makembedding.char_embedding_matrix)
