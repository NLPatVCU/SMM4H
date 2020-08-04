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

def read_embeddings_from_file(path):
    """
    Function to read external embedding files to build an index mapping words (as strings)
    to their vector representation (as number vectors).
    :return dictionary: word vectors
    """
    print("Reading external embedding file ......")
    if not os.path.isfile(path):
        raise FileNotFoundError("Not a valid file path")

    embeddings_index = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def read_embeddings_from_file_error_handling(path):
    """
    Function to read external embedding files to build an index mapping words (as strings)
    to their vector representation (as number vectors). Works exactly the same
    as def read_embeddings_from_file except has addional error handling for some
    embeddings that are tricky to read.
    :return dictionary: word vectors
    """
    print("Reading external embedding file ......")
    if not os.path.isfile(path):
        raise FileNotFoundError("Not a valid file path")

    embeddings_index = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
            # for use with twitter word embedding
            vector = []
            error_count = 0
            for val in values[1:]:
                if val != "\n":
                    try:
                        val_float = float(val)
                        vector.append(val_float)
                    except ValueError:
                        print("error")
            coefs = np.asarray(vector)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


class CNN:
    def __init__(self, Xdata_train, Ydata_train, Xdata_val, Ydata_val, embedding, dim, maxlen, maxwords, filters, cross_val=True, test=False, data_test=None):
        self.Xdata_train = read_from_file(Xdata_train)
        self.Ydata_train = read_from_file(Ydata_train)
        self.Xdata_val = read_from_file(Xdata_val)
        self.Ydata_val = read_from_file(Ydata_val)
        self.embedding = read_embeddings_from_file(embedding)
        self.dim = dim
        self.maxlen = maxlen
        self.maxwords = maxwords
        self.cross_val = cross_val
        self.test = test
        self.filter_length = filters
        if data_test != None:
            self.data_test = read_from_file(data_test)

        X_data, binary_Y, word_index, labels = self.process_train()
        X_data_val, binary_Y_val = self.process_val()
        if self.test:
            X_data_test = self.process_test()
        embedding_matrix = self.init_embedding(word_index)
        if self.cross_val:
            self.cv(X_data, binary_Y, embedding_matrix, X_data_val, binary_Y_val, labels)

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
        # y_true_ind = np.argmax(y_test, axis=1)
        y_pred = [encoder_classes[i] for i in y_pred_ind]
        # y_true = [encoder_classes[i] for i in y_true_ind]
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
        :return:
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

        # table_data = [[label,
                       # format(fold_statistics[label]['precision'], ".3f"),
                       # format(fold_statistics[label]['recall'], ".3f"),
                       # format(fold_statistics[label]['f1'], ".3f")]
                      # for label in labels + ['system']]

        # print(tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1'],
                       # tablefmt='orgtbl'))
        return fold_statistics

    def get_features(self, text_series, tokenizer):
        """
        transforms text data to feature_vectors that can be used in the ml model.
        tokenizer must be available.
        """
        sequences = tokenizer.texts_to_sequences(text_series)
        return pad_sequences(sequences, maxlen=self.maxlen)


    def prediction_to_label(self, prediction):
        """
        Turns the prediction into a label.
        """
        tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
        return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))

    def fit_Model(self, model, x_train, y_train):
        """
        fit the defined model to train on the data
        :param model: trained model
        :param x_train: training data
        :param y_train: training labels
        :return:
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

    def init_embedding(self, word_index):
        embeddings_index = self.embedding
        embedding_dim = self.dim
        maxlen = self.maxlen
        max_words = self.maxwords
        embedding_matrix = np.zeros((max_words, embedding_dim))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def process_train(self):
        # for train data
        df_data = pd.DataFrame(self.Xdata_train, columns=['tweet'])
        df_label = pd.DataFrame(self.Ydata_train, columns=['label'])
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


    def process_val(self):
        # for validation data
        df_data_val = pd.DataFrame(self.Xdata_val, columns=['tweet'])
        df_label_val = pd.DataFrame(self.Ydata_val, columns=['label'])
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
        df_test = pd.DataFrame(self.data_test, columns=['tweet'])

        tokenizer_test = Tokenizer(num_words=self.maxwords, lower=True)
        print(df_test['tweet'])
        tokenizer_test.fit_on_texts(df_test['tweet'])
        X_data_test = self.get_features(df_test['tweet'], tokenizer_test)
        return X_data_test

    def cv(self, x_train_data, y_train_data, embedding_matrix, x_val, y_val, labels, X_data_test=None):
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
            """
            if char:
                chars = Sequential()
                chars.add(Embedding(36745, 50, weights=[char_embeddings], input_length=maxlen))
                # model.add(Embedding(36745, 50, weights=[char_embeddings], input_length=maxlen))
                words = Sequential()
                words.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen))

                model = Sequential()
                model.add(Concatenate([chars, words], axis=-1))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(MaxPool1D(5))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(2, activation='sigmoid'))
                model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            """


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

    def train_test():
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




o1 = CNN("../data/train/tweets_none", "../data/train/labels_none", "../data/validation/tweets_val_none", "../data/validation/labels_val_none", "../embeddings/twitter50.txt", 50, 300, 5000, 32)
