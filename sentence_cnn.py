import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from keras.layers import *
from keras.models import *


class Sentence_CNN:

    def __init__(self, embedding_path, class_weight = False, cross_validation = True, epochs=20, batch_size=512, filters=32, filter_conv=1, filter_maxPool=5,
                 activation='relu', output_activation='sigmoid', drop_out=0.5, loss='sparse_categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy']):

        self.data_model = model
        self.embedding = read_embeddings_from_file(embedding_path)
        self.cv = cross_validation
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.filter_conv = filter_conv
        self.filter_maxPool = filter_maxPool
        self.activation = activation
        self.output_activation = output_activation
        self.drop_out = drop_out
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        if self.cv:
            self.cross_validate()
        else:
            self.test()

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
        with open(path) as f:
            next(f)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
        return embeddings_index

    def fit_Model(self, model, x_train, y_train):
        """
        fit the defined model to train on the data
        :param model: trained model
        :param x_train: training data
        :param y_train: training labels
        :return:
        """
        if self.class_weight:
            class_weights= {0:1, 1:10}
            history = model.fit(x_train, y_train, epochs=self.epochs,
                                batch_size=self.batch_size, class_weight=class_weights)
        else:
            history = model.fit(x_train, y_train, epochs=self.epochs,
                                batch_size=self.batch_size)
        loss = history.history['loss']
        acc = history.history['acc']

        return model, loss, acc


    def predict(model, x_test, y_test, encoder_classes):
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


    def cv_evaluation_fold(y_pred, y_true, labels):
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

        return fold_statistics


    def cv(self, folds=5):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        skf.get_n_splits(x_train_data, y_train_data)

        fold = 1

        for train_index, test_index in skf.split(x_train_data, y_train_data):

            x_train, x_test = x_train_data[train_index], x_train_data[test_index]
            y_train, y_test = y_train_data[train_index], y_train_data[test_index]
            print("Training Fold %i" % fold)

            model = Sequential()
            model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen))
            model.add(Conv1D(self.filter_length, self.filter_conv, activation=self.activation))
            model.add(MaxPool1D(self.filter_maxPool))
            model.add(Conv1D(self.filters, self.filter_conv, activation=self.activation))
            model.add(Dropout(self.drop_out))
            model.add(Flatten())
            model.add(Dense(self.filters, activation=self.activation))
            model.add(Dense(2, activation=self.output_activation))

            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

            cv_model, loss, acc = fit_Model(model, x_train, y_train)
            y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)
            y_true = [str(lab) for lab in y_true]
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
        if write_results_file:
            output_to_file(np_true, np_pred, labels)

    def test(self):
        print("This hasn't been finished yet.")
