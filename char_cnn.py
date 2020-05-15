import sys
sys.path.append("../Sam/RelEx/relex")

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
from RelEx_NN.model import evaluate
import numpy as np
from keras_wc_embd import get_dicts_generator, get_embedding_layer, get_embedding_weights_from_file
import tensorflow as tf
import chars2vec

from tensorflow import set_random_seed
set_random_seed(42)

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

def fit_Model(model, x_train, y_train):
    """
    fit the defined model to train on the data
    :param model: trained model
    :param x_train: training data
    :param y_train: training labels
    :return:
    """
    history = model.fit(x_train, y_train, epochs=20,
                        batch_size=512)
    loss = history.history['loss']
    acc = history.history['accuracy']

    return model, loss, acc


# getting data from CSVs
x_data_val = read_from_file("data/validation/tweets_val")
y_data_val = read_from_file("data/validation/labels_val")
train_data = read_from_file("data/train/tweets_none")
train_labels = read_from_file("data/train/labels_none")

df_data = pd.DataFrame(train_data, columns=['tweet'])
df_label = pd.DataFrame(train_labels, columns=['label'])
df_data.reset_index(drop=True, inplace=True)
df_label.reset_index(drop=True, inplace=True)
df = pd.concat((df_data, df_label), axis=1)

df_data_val = pd.DataFrame(x_data_val, columns=['tweet'])
df_label_val = pd.DataFrame(y_data_val, columns=['label'])
df_data_val.reset_index(drop=True, inplace=True)
df_label_val.reset_index(drop=True, inplace=True)
df_val = pd.concat((df_data_val, df_label_val), axis=1)

# label processing
binarizer = LabelBinarizer()


y_train_binary = binarizer.fit_transform(df['label'])
labels = binarizer.classes_
print(labels)
num_classes = len(labels)
y_train = []
for label_arr in y_train_binary:
    for label in label_arr:
        y_train.append(label)
y_train_data = np.array(y_train)

y_test_binary = binarizer.fit_transform(df_val['label'])
y_test = []
for label_arr in y_test_binary:
    for label in label_arr:
        y_test.append(label)
y_test_data = np.array(y_test)

# text
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['tweet'])
word_index = tokenizer.word_index
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tokenizer.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tokenizer.word_index[tokenizer.oov_token] = max(char_dict.values()) + 1
# Convert string to index
train_sequences = tokenizer.texts_to_sequences(df['tweet'])
test_texts = tokenizer.texts_to_sequences(df_val['tweet'])

# Padding
x_train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
x_test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

# Convert to numpy array
x_train_data = np.array(x_train_data, dtype='float32')
x_test_data = np.array(x_test_data, dtype='float32')

embedding_weights = []
embedding_weights.append(np.zeros(len(tokenizer.word_index)))
for char, i in tokenizer.word_index.items():
    onehot = np.zeros(len(tokenizer.word_index))
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.asarray(embedding_weights)


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

    input = Input(shape=(1014,))
    embed = Embedding(len(tokenizer.word_index)+1, 69, weights=[embedding_weights], input_length=1014)(input)
    conv1 = Conv1D(32, 1, activation='relu')(embed)
    maxpool1 = MaxPool1D(5)(conv1)
    conv2 = Conv1D(32, 1, activation='relu')(maxpool1)
    dropout1 = Dropout(.5)(conv2)
    flat = Flatten()(dropout1)
    dense1 = Dense(32, activation='relu')(flat)
    dense2 = Dense(2, activation='sigmoid')(dense1)
    model = Model(outputs = dense2, inputs=input)

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    cv_model, loss, acc = fit_Model(model, x_train, y_train)
    y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)
    y_true = [str(lab) for lab in y_true]
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    print("--------------------------- Results ------------------------------------")
    print(classification_report(y_true, y_pred, labels=['0', '1']))
    print(confusion_matrix(y_true, y_pred))
    fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels=labels)
    fold += 1

y_pred_val, y_true_val = evaluate.predict(cv_model, x_test_data, y_test_data, labels)
y_true_val = [str(lab) for lab in y_true_val]
print("--------------------------- Results ------------------------------------")
print(classification_report(y_true_val, y_pred_val, labels=labels))
print(confusion_matrix(y_true_val, y_pred_val))
