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

cv = True
write_results_file = False
char = False
char_2 = True


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
            # for use with twitter word embedding
            if sys.argv[1] == "word2vectwitter" or sys.argv[1] == "fasttext":
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
            else:
                coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)


def prediction_to_label(prediction):
    tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))

def fit_Model(model, x_train, y_train):
    """
    fit the defined model to train on the data
    :param model: trained model
    :param x_train: training data
    :param y_train: training labels
    :return:
    """
    if sys.argv[3] == "weights":
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

def output_to_file(true, pred, target):
    """
    Function to create .txt file and csv file of classification report
    """
    report = classification_report(true, pred, target_names=target)
    report_dict = classification_report(true, pred, target_names=target, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    #writes .txt file with results
    txt_file = open(output_txt_path, 'a')
    txt_file.write(report)
    txt_file.close()

    # writes csv file
    csv_report = df_report.to_csv()
    csv_file = open(output_csv_path, 'a')
    csv_file.write(csv_report)
    csv_file.close()

if sys.argv[1] == "glove100twitter":
    embedding_path = "../embeddings/glove.twitter.27B.100d.txt"
elif sys.argv[1] == "glove200twitter":
    embedding_path = "../embeddings/glove.twitter.27B.200d.txt"
elif sys.argv[1] == "glove50twitter":
    embedding_path = "../embeddings/glove.twitter.27B.50d.txt"
elif sys.argv[1] == "glove50":
    embedding_path = "../embeddings/glove.6B.50d.txt"
elif sys.argv[1] == "glove100":
    embedding_path = "../embeddings/glove.6B.100d.txt"
elif sys.argv[1] == "glove200":
    embedding_path = "../embeddings/glove.6B.200d.txt"
elif sys.argv[1] == "glove300":
    embedding_path = "../embeddings/glove.6B.300d.txt"
elif sys.argv[1] == "mimic200":
    embedding_path = "../embeddings/mimic3_d200.bin"
elif sys.argv[1] == "mimic300":
    embedding_path = "../embeddings/mimic3_d300.txt"
elif sys.argv[1] == "mimic400":
    embedding_path = "../embeddings/mimic3_d400.txt"
elif sys.argv[1] == "wikipubmed":
    embedding_path = "../embeddings/wikipubmed.bin"
elif sys.argv[1] == "wiki":
    embedding_path = "../embeddings/wikivectors.200.bin"
elif sys.argv[1] == "fasttext":
    embedding_path ="../embeddings/fastText_pretrained_twitter.vec"
elif sys.argv[1] == "word2vectwitter":
    embedding_path ="../embeddings/word2vectwitter_text.bin"

embeddings_index = read_embeddings_from_file(embedding_path)
embedding_dim = int(sys.argv[2])
maxlen = 300
max_words = 5000
embedding_matrix = np.zeros((max_words, embedding_dim))

if sys.argv[3] == "desample":
    x_data_val = read_from_file("data/validation/tweets_val")
    y_data_val = read_from_file("data/validation/labels_val")
    train_data = read_from_file("data/train/tweets_desample")
    train_labels = read_from_file("data/train/labels_desample")
elif sys.argv[3] == "none" or sys.argv[3] == "weights":
    x_data_val = read_from_file("data/validation/tweets_val")
    y_data_val = read_from_file("data/validation/labels_val")
    train_data = read_from_file("data/train/tweets_none")
    train_labels = read_from_file("data/train/labels_none")
elif sys.argv[3] == "oversample":
    x_data_val = read_from_file("data/validation/tweets_val")
    y_data_val = read_from_file("data/validation/labels_val")
    train_data = read_from_file("data/train/tweets_oversample")
    train_labels = read_from_file("data/train/labels_oversample")

print(len(y_data_val))
print(len(x_data_val))

# for train data
df_data = pd.DataFrame(train_data, columns=['tweet'])
df_label = pd.DataFrame(train_labels, columns=['label'])
df_data.reset_index(drop=True, inplace=True)
df_label.reset_index(drop=True, inplace=True)
df = pd.concat((df_data, df_label), axis=1)

binarizer = LabelBinarizer()
binarizer.fit(df['label'])
labels = binarizer.classes_
print(labels)
num_classes = len(labels)



tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['tweet'])
X_data = get_features(df['tweet'])
word_index = tokenizer.word_index

if char:
    words = []
    for tuple in word_index.items():
        words.append(tuple[0])
    c2v_model = chars2vec.load_model('eng_50')
    char_embeddings = c2v_model.vectorize_words(words)

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
else:
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector



binary_y = binarizer.transform(df['label'])
binary_Y = []
for label_arr in binary_y:
    for label in label_arr:
        binary_Y.append(label)
binary_Y = np.array(binary_Y)

# for validation data
df_data_val = pd.DataFrame(x_data_val, columns=['tweet'])
df_label_val = pd.DataFrame(y_data_val, columns=['label'])
df_data_val.reset_index(drop=True, inplace=True)
df_label_val.reset_index(drop=True, inplace=True)
df_val = pd.concat((df_data_val, df_label_val), axis=1)


binarizer_val = LabelBinarizer()
binarizer_val.fit(df_val['label'].astype(str))
labels_val = binarizer_val.classes_

print(df_val['tweet'])
tokenizer.fit_on_texts(df_val['tweet'])
X_data_val = get_features(df_val['tweet'])

binary_y_val = binarizer_val.transform(df_val['label'].astype(str))
binary_Y_val = []
for label_arr in binary_y_val:
    for label in label_arr:
        binary_Y_val.append(label)
binary_Y_val = np.array(binary_Y_val)

if char_2:
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

if sys.argv[4] == "SMOTE":
    sm = SMOTE(random_state = 2)
    x_train_data, y_train_data = sm.fit_sample(X_data, binary_Y.ravel())
    x_val = X_data_val
    y_val = binary_Y_val
elif sys.argv[4] == "ADASYN":
    a = ADASYN(random_state = 2)
    x_train_data, y_train_data = a.fit_sample(X_data, binary_Y.ravel())
    x_val = X_data_val
    y_val = binary_Y_val
else:
    x_train_data = X_data
    y_train_data = binary_Y
    x_val = X_data_val
    y_val = binary_Y_val

print(binary_Y_val)
print(x_val)

if cv:
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
        filter_length = 32
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
        else:
            if char_2:
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
                model.summary()

            else:
                model = Sequential()
                model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=1014))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(MaxPool1D(5))
                model.add(Conv1D(filter_length, 1, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(2, activation='sigmoid'))

            model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cv_model, loss, acc = fit_Model(model, x_train, y_train)
        y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)
        y_true = [str(lab) for lab in y_true]
        originalclass.extend(y_true)
        predictedclass.extend(y_pred)
        print("--------------------------- Results ------------------------------------")
        print(classification_report(y_true, y_pred, labels=labels))
        print(confusion_matrix(y_true, y_pred))
        fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels=labels)

        # evaluation_statistics[fold] = fold_statistics


        # print(classification_report(np_true, np_pred,target_names=labels))
        fold += 1
    # print(classification_report(np.array(originalclass), np.array(predictedclass), labels=labels))
    # print(classification_report(np.array(originalclass), np.array(predictedclass),target_names=labels))
    y_pred_val, y_true_val = evaluate.predict(cv_model, x_val, y_val, labels)
    y_true_val = [str(lab) for lab in y_true_val]
    print("--------------------------- Results ------------------------------------")
    print(classification_report(y_true_val, y_pred_val, labels=labels))
    print(confusion_matrix(y_true_val, y_pred_val))
    if write_results_file:
        output_to_file(np_true, np_pred, labels)

"""
else:
    # train - test split

    x_train, x_test, y_train, y_test = train_test_split(X_data, binary_Y, test_size=0.2, random_state=9000)
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
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=512,
                        validation_split=0.1)
    metrics = model.evaluate(x_test, y_test)
    print("{}: {}".format(model.metrics_names[0], metrics[0]))
    print("{}: {}".format(model.metrics_names[1], metrics[1]))
    y_pred = np.array(model.predict(x_test))


    print(classification_report(y_test, y_pred, target_names=labels))
    if write_results_file:
        output_to_file(np_true, np_pred, labels)
"""
