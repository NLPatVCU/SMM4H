  import sys
sys.path.append("../Sam/RelEx/relex")

import os
import pandas as pd
import sys
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

SMOTE_flag = False
class_weights = False
cv = True
write_results_file = False
if sys.argv[1] == "glove100twitter":
    embedding_path = "../embeddings/glove.twitter.27b.100d.txt"
if sys.argv[1] == "glove200twitter":
    embedding_path = "../embeddings/glove.twitter.27b.200d.txt"
if sys.argv[1] == "glove50":
    embedding_path = "../embeddings/glove.twitter.27b.50d.txt"
if sys.argv[1] == "glove25":
    embedding_path = "../embeddings/glove.twitter.27b.25d.txt"

results_txt_path = "/home/cora/Desktop/"
results_csv_path = "/home/cora/Desktop/"

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
    with open(path) as f:
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
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
    class_weights= {0:1, 1:10}
    history = model.fit(x_train, y_train, epochs=20,
                        batch_size=512, class_weight=class_weights)
    loss = history.history['loss']
    acc = history.history['acc']

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

embeddings_index = read_embeddings_from_file(embedding_path)
embedding_dim = int(sys.argv[2])
maxlen = 300
max_words = 10000


embeddings_index = {}
with open(embedding_path) as f:
    next(f)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()


if sys.argv[3] == "desample_val":
    x_data_val = read_from_file("tweets_desample_val")
    y_data_val = read_from_file("labels_desample_val")
    train_data = read_from_file("tweets_desample")
    train_labels = read_from_file("labels_desample")
if sys.argv[3] == "desample":
    x_data_val = read_from_file("tweets_val")
    y_data_val = read_from_file("labels_val")
    train_data = read_from_file("tweets_desample")
    train_labels = read_from_file("labels_desample")
if sys.argv[3] = "no_desample":
    x_data_val = read_from_file("tweets_val")
    y_data_val = read_from_file("labels_val")
    train_data = read_from_file("tweets")
    train_labels = read_from_file("labels")

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
embedding_matrix = np.zeros((max_words, embedding_dim))
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

tokenizer_val = Tokenizer(num_words=max_words, lower=True)
tokenizer_val.fit_on_texts(df_val['tweet'])
X_data_val = get_features(df_val['tweet'])
binary_y_val = binarizer_val.transform(df_val['label'].astype(str))
binary_Y_val = []
for label_arr in binary_y_val:
    for label in label_arr:
        binary_Y_val.append(label)
binary_Y_val = np.array(binary_Y_val)


if SMOTE_flag:
    sm = SMOTE(random_state = 2)
    x_train_data, y_train_data = sm.fit_sample(X_data, binary_Y.ravel())
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
        """
        history = model.fit(x_train, y_train, epochs=20, batch_size=512)

        np_pred = np.array(model.predict(x_test))
        np_pred[np_pred < 0.5] = 0
        np_pred[np_pred > 0.5] = 1
        np_pred = np_pred.astype(int)
        np_true = np.array(y_test)

        originalclass.extend(np_true)
        predictedclass.extend(np_pred)
        print(classification_report(np_true, np_pred,target_names=labels))
        """

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
