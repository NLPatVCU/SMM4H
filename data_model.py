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
from RelEx_NN.model import evaluate


class Data_Model:
    def __init___(self, data_x, data_y, val_x, val_y, SMOTE_flag=False):
        self.data_x = data_x
        self.data_y = data_y
        self.val_x = val_x
        self.val_y = val_y
        self.SMOTE_flag = SMOTE_flag
        self.x_train, self.y_train = self.create_model(data_x, data_y)
        self.x_validation, self.y_validation = self.create_model(val_x, val_y)


        def create_model(self, x, y):
            df_data = pd.DataFrame(x, columns=['tweet'])
            df_label = pd.DataFrame(y, columns=['label'])
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
            binary_y = binarizer.transform(df['label'])
            binary_Y = []
            for label_arr in binary_y:
                for label in label_arr:
                    binary_Y.append(label)
            binary_Y = np.array(binary_Y)
