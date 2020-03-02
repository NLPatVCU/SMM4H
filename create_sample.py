from random import sample
from sklearn.utils import shuffle
import pandas as pd

# dataset = pd.read_csv("task2_en_training.tsv", sep='\t')
# labels = dataset['class'].tolist()
# tweets = dataset['tweet'].tolist()

labels = []
tweets = []

with open("labels_val") as label_file_processed:
    labels = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()

with open("tweets_val") as tweets_file_processed:
    tweets = [tweet.rstrip() for tweet in tweets_file_processed]
    tweets_file_processed.close()

label_1_sample_amount = 192
label_2_sample_amount = 2304

label_1 = 1
label_2 = 0

label_1_tweets = []
label_1_labels = []

label_2_tweets = []
label_2_labels = []

for label, tweet in zip(labels, tweets):
    if label == label_1:
        label_1_labels.append(label)
        label_1_tweets.append(tweet)

    if label == label_2:
        label_2_labels.append(label)
        label_2_tweets.append(tweet)



label_1_labels_sample, label_1_tweets_sample =zip(*sample(list(zip(label_1_labels, label_1_tweets)), label_1_sample_amount))
label_2_labels_sample, label_2_tweets_sample =zip(*sample(list(zip(label_2_labels, label_2_tweets)), label_2_sample_amount))

labels_sample = label_1_labels_sample + label_2_labels_sample
tweets_sample = label_1_tweets_sample + label_2_tweets_sample

labels_sample_shuffled, tweets_sample_shuffled = shuffle(labels_sample, tweets_sample)


with open("tweets_val_sample", "w") as tweets_file:
    for tweet in tweets_sample_shuffled:
        tweets_file.write(tweet +"\n")
    tweets_file.close()

with open("labels_val_sample", "w") as labels_file:
    for label in labels_sample_shuffled:
        labels_file.write("%i\n" % label)
    labels_file.close()
