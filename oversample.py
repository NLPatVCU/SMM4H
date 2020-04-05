from sklearn.utils import shuffle
from random import seed
from random import randint
import sys
import pandas

def oversample(labels, sentences, multiplier):
    labels_shuffled, sentences_shuffled = shuffle(labels, sentences)

    labels_shuffled_0 = []
    sentences_shuffled_0 = []
    labels_shuffled_1 = []
    sentences_shuffled_1 = []

    # divides data into lists based on labels
    for s in sentences_shuffled:
        if labels_shuffled[sentences_shuffled.index(s)] == 1:
            labels_shuffled_1.append(1)
            sentences_shuffled_1.append(s)
        else:
            labels_shuffled_0.append(0)
            sentences_shuffled_0.append(s)

    new_labels_1 = labels_shuffled_1
    new_sentences_1 = sentences_shuffled_1
    for x in range(0, multiplier):
        new_labels_1 = new_labels_1 + labels_shuffled_1
        new_sentences_1 = new_sentences_1 + sentences_shuffled_1

    oversampled_labels = new_labels_1 + labels_shuffled_0
    oversampled_sentences = new_sentences_1 + sentences_shuffled_0

    oversampled_labels_shuffled, oversampled_sentences_shuffled = shuffle(oversampled_labels, oversampled_sentences)
    print(labels.count(1))
    print(oversampled_labels_shuffled.count(1))

    print(labels.count(0))
    print(oversampled_labels_shuffled.count(0))

    return oversampled_sentences_shuffled, oversampled_labels_shuffled


with open("./data/train/labels") as label_file_processed:
    labels = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()

with open("./data/train/tweets") as tweets_file_processed:
    tweets = [tweet.rstrip() for tweet in tweets_file_processed]
    tweets_file_processed.close()


tweets_oversampled, labels_oversampled = oversample(labels, tweets, int(sys.argv[1]))

with open("./data/train/tweets_oversample", "w") as tweets_file:
    for sentence in tweets_oversampled:
        tweets_file.write(sentence +"\n")
    tweets_file.close()

with open("./data/train/labels_oversample", "w") as labels_file:
    for label in labels_oversampled:
        labels_file.write("%i\n" % label)
    labels_file.close()
