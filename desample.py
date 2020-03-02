from sklearn.utils import shuffle
from random import seed
from random import randint

def desample(labels, sentences, ratio1, ratio2, ratio1_label, ratio2_label):
    """
    :param labels: array of labels
    :param sentences: array of sentences
    :param ratio1: goal ratio of the label declared in ratio1_label
    :param ratio2: goal ratio of the label declared in ratio2_label
    :param ratio1_label: label that corresponds to ratio1
    :param ratio2_label: label that corresponds to ratio2

    :return: desampled sentences, desampled labels

    """
    # shuffles labels
    labels_shuffled, sentences_shuffled = shuffle(labels, sentences)

    # gets current ratio for label2
    current_ratio2 = labels.count(ratio2_label)/labels.count(ratio1_label)

    labels_shuffled_ratio2 = []
    sentences_shuffled_ratio2 = []
    labels_shuffled_ratio1 = []
    sentences_shuffled_ratio1 = []

    # divides data into lists based on labels
    for i in sentences_shuffled:
        if labels_shuffled[sentences_shuffled.index(i)] == ratio2_label:
            sentences_shuffled_ratio2.append(i)
            labels_shuffled_ratio2.append(0)
        else:
            sentences_shuffled_ratio1.append(i)
            labels_shuffled_ratio1.append(1)

    # simplfies ratios
    if ratio1 != 1:
        ratio1 = 1
        ratio2 = ratio2/ratio1

    # calculates number to remove
    num_to_remove = int(round(((current_ratio2-ratio2)/current_ratio2)*labels.count(ratio2_label)))
    print(num_to_remove)

    # removes correct number
    for x in range(0, num_to_remove):
        seed()
        index = randint(0, (len(labels_shuffled_ratio2) -1))
        labels_shuffled_ratio2.pop(index)
        sentences_shuffled_ratio2.pop(index)

    # recombines lists
    labels_desampled = labels_shuffled_ratio2 + labels_shuffled_ratio1
    sentences_desampled = sentences_shuffled_ratio2 + sentences_shuffled_ratio1

    # shuffles lists
    sentences_desampled_shuffled, labels_desampled_shuffled = shuffle(sentences_desampled, labels_desampled)

    print(labels_desampled_shuffled.count(0))
    print(labels_desampled_shuffled.count(1))

    return sentences_desampled_shuffled, labels_desampled_shuffled

with open("labels") as label_file_processed:
    labels = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()

with open("tweets") as tweets_file_processed:
    tweets = [tweet.rstrip() for tweet in tweets_file_processed]
    tweets_file_processed.close()

sentences, labels = desample(labels, tweets, 1, 8, 1, 0)


with open("tweets_desample", "w") as tweets_file:
    for sentence in sentences:
        tweets_file.write(sentence +"\n")
    tweets_file.close()

with open("labels_desample", "w") as labels_file:
    for label in labels:
        labels_file.write("%i\n" % label)
    labels_file.close()
