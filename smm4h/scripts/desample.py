from sklearn.utils import shuffle
from random import seed
from random import randint
import sys
import pandas

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
    print(labels.count(1))
    print(labels.count(0))
    print("\n")
    # shuffles labels
    labels_shuffled, sentences_shuffled = shuffle(labels, sentences)
    # gets current ratio for label2
    current_ratio2 = labels.count(ratio2_label)/labels.count(ratio1_label)

    labels_shuffled_ratio2 = []
    sentences_shuffled_ratio2 = []
    labels_shuffled_ratio1 = []
    sentences_shuffled_ratio1 = []

    print(labels_shuffled.count(1))
    print(labels_shuffled.count(0))
    print("\n")

    # divides data into lists based on labels
    for s in sentences_shuffled:
        if labels_shuffled[sentences_shuffled.index(s)] == 1:
            labels_shuffled_ratio1.append(1)
            sentences_shuffled_ratio1.append(s)
        else:
            labels_shuffled_ratio2.append(0)
            sentences_shuffled_ratio2.append(s)

    print(len(labels_shuffled_ratio1))
    print(len(labels_shuffled_ratio2))

    print(len(sentences_shuffled_ratio1))
    print(len(sentences_shuffled_ratio2))
    print("\n")

    # simplfies ratios
    if ratio1 != 1:
        ratio1 = 1
        ratio2 = ratio2/ratio1

    # calculates number to remove
    num_to_remove = int(((current_ratio2-ratio2)/current_ratio2)*labels.count(0))
    print(num_to_remove)
    # removes correct number
    for x in range(0, num_to_remove):
        seed()
        index = randint(-1, len(labels_shuffled_ratio2)-1)
        labels_shuffled_ratio2.pop(index)
        sentences_shuffled_ratio2.pop(index)

    # recombines lists
    labels_desampled = labels_shuffled_ratio2 + labels_shuffled_ratio1
    sentences_desampled = sentences_shuffled_ratio2 + sentences_shuffled_ratio1

    # shuffles lists
    sentences_desampled_shuffled, labels_desampled_shuffled = shuffle(sentences_desampled, labels_desampled)

    return sentences_desampled_shuffled, labels_desampled_shuffled