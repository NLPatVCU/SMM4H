from sklearn.utils import shuffle
from random import seed
from random import randint
import sys
import pandas
import numpy as np


class Unbalanced:
    def __init__(self, X, Y, unbalanced, multiplier=None, ratio1=None, ratio2=None, ratio1_label=None, ratio2_label=None):
        """
        Oversamples or desamples data. Only works with 2 classes.

        :param X:  X data
        :type X: List
        :param Y: Y data
        :type Y: List
        :param unbalanced: flag for desample, oversample or none. Four options: desample, oversample, weights, none.
        :type unbalanced: Str
        :param multiplier: number to duplicate by for oversampling. 1 more than duplications desired.
        :type multiplier: Int
        :param ratio1: ratio desired for first label
        :type ratio1: Int
        :param ratio2: ratio desired for second label
        :type ratio2: Int
        :param ratio1_label: first label
        :type ratio1_label: Int
        :param ratio2_label: second label
        :type ratio2_label: Int
        """
        self.unbalanced = unbalanced

        if self.unbalanced == "oversample":
            self.multiplier = multiplier
            self.X, self.Y = self.oversample(X, Y)
        elif self.unbalanced == "desample":
            self.ratio1 = ratio1
            self.ratio2 = ratio2
            self.ratio1_label = ratio1_label
            self.ratio2_label = ratio2_label
            self.X, self.Y = self.desample(X, Y)
        else:
            self.X = X
            self.Y = Y

    def desample(self, sentences, labels):
        """
        Desamples.

        :param labels: y data
        :type labels: List
        :param sentences: x data
        :type sentences: List

        :return: desampled sentences, desampled labels
        :rtype: List
        """
        # shuffles labels
        labels_shuffled, sentences_shuffled = shuffle(labels, sentences)
        # gets current ratio for label2
        current_ratio2 = labels.count(self.ratio2_label)/labels.count(self.ratio1_label)

        labels_shuffled_ratio2 = []
        sentences_shuffled_ratio2 = []
        labels_shuffled_ratio1 = []
        sentences_shuffled_ratio1 = []

        # divides data into lists based on labels
        for s in sentences_shuffled:
            if labels_shuffled[sentences_shuffled.index(s)] == 1:
                labels_shuffled_ratio1.append(1)
                sentences_shuffled_ratio1.append(s)
            else:
                labels_shuffled_ratio2.append(0)
                sentences_shuffled_ratio2.append(s)

        # simplfies ratios
        if self.ratio1 != 1:
            self.ratio1 = 1
            self.ratio2 = self.ratio2/self.ratio1

        # calculates number to remove
        num_to_remove = int(((current_ratio2-self.ratio2)/current_ratio2)*labels.count(0))

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

    def oversample(self, labels, sentences):
        """
        Oversamples.

        :param labels: y data
        :type labels: List
        :param sentences: x data
        :type sentences: List

        :return: oversampled sentences, oversampled labels
        :rtype: List
        """

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
        for x in range(0, self.multiplier):
            new_labels_1 = new_labels_1 + labels_shuffled_1
            new_sentences_1 = new_sentences_1 + sentences_shuffled_1

        oversampled_labels = new_labels_1 + labels_shuffled_0
        oversampled_sentences = new_sentences_1 + sentences_shuffled_0

        oversampled_labels_shuffled, oversampled_sentences_shuffled = shuffle(oversampled_labels, oversampled_sentences)

        return oversampled_sentences_shuffled, oversampled_labels_shuffled
