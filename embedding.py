import sys
import io
import os
import pandas as pd
import sys
import keras
import numpy as np

class MakeEmbedding:
    def __init__(self, word_index, embedding, dim, maxwords, error_handling=False):
        """
        Creates embedding for CNN

        :param word_index: word_index from model file
        :param embedding: path to embedding file
        :param dim: dimension of the embedding
        :param maxwords: maximum number of words to use
        :param error_handling: flag for embedding read function that handles errors
        """

        self.word_index = word_index
        self.dim = dim
        self.maxwords = maxwords

        self.error_handling = error_handling

        if self.error_handling:
            self.embedding = self.read_embeddings_from_file_error_handling(embedding)
        else:
            self.embedding = self.read_embeddings_from_file(embedding)

        self.embedding_matrix = self.init_embedding()

    def read_embeddings_from_file(self, path):
        """
        Function to read external embedding files to build an index mapping words (as strings)
        to their vector representation (as number vectors).

        :param path: path to emebedding
        :return dictionary: word vectors
        """
        # rasies error if there's an invalid file path
        print("Reading external embedding file ......")
        if not os.path.isfile(path):
            raise FileNotFoundError("Not a valid file path")

        # creates embedding index
        embeddings_index = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            next(f)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def read_embeddings_from_file_error_handling(self, path):
        """
        Function to read external embedding files to build an index mapping words (as strings)
        to their vector representation (as number vectors). Works exactly the same
        as def read_embeddings_from_file except has addional error handling for some
        embeddings that are tricky to read.

        :param path: path to emebedding
        :return dictionary: word vectors
        """

        # rasies error if there's an invalid file path
        print("Reading external embedding file ......")
        if not os.path.isfile(path):
            raise FileNotFoundError("Not a valid file path")

        # creates embedding index
        embeddings_index = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            next(f)
            for line in f:
                values = line.split()
                word = values[0]
                vector = []
                # this code processes error that occur when reading in an embedding
                for val in values[1:]:
                    if val != "\n":
                        try:
                            val_float = float(val)
                            vector.append(val_float)
                        except ValueError:
                            print("There's beem an error")
                coefs = np.asarray(vector)
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def init_embedding(self):
        """
        function that creates embedding matrix from word_index.

        :param word_index: word index of X data
        :return: embedding matrix
        """

        embedding_matrix = np.zeros((self.maxwords, self.dim))

        for word, i in self.word_index.items():
            embedding_vector = self.embedding.get(word)
            if i < self.maxwords:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix
