from model import Model
from embedding import MakeEmbedding
from cnn import CNN

model = Model("../data/train/tweets_none", "../data/train/labels_none", "../data/validation/tweets_val_none", "../data/validation/labels_val_none", 5000, 300)
makembedding = MakeEmbedding(model.word_index, "../embeddings/twitter50.txt", 50, 5000)
cnn = CNN(model.X_data, model.binary_Y, makembedding.embedding_matrix, model.X_data_val, model.binary_Y_val, model.labels, 50, 300, 5000, 32)
