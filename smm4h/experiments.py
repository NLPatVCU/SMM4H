from model import Model
from embedding import MakeEmbedding
from cnn import CNN
from preprocessing import Preprocessing
from unbalanced import Unbalanced
from file import File
import sys
sys.path.append(".")

tweets_train = Preprocessing("../../data/train/task2_en_training.tsv", False, "tweet", "class").tweets
# labels_train = File().read_from_file("../../data/train/labels")
labels_train = Preprocessing("../../data/train/task2_en_training.tsv", False, "tweet", "class").labels
tweets_val = Preprocessing("../../data/validation/task2_en_validation.tsv", False, "tweet", "class").tweets
# labels_val =  File().read_from_file("../../data/validation/labels_val")
labels_val = Preprocessing("../../data/validation/task2_en_validation.tsv", False, "tweet", "class").labels

unbalanced = Unbalanced(tweets_train, labels_train, "desample", None, 1, 4, "1", "0")
oversampleX = unbalanced.X
oversampleY = unbalanced.Y
tweets_test = Preprocessing("../../data/test/test.tsv", True, "tweet", None).tweets


model = Model(tweets_train, labels_train, tweets_val, labels_val, 5000, 300, True, tweets_test)
makembedding = MakeEmbedding(model.word_index, "../../embeddings/glove.twitter.27B.50d.txt", 50, model.maxwords)
cnn = CNN(model.X_data, model.binary_Y, makembedding.embedding_matrix, model.X_data_val,
        model.binary_Y_val, model.labels, makembedding.dim, model.maxlen, model.maxwords,
        32, False, True, [1, 10], True, model.X_data_test,"../../data/test/test.tsv",
         "../../data/test/test_new.tsv", "tweet", "class", "tweet_id" )
