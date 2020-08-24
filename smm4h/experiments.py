from model import Model
from embedding import MakeEmbedding
from cnn import CNN
from preprocessing import Preprocessing
from unbalanced import Unbalanced
from file import File
import sys
sys.path.append(".")

# put in the paths that correspond to where your data is
tweets_train = Preprocessing("../../data/train/task2_en_training.tsv", False, "tweet", "class").tweets
labels_train = Preprocessing("../../data/train/task2_en_training.tsv", False, "tweet", "class").labels
tweets_val = Preprocessing("../../data/validation/task2_en_validation.tsv", False, "tweet", "class").tweets
labels_val = Preprocessing("../../data/validation/task2_en_validation.tsv", False, "tweet", "class").labels
tweets_test = Preprocessing("../../data/test/test.tsv", True, "tweet", None).tweets

# this is how you desample
unbalanced = Unbalanced(tweets_train, labels_train, "desample", None, 1, 4, "1", "0")
# this is how you oversample
# unbalanced = Unbalanced(tweets_train, labels_train, "oversample", 4)
unbalancedX = unbalanced.X
unbalancedY = unbalanced.Y


model = Model(unbalancedX, unbalancedY, tweets_val, labels_val, 5000, 300, True, tweets_test)

# if you use desample or oversample
# model = Model(tweets_train, labels_train, tweets_val, labels_val, 5000, 300, True, tweets_test)

# put in the path of your embedding & it's dimension
makembedding = MakeEmbedding(model.word_index, "../../embeddings/glove.twitter.27B.50d.txt", 50, model.maxwords)

# for test data (using class weights)
cnn = CNN(model.X_data, model.binary_Y, makembedding.embedding_matrix, model.X_data_val,
        model.binary_Y_val, model.labels, makembedding.dim, model.maxlen, model.maxwords,
        32, False, True, [1, 10], True, model.X_data_test,"../../data/test/test.tsv",
         "../../data/test/test_new.tsv", "tweet", "class", "tweet_id"

# with out test data (and without class weights)
# cnn = CNN(model.X_data, model.binary_Y, makembedding.embedding_matrix, model.X_data_val,
        # model.binary_Y_val, model.labels, makembedding.dim, model.maxlen, model.maxwords,
        # 32, False, False, None, False )
