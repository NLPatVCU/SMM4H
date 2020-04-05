from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('../embeddings/word2vectwitter.bin', binary=True)
model.save_word2vec_format('../embeddings/word2vectwitter.txt', binary=False)
