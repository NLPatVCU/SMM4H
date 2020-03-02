from sklearn.model_selection import train_test_split

with open("labels") as label_file_processed:
    labels = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()

with open("tweets") as tweets_file_processed:
    tweets = [tweet.rstrip() for tweet in tweets_file_processed]
    tweets_file_processed.close()


X_train, X_test, y_train, y_test = train_test_split(tweets, labels, stratify=labels,test_size=0.20)

with open("tweets_split_80", "w") as tweets_file:
    for sentence in X_train:
        tweets_file.write(sentence +"\n")
    tweets_file.close()

with open("labels_split_80", "w") as labels_file:
    for label in y_train:
        labels_file.write("%i\n" % label)
    labels_file.close()

with open("tweets_split_20", "w") as tweets_file:
    for sentence in X_test:
        tweets_file.write(sentence +"\n")
    tweets_file.close()

with open("labels_split_20", "w") as labels_file:
    for label in y_test:
        labels_file.write("%i\n" % label)
    labels_file.close()
