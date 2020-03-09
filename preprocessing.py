import emoji
import pandas as pd


class Preprocessing:
    def __init__(self, file):
        self.file = file
        dataset = pd.read_csv(file, sep='\t')
        self.tweets = dataset['tweet'].tolist()
        tweets_no_html = self.remove_html(self.tweets)
        tweets_punctuation = self.remove_punctuation(tweets_no_html)
        tweets_no_hashtag = self.replace_hashtags(tweets_punctuation)
        tweets_no_links = self.replace_links(tweets_no_hashtag)
        tweets_no_usernames = self.replace_usernames(tweets_no_links)
        tweets = self.replace_emojis(tweets_no_usernames)
        print(tweets)

        with open("./data/validation/tweets_val", "w") as tweets_file:
            for tweet in tweets:
                tweets_file.write("\n" + tweet)
        tweets_file.close()

        self.labels = dataset['class'].tolist()
        with open("./data/validation/labels_val", "w") as labels_file:
            for label in self.labels:
                labels_file.write("%i\n" % label)
        labels_file.close()

    def remove_punctuation(self, tweets):
        tweets_clean = []
        for tweet in tweets:
            tweet = tweet.replace('“', '')
            tweet = tweet.replace('"', '')
            tweets_clean.append(tweet)
        return tweets_clean


    def replace_emojis(self, tweets):
        tweets_no_emojis = []
        for tweet in tweets:
            tweet_demoji = emoji.demojize(tweet, delimiters=(' ', ' ')).replace('  ', ' ')
            tweet_no_underscore = ""
            for index, char in enumerate(tweet_demoji):
                if index > 0 and index < len(tweet_demoji) - 1:
                    if char == "_" and tweet_demoji[index-1].isalpha() and tweet_demoji[index+1].isalpha():
                        tweet_no_underscore += " "
                    else:
                        tweet_no_underscore += char
                else:
                    tweet_no_underscore += char
            tweets_no_emojis.append(tweet_no_underscore)
        return tweets_no_emojis

    def replace_usernames(self, tweets ):
        tweets_no_usernames = []
        for tweet in tweets:
            tweet_no_users = ""
            for word in tweet.split(' '):
                if word.startswith('@'):
                    tweet_no_users += " username "
                else:
                    tweet_no_users += " " + word + " "
            tweet_correct_spacing = tweet_no_users.replace('  ', ' ').strip()
            tweets_no_usernames.append(tweet_correct_spacing)
        return tweets_no_usernames

    def replace_links(self, tweets):
        tweets_no_links=[]
        for tweet in tweets:
            tweet_no_link = ""
            for word in tweet.split(' '):
                if word.startswith('http://'):
                    tweet_no_link+=" hyperlink "
                else:
                    tweet_no_link+=" " + word + " "
            tweet_correct_spacing = tweet_no_link.replace('  ', ' ').strip()
            tweets_no_links.append(tweet_correct_spacing)
        return tweets_no_links

    def replace_hashtags(self, tweets):
        tweets_no_hashtag = []
        for tweet in tweets:
            tweet_no_hashtag = ""
            for index, char in enumerate(tweet):
                if index < len(tweet) - 1:
                    if char == "#" and tweet[index+1].isalpha():
                        tweet_no_hashtag += " hastag "
                    else:
                        tweet_no_hashtag += char
                else:
                    tweet_no_hashtag += char
            tweet_correct_spacing = tweet_no_hashtag.replace('  ', ' ')
            tweets_no_hashtag.append(tweet_correct_spacing)
        return tweets_no_hashtag

    def remove_html(self, tweets):
        tweets_no_html = []
        for tweet in tweets:
            tweet_no_html = ""
            for word in tweet.split(' '):
                if word in ["&amp;"]:
                    tweet_no_html += ""
                else:
                    tweet_no_html += " " + word + " "
            tweets_no_html.append(tweet_no_html)
        return tweets_no_html


o1 = Preprocessing("./data/validation/task2_en_validation.tsv")
