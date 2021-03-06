import emoji
import pandas as pd

class Preprocessing:
    def __init__(self, file, test, x_col_name, y_col_name):
        """
        This file preprocesses the data and cleans it.

        :param file: path to data file
        :type file: Str
        :param test: flag for if this is processing the test data
        :type test: Bool
        :param x_col_name: Name of the column that has the X data
        :type x_col_name: Str
        :param y_col_name: Name of the column that has the Y data
        :type y_col_name: Str
        """
        self.file = file
        self.test = test
        self.x_col_name = x_col_name
        dataset = pd.read_csv(file, sep='\t')
        tweets = dataset[self.x_col_name].tolist()
        tweets_no_html = self.remove_html(tweets)
        tweets_punctuation = self.remove_punctuation(tweets_no_html)
        tweets_no_hashtag = self.replace_hashtags(tweets_punctuation)
        tweets_no_links = self.replace_links(tweets_no_hashtag)
        tweets_no_usernames = self.replace_usernames(tweets_no_links)
        self.tweets = self.replace_emojis(tweets_no_usernames)

        if self.test:
            self.labels = None
            self.y_col_name = None
        else:
            self.y_col_name = y_col_name
            self.labels = dataset[self.y_col_name].tolist()
            self.labels = [str(lab) for lab in self.labels]

    def remove_punctuation(self, tweets):
        """
        Removes double quotes

        :param tweets:  X data
        :type tweets: List
        :return: tweets in list with double quotes removed
        :rtype: List
        """
        tweets_clean = []
        for tweet in tweets:
            tweet = tweet.replace('“', '')
            tweet = tweet.replace('"', '')
            tweets_clean.append(tweet)
        return tweets_clean


    def replace_emojis(self, tweets):
        """
        Remove emojis and replace them with word that represents  them

        :param tweets: X data
        :type tweets: List
        :return: tweets in list with emojis replaced
        :rtype: List
        """
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
        """
        Replaces usernames with word username

        :param tweets: X data
        :type tweets: List
        :return: tweets in list with usernames replaced
        :rtype: List
        """
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
        """
        Replaces links with word hyperlink

        :param tweets: X data
        :type tweets: List
        :return: tweets in list with links replaced
        :rtype: List
        """
        tweets_no_links=[]
        for tweet in tweets:
            tweet_no_link = ""
            for word in tweet.split(' '):
                if word.startswith('http://'):
                    tweet_no_link+=" hyperlink " # we say in paper that we replace with "link" 
                else:
                    tweet_no_link+=" " + word + " "
            tweet_correct_spacing = tweet_no_link.replace('  ', ' ').strip()
            tweets_no_links.append(tweet_correct_spacing)
        return tweets_no_links

    def replace_hashtags(self, tweets):
        """
        Replaces hashtags with word hashtag

        :param tweets: X data
        :type tweets: List
        :return: tweets in list with hashtags replaced
        :rtype: List
        """
        tweets_no_hashtag = []
        for tweet in tweets:
            tweet_no_hashtag = ""
            for index, char in enumerate(tweet):
                if index < len(tweet) - 1:
                    if char == "#" and tweet[index+1].isalpha():
                        tweet_no_hashtag += " hashtag "
                    else:
                        tweet_no_hashtag += char
                else:
                    tweet_no_hashtag += char
            tweet_correct_spacing = tweet_no_hashtag.replace('  ', ' ')
            tweets_no_hashtag.append(tweet_correct_spacing)
        return tweets_no_hashtag

    def remove_html(self, tweets):
        """
        Removes &amp

        :param tweets: X data
        :type tweets: List
        :return: tweets in list &amp removed
        :rtype: List
        """
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

    def remove_drug_names(self, tweets):
        """
        Replaces drug names with word drug. This is an experimental feature.

        :param tweets: X data
        :type tweets: List
        :return: tweets in list with drug names replaced
        :rtype: List
        """
        # this code creates a list of drug names. You will have to change it so
        # it works with whatever file of drug name you have 
        tweets_no_drug_names = []
        drugs = pd.read_csv("Products.txt", sep="\t", error_bad_lines=False)
        drug_name_list = drugs['DrugName'].tolist()
        my_list = []
        for drugs in drug_name_list:
            for drug in drugs.split(' '):
                my_list.append(drug.strip().lower())


        for tweet in tweets:
            tweet_no_drugs = ""
            for word in tweet.split(' '):
                if word in my_list:
                    tweet_no_drugs += " drug "
                else:
                    tweet_no_drugs += " " + word + " "
            tweets_no_drug_names.append(tweet_no_drugs)
        return tweets_no_drug_names
