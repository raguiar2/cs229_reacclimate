#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import tweepy
import os.path
import os
import sklearn
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import csv
import re #regular expression
from textblob import TextBlob
import string
#import preprocessor as p
import argparse
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords, wordnet
#from preprocessor.api import clean, tokenize, parse

# In[4]:


#Twitter credentials for the app
consumer_key = 'Lydypy5GRHslhuWsXTAagVFpO'
consumer_secret = 'K9HA6MyfRWm73G50WHvzBPxfY0gWfJRk5ajcUmGRCg4e9NiM69'
access_key= '789687511-BGbhUzj8zVLk9HeKKxrCZnzJ21xb3qXqZMHyf0gX'
access_secret = 'kIDmi6vhOiePyEIZ5XXrOV8rl0xLOe5wLQ2XbhH2qCLsr'


# In[5]:


#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[6]:


#columns of the csv file 
COLS = ['screen_name', 'id', 'parent_id', 'created_at', 'favorite_count', 'retweet_count', 'follower_count',        'clean_text', 'polarity','subjectivity', 'hashtags','location', 'coordinates']


# In[7]:


#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])


# In[8]:


# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])


# In[9]:


#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)


# In[10]:


#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


# In[11]:


def clean_tweets(tweet):
 
    #stop_words = set(stopwords.words('english'))
#after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r'RT *[^:]*:','', tweet);
    #tweet = re.sub(r'[\.@[a-zA-Z]*:]+','', tweet);
    tweet = re.sub(r':', '', tweet)
#replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
#remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
#remove url
    tweet = re.sub(r"http\S+", "", tweet)
#remove more symbols
    tweet = tweet.replace("-", "")
    tweet = tweet.replace("_", "")
    tweet = tweet.replace("*", "")
    tweet = tweet.replace(".", "")
#filter using NLTK library append it to a string
    filtered_tweet = []
    word_tokens = word_tokenize(tweet)
#looping through conditions
    for w in word_tokens:
#check tokens against stop words , emoticons and punctuations
        if w not in emoticons and w not in string.punctuation:
            if w.isdigit():
                continue
            if w[0] == ".":
                continue
            if '9' in w:
                w = w.replace("9", "")
                w = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', w)
            
            w = w.lower()
            
            if len(w) == 1 and w not in ['a', 'e', 'i','o','u']:
                continue
            elif len(w) == 2 and w[0] in string.punctuation:
                continue
            filtered_tweet.append(w.lower())
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)return tweet


# In[12]:


contractions_dict = { 
"ain't": "have not",
"aren't": "are not",
"can't": "cannot",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'll": "he will",
"he's": "he has",
"how'd": "how did",
"how'll": "how will",
"how's": "how are",
"i'd": "I would",
"i'll": "I will",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"o'clock": "of the clock",
"oughtn't": "ought not",
"she'd": "she would",
"she'll": "she will",
"she's": "she has",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there would",
"there's": "there has",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"y'all": "you all",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"you've": "you have"
}


# In[13]:


contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()), flags = re.IGNORECASE)
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0).lower()]
    return contractions_re.sub(replace, s)


# In[14]:


def addToDataFrame(status, file):
    ne_entry = []
    parent_id = ""
    if hasattr(status, 'in_reply_to_status_id_str'):
        parent_id = status.in_reply_to_status_id_str

    #if "retweeted_status" in status:
        #status['full_text'] = status["retweeted_status"]['full_text']

    original_tweet = status['full_text']

    tweet = status['full_text'].replace("&amp;", "and")
    tweet = tweet.replace("’", "'")
    tweet = tweet.replace("#", "9")
    tweet = expand_contractions(tweet)

    #p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY)
    #clean_text = clean(tweet)
    filtered_tweet=clean_tweets(tweet)

    filtered_tweet = filtered_tweet.strip()
    if len(filtered_tweet.split()) < 2:
        return

    blob = TextBlob(filtered_tweet)
    Sentiment = blob.sentiment     
    polarity = Sentiment.polarity
    subjectivity = Sentiment.subjectivity

    new_entry += [status['user']['screen_name'], status['id'], parent_id, status['created_at'], status['favorite_count'],                   status['retweet_count'], status['user']['followers_count'], filtered_tweet, polarity,subjectivity]
    hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
    new_entry.append(hashtags) #append the hashtags

    #append_locations
    try:
        location = status['user']['location']
    except TypeError:
        location = ''
    new_entry.append(location)

    try:
        coordinates = status['place']['full_name']
    except TypeError:
        coordinates = None
    new_entry.append(coordinates)


    single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
    global df
    df = df.append(single_tweet_df, ignore_index=True)
    csvFile = open(file, 'a+' ,encoding='utf-8')



# In[15]:


def write_tweets(query, file):
    #If the file exists, then read the existing data from the CSV file.
    '''
    if os.path.isfile(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=COLS)
    '''
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q = query, count=200, lang = 'en',                               include_rts=False, tweet_mode = 'extended', since='2019-04-15').pages(1000):
        for status in page:
           
            status = status._json
            
            if status['id'] in df['id'].values:
                continue
                     
            
            #filtering tweets by number of favorites
            if status['favorite_count'] < 4:
                continue
            
            addToDataFrame(status, file)
            
            #get replies
            name = status['user']['screen_name']
            for page in tweepy.Cursor(api.search, q='to:'+name, count=200, lang = 'en',                                       include_rts=False, tweet_mode = 'extended', since='2019-04-15').pages(5):
                for tweet in page:
                    if (tweet.in_reply_to_status_id_str==str(status['id'])):
                        addToDataFrame(tweet._json, file)
                 


# In[2]:


#declare file paths as follows for three files
query1 = "climate change -filter:retweets"
query2 = "global warming -filter:retweets"
query3 = "climate -change -filter:retweets"

file = "/Users/User/229project/cs229_reacclimate/data/test.csv"

if os.path.isfile(file):
    df = pd.read_csv(file, header=None)
else:
    df = pd.DataFrame(columns=COLS)

write_tweets(query1, file)
write_tweets(query2, file)
write_tweets(query3, file)

df.to_csv(file, mode='a+', columns=COLS, index=False, encoding="utf-8")


def getUserTweets(handle, file):
     
    for page in tweepy.Cursor(api.user_timeline, id=handle,                              count=200, include_rts=True, tweet_mode = 'extended').pages():
        for status in page:
            
            new_entry = []
            status = status._json
            
            climateWords = ['climate', 'global warming', 'sea level', 'mass extinction', 'clean energy',                             'renewable energy', 'carbon emission', 'wildlife', 'biodiversity']
            if not any(word in status['full_text'] for word in climateWords):
                continue
                
            if status['id'] in df['id'].values:
                continue
                  
            addToDataFrame(status, file)   
            
            #get replies
            name = status['user']['screen_name']
            for page in tweepy.Cursor(api.search, q='to:'+name, count=200, lang = 'en',                                       include_rts=False, tweet_mode = 'extended').pages(5):
                for tweet in page:
                    if (tweet.in_reply_to_status_id_str==str(status['id'])):
                        tweet = tweet._json
                        
                        if tweet['user']['screen_name'] not in users:
                            new_users.append(tweet['user']['screen_name'])
                            users.append(tweet['user']['screen_name'])
                        
                        addToDataFrame(tweet, file)
                    
             


# In[91]:


#get climate tweet history of all users (and more replies!)

#file = "/Users/User/229project/cs229_reacclimate/data/test.csv"
#newDF = pd.read_csv(file, usecols = [1])
users = df['screen_name'].values.tolist()

new_users = [] #if we get more replies from the tweets of users, this can track the unique new users

for handle in users:
    getUserTweets(handle, file)
    
newDF.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")


# In[ ]:


'''
Thoughts: 
    -We can limit the climate tweets we download  by num of favorites, num retweets, or num followers, 
        or a combination of those...this actually cuts the data by a lot...the vast majority of tweets
        gathered have 0 likes and retweets...there are a LOT of tweets about climate so it might make sense to pick 
        the more popular tweets. What should be that threshold? As an example:
        with no limitation on the number of favorites, 100 pages of download was 10k tweets for query 'climate change' 
        all from the same day I downloaded...but with even a limit of at least 2 favorites, the number of tweets 
        for 100 pages was cut to about 2k. Currently I set the threshold to 5 and am looking at 200 pages 
        of tweets starting from today, May 10th.
    -Is there any particular date we want to get tweets from...again, there are so many postings that most of 
        the tweets we'll get will be on that date. 
    -Engagement can be measured by some sort of ratio of number of sentiment/likes/rts to number of followers
    -How many of these "base" tweets/users do we get...from which we'll get the replies...and then all
    users' climate-focused tweet histories...
        -Should we get "base" tweets at different time points? In different locations?
    -Its 2am right now, so many of the tweets I downloaded are not from the United States...unf its not that easy 
        to limit tweets by  location...have to put in a geocode with longtitude, latitude, and radius that encompasses 
        the US  in query. It's doable...just need to figure out what those are...and to it'll have to include Canada
        if we're trying to reach Alaska. On the other hand, if we pull at a better time, aka not when the US is sleeping
        I'm confident that most of the tweets will be from the US.
        
Next steps:
    -Answer above questions
    -Re-download a lot more base tweets
        -Download replies off base tweets via https://gist.github.com/edsu/54e6f7d63df3866a87a15aed17b51eaf methodology...
    -Get all screen names from above tweets and download their climate-related tweets
        --this will involve running individual queries on an array of thousands of handles...I'm currently
        already running into 429 "too many request errors" which slowwwwss down the download, so may need to
        figure out how to work around this. I think it may not exist with a premium account.
'''


# In[ ]:




