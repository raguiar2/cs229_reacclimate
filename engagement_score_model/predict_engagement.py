#!/usr/bin/env python
# coding: utf-8
import torch
import random
import pandas as pd
from torchtext import data
import torch.nn as nn
import torch.optim as optim
from models import LTSM
import util
import time
import pdb
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet 
from collections import defaultdict, namedtuple


######################################################
#Hyperparameters and config variables
######################################################
SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64 * 4
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 30
best_valid_loss = float('inf')
tPath = '../twitter/data/'
trainFile = './test.csv'
testFile = './test.csv'
valFile = './val.csv'

df = pd.read_csv(valFile)
usrGrpCnt = len(df.columns) - 1
sentCategoryCnt = len(df[df.columns[-1]].unique())

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True, lower=True)
LABEL = data.LabelField(dtype = torch.long)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csvFields = [   ('text', TEXT) ]
for userGrp in range( usrGrpCnt ):
    label = 'group%s' % userGrp
    csvFields.append( ( label, LABEL ) )

train_data, valid_data, test_data = data.TabularDataset.splits(
                path='.', 
                train=trainFile,
                validation=valFile, 
                test=testFile, 
                format='csv',
                fields=csvFields,
                skip_header=True,
            )

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

INPUT_DIM = 25002
PAD_IDX = 1
modelGrp0 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, 1*sentCategoryCnt, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
modelGrp1 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, 1*sentCategoryCnt, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

modelGrp0 = modelGrp0.to(device)
modelGrp1 = modelGrp1.to(device)

modelGrp0.load_state_dict(torch.load('lstm_model_group0.pt'))
modelGrp1.load_state_dict(torch.load('lstm_model_group1.pt'))

print(util.predict_engagement(modelGrp0, 'Climate change is terrible', TEXT, device))
print(util.predict_engagement(modelGrp1, 'We need to act now to fix climate change', TEXT, device))

'''
# iterate through words in a unique corpus dictionary
unique_words = set()
word_to_tweets = defaultdict(list)
tweet_to_engagement = defaultdict(list)
alt_tweet_to_engagement = defaultdict(lambda: defaultdict(list))

tweet_file = pd.read_csv(testFile)
tweets = tweet_file['text']
tweets = tweets[:3]

for tweet_idx, tweet in enumerate(tweets):
    filtered_words = [word for word in tweet.split(' ') if word not in stopwords.words('english')]
    group_one_engagement = 0
    group_zero_engagement = 1
    # TODO: replace with model here
#     group_one_engagement = util.predict_engagement(model_group_one, tweet, TEXT, device)
#     group_zero_engagement = util.predict_engagement(model_group_zero, tweet, TEXT, device)
    for word in filtered_words:
        tweet_to_engagement[word].append((group_zero_engagement, group_one_engagement))

for tweet_idx, tweet in enumerate(tweets):
    filtered_words = [word for word in tweet.split(' ') if word not in stopwords.words('english')]
    unique_words = unique_words.union(filtered_words)
    for word in filtered_words:
        word_to_tweets[word].append(tweet_idx)
        

# for each word, get 5 alternatives
for word in unique_words:
    syns = wordnet.synsets(word) 
    alternatives = []
    for synonym in syns:
        syn = synonym.lemmas()[0].name()
        if syn != word and syn not in alternatives:
            alternatives.append(syn)
        if len(alternatives) == 5:
            break
    tweets_with_word = word_to_tweets[word]
    # for each alt, iterate through tweets that contain this word, substitute word with alt
    for alt in alternatives:
        for tweet_idx in tweets_with_word:
            tweet = tweets[tweet_idx]
            alt_tweet = tweet.replace(word, alt)
            # recompute engagement score delta across all user groups
            group_one_engagement = 0
            group_zero_engagement = 1
            # TODO: replace with model here
            #group_one_engagement = util.predict_engagement(model_group_one, alt_tweet, TEXT, device)
            #group_zero_engagement = util.predict_engagement(model_group_zero, alt_tweet, TEXT, device)
            alt_tweet_to_engagement[word][alt].append((group_one_engagement, group_zero_engagement))
# record alt with highest delta
replacements = []
Replacement = namedtuple('Replacement', ['delta', 'original', 'alt'])
for word in tweet_to_engagement:
    engagement_list = tweet_to_engagement[word]
    avg_engagement_orig = np.mean(engagement_list, axis=0)
    alt_words = alt_tweet_to_engagement[word]
    for alt_word in alt_words:
        avg_engagement_alt = np.mean(alt_words[alt_word], axis=0)
        delta = sum(avg_engagement_alt - avg_engagement_orig)
        replacements.append(Replacement(delta, word, alt_word))
replacements.sort(key=lambda x: x.delta)
# record top 10 words with highest delta and that is our answer
print(replacements[-10:])

'''
# In[88]:





# In[89]:





# In[ ]:




