#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load the pytorch model
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


# In[2]:


SEED = 1234
MAX_VOCAB_SIZE = 10_000
BATCH_SIZE = 64 * 64
EMBEDDING_DIM = 50
HIDDEN_DIM = int(256/8)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 10
best_valid_loss = float('inf')
tPath = '../twitter/data/'
trainFile = './train.csv'
testFile = './test.csv'
valFile = './val.csv'

df = pd.read_csv(valFile)
usrGrpCnt = len(df.columns) - 1
sentCategoryCnt = len(df[df.columns[-1]].unique())
output_dim = 1

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True, lower=True)
LABEL = data.LabelField(dtype = torch.float)
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
                 vectors = "glove.twitter.27B.50d", 
                 unk_init = torch.Tensor.normal_)

INPUT_DIM = 10002
PAD_IDX = 1
modelGrp0 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, output_dim, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
modelGrp1 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, output_dim, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

model_group_zero = modelGrp0.to(device)
model_group_one = modelGrp1.to(device)


# In[3]:


model_group_zero.load_state_dict(torch.load('lstm_model_group0.pt'))
model_group_one.load_state_dict(torch.load('lstm_model_group1.pt'))


# In[10]:


print("example engagement scores:")
follower_count = torch.tensor( [[0.2]] ).to(device)
first_ex_engagement = util.predict_engagement(model_group_zero, 'Climate change is terrible', TEXT, device, follower_count).item()
second_ex_engagement = util.predict_engagement(model_group_one, 'We need to act now to fix climate change', TEXT, device, follower_count).item()
print('"Climate change is terrible": ', first_ex_engagement)
print('"We need to act now to fix climate change": ', second_ex_engagement)


# In[11]:


# iterate through words in a unique corpus dictionary
unique_words = set()
word_to_tweets = defaultdict(list)
tweet_to_engagement = defaultdict(list)
alt_tweet_to_engagement = defaultdict(lambda: defaultdict(list))

tweet_file = pd.read_csv(testFile)
tweets = tweet_file['clean_text']
followers = tweet_file['follower_count']
tweets = tweets[:1000]
followers = followers[:1000]

inverse_box = lambda x: (x*(-0.6)+1)**(1/-0.6)

print("predicting engagements")
for tweet_idx, tweet in enumerate(tweets):
    filtered_words = [word for word in tweet.split(' ') if word not in stopwords.words('english')]
    num_followers = torch.tensor(followers[tweet_idx]).to(device)
    group_one_engagement = util.predict_engagement(model_group_one, tweet, TEXT, device, num_followers).item()
    group_zero_engagement = util.predict_engagement(model_group_zero, tweet, TEXT, device, num_followers).item()
    for word in filtered_words:
        tweet_to_engagement[word].append(((group_zero_engagement), (group_one_engagement)))
print("engagements predictsd")
        
print("getting words")
for tweet_idx, tweet in enumerate(tweets):
    filtered_words = [word for word in tweet.split(' ') if word not in stopwords.words('english')]
    unique_words = unique_words.union(filtered_words)
    for word in filtered_words:
        word_to_tweets[word].append(tweet_idx)
print("got words")


print("getting alts")
# for each word, get 5 alternatives
print("len of alts ", len(unique_words))
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
            num_followers = torch.tensor(followers[tweet_idx]).to(device)
            alt_tweet = tweet.replace(word, alt)
            # recompute engagement score delta across all user groups
            group_one_engagement = util.predict_engagement(model_group_one, alt_tweet, TEXT, device, num_followers).item()
            group_zero_engagement = util.predict_engagement(model_group_zero, alt_tweet, TEXT, device, num_followers).item()
            alt_tweet_to_engagement[word][alt].append(((group_one_engagement), (group_zero_engagement)))
print("got alts")
# record alt with highest delta
print("getting replacements")
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
print("got replacements")
# record top 10 words with highest delta and that is our answer
print("top 20 words and replacements are")
for replacement in replacements[-20:][::-1]:
    print("delta: {}, originial: {}, new: {}".format(replacement.delta, replacement.original, replacement.alt))


# In[ ]:


tweet_file


# In[ ]:


alt_tweet_to_engagement['dangers']['risk']


# In[ ]:





# In[ ]:




