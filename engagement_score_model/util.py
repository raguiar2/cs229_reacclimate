import torch
import numpy as np
import pandas as pd
import spacy
import os
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

nlp = spacy.load('en')
def predict_engagement(model, sentence, TEXT, device):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

###################################
#This function constructs the engagement score dataset
#tweetFile is the raw csv file from step 1
#userFile is the user group assignment from step 2
##################################
def construct_engagement_score_ds( tweetFile, userFile ):
    assert os.path.isfile(tweetFile), 'Invalid Tweet file'
    assert os.path.isfile(userFile), 'Invalid userFile'

    userNameCol = 'screen_name'
    parentIdCol = 'parent_id'
    usrGrpCol = 'group'
    idCol = 'id'
    egmtCol = 'engagement'

    tweetDf = pd.read_csv( tweetFile )
    userDf = pd.read_csv( userFile )

    #Add user group assignment to tweet Df
    tweetDf = pd.merge( tweetDf, userDf, how='inner', on=userNameCol )
    
    engagementDf = []
    for pId in tweetDf[parentIdCol].dropna().unique().astype(
            tweetDf[idCol].dtype):
        curTweetEntry = []
        tweet = tweetDf[tweetDf[idCol] == pId]
        if tweet.shape[0] == 0:
            print( 'Invalid parent ID %s, continue', pId )
            continue
        parentFollower = tweet['follower_count'].values[0]+1
        curTweetEntry.append( tweet['clean_text'].values[0] ) 
        replyDf = tweetDf[tweetDf[parentIdCol]==pId]
        #Shifting to polarity to [0 to 2] TODO: why aren't we using the continous value as a feature? 
        replyDf.loc[:,egmtCol] = np.log((( replyDf['favorite_count'] +
                replyDf['retweet_count'] + 1) * ( replyDf['polarity'] + 1 ) / 
                parentFollower ).values[0] + 1E-15)

        for grp in sorted(userDf[usrGrpCol].unique()):
            if grp in replyDf[usrGrpCol].values:
                curTweetEntry.append( replyDf[replyDf[usrGrpCol]==grp]
                        [egmtCol].mean() )
            else:
                # What is this -15? 
                curTweetEntry.append( -15 )
        engagementDf.append( curTweetEntry )

    engagementDf = pd.DataFrame( engagementDf )
    engagementDf.columns = ['text'] + sorted(userDf[usrGrpCol].unique()) 
    return engagementDf

def convert_to_categorical( thresholds, tarCol, df ):
    catDf = df.copy()
    for i in range(1,len(thresholds)):
        idx = (df[tarCol] < thresholds[i]) & (
                df[tarCol] >= thresholds[i-1])
        catDf.loc[idx,tarCol] = i-1
    return catDf




