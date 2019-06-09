import numpy as np
import pandas as pd
import os
import pdb

########################################
# Assumes the first col of df is name
# the rest of the cols are user group engagements
#######################################
def get_engagement_category_thres( df, catCount ):
    engagements = []
    engagements = np.sort( df['engagement'] )

    divIndex = [0]
    stepSize = np.floor( engagements.shape[0] / catCount )
    divIndex = np.arange( 0, engagements.shape[0] + 1, stepSize, dtype=np.int )
    divIndex[-1] = engagements.shape[0] - 1
    categoryThres = []
    for i in range( len(divIndex) ):
        categoryThres.append( engagements[ divIndex[ i ] ] )

    print( 'Engagement score breakdown is %s' % categoryThres )
    return categoryThres

def convert_to_categorical( thresholds, df ):
    catDf = df.copy()
    for col in [df.columns[-1]]:
        for i in range(1,len(thresholds)):
            idx = (df[col] <= thresholds[i]) & (
                    df[col] >= thresholds[i-1])
            print( 'Col %s between %.1f and %.1f, there are '
                    '%s datapoints' % ( col, thresholds[i],
                        thresholds[i-1], np.sum( idx ) ) )
            catDf.loc[idx,col] = int(i-1)
            catDf[col] = catDf[col].astype(int)
    return catDf

###################################
#This function constructs the engagement score dataset
#tweetFile is the raw csv file from step 1
#userFile is the user group assignment from step 2
##################################
def construct_engagement_score_ds( tweetFiles, userFile ):
    for tFile in tweetFiles:
        assert os.path.isfile(tFile), 'Invalid Tweet file'
    assert os.path.isfile(userFile), 'Invalid userFile'

    userNameCol = 'screen_name'
    parentIdCol = 'parent_id'
    usrGrpCol = 'group'
    idCol = 'id'
    egmtCol = 'engagement'

    tweetDf = None
    for tFile in tweetFiles:
        curDf = pd.read_csv( tFile )
        if tweetDf is None:
            tweetDf = curDf
        else:
            tweetDf = tweetDf.append( curDf )
        print( tweetDf.shape )
    userDf = pd.read_csv( userFile )

    userGrps = userDf[usrGrpCol].unique()
    print( 'User groups: %s' % userGrps )
    #Add user group assignment to tweet Df
    tweetDf = pd.merge( tweetDf, userDf, how='inner', on=userNameCol )
    print( 'tweetDf: ', tweetDf.shape )

    tweetDf.loc[:,egmtCol] = ( tweetDf['favorite_count'] + 
            tweetDf['retweet_count'] + 1 ) / np.log( 
            tweetDf['follower_count'] + 10 )
    
    tweetDf = tweetDf[ [ 'clean_text', 'group', egmtCol ] ]

    return tweetDf

tweetFiles = [ '../data/GWUFile1.csv',
               '../data/GWUFile2.csv',
               '../data/GWUFile3.csv',
               '../data/GWUFile4.csv',
            ]
userFile = '../clustering/clusters.csv'
oFile = './engagement.csv'
trainFile = './train.csv'
valFile = './val.csv'
testFile = './test.csv'

testRatio = 0.1
valRatio = 0.1
catCnt = 3

engagementDf = construct_engagement_score_ds(tweetFiles, userFile)
engagementDf.to_csv('ds_optB.csv',index=False)
#engagementDf = pd.read_csv('ds_optB.csv')
thresholds = get_engagement_category_thres( engagementDf, catCnt )
engagementDf = convert_to_categorical( thresholds, engagementDf )
engagementDf.to_csv(oFile, index=False)

testSize = np.round( engagementDf.shape[0] * testRatio, 0 )
valSize = np.round( engagementDf.shape[0] * valRatio, 0 )
print( 'Test size is %s, val size is %s' % ( testSize, valSize ) )
engagementDf.loc[0:testSize].to_csv( testFile, index=False )
engagementDf.loc[testSize:(testSize+valSize)].to_csv( valFile, index=False )
engagementDf.loc[(testSize+valSize):].to_csv( trainFile, index=False )
