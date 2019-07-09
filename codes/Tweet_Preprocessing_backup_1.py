# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:07:23 2018

@author: Olashile Adebimpe
"""

import os
import re
import sys
import nltk
import string
import random
import spacy
import scipy.stats as stats
import pylab as pl
import math
import guidedlda
import pandas as pd
import numpy as np
import string as st
import seaborn as sns
import multidict as multidict
import matplotlib.pyplot as plt
import warnings
from nltk import tag
from PIL import Image
from wordcloud import WordCloud
from random import shuffle
from textblob import TextBlob
from datetime import datetime
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from gensim.corpora import Dictionary 
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')

#to get the current  path of the program and append the data directory path
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'data')
image_path = os.path.join(root_dir, 'images' , 'Twitter-logo.png')

#==============================================================================
# Getting a list of all the files in the directory where all the files are stored
#==============================================================================
dirFileList = os.listdir(data_dir)

#adding the letters and numbers to the set of stop words.
stop = set(stopwords.words('english')) | set(string.ascii_letters)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

#==============================================================================
#==============================================================================
#==============================================================================
# # # Below are functions used in the body of the program
#==============================================================================
#==============================================================================
#==============================================================================

#==============================================================================
# #for removing URl
#==============================================================================
def remove_hyperlink(df):
    df = re.sub(r"http\S+", "", df)
    return re.sub(r'(\\x(.){2})', '',df)


#==============================================================================
# #removes Mentions and hastags
#==============================================================================
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  list(string.punctuation):
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if ((word[0] not in entity_prefixes) and (word != 'RT')):
                words.append(word)
    return ' '.join(words)

#==============================================================================
# The below function remove stopwords from our document
# remove punctuations also
# and normalize using WordNetLemmatizer
#==============================================================================

def clean(doc):    
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])    
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)    
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())    
	return normalized

#==============================================================================
# Function for getting the sentiment analysis of a tweet
#==============================================================================

def get_tweet_sentiment(tweet):
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)        
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
        
#==============================================================================
# Function for displaying the modelled topics
#==============================================================================

def display_topics(model, feature_names_x, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        #print ("Topic %d:" % (topic_idx))
        print ("Topic %2d: " % (topic_idx+1)+" ".join([feature_names_x[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

#==============================================================================
# Function for getting  a list of the modelled topics
#==============================================================================
def get_list_of_topics(model, feature_names_x, no_top_words):
    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        topic_list.append([feature_names_x[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topic_list

#==============================================================================
# function for grouping/aggregating tweets using the sentiment of the tweets 
# and the time od the day the tweet was posted.
# it returns the hastag and the groupped tokens
#==============================================================================

def grouping_date_sentiment(df):
    full_list = []
    #gets the list of sentiments in the dataframe
    for sentiment in list(df.sentiment.unique()):
        #creates a new list for each sentiment
        full_list_sentiment = []
        #gets only tweets for the selected sentiment
        temp = df.loc[df['sentiment'] == sentiment]
        #sorts by date and remove duplicate tweets
        temp['Date'] =pd.to_datetime(temp.Date)
        temp = temp.sort_values(by='Date')
        temp = list(set(temp.clean_tweet))
        
        for row in temp:
            #list of all word tokens
            full_list_sentiment.extend(row.split())
        #breaking down my token by len of 2000
        chunks = [full_list_sentiment[x:x+2000] for x in range(0, len(full_list_sentiment), 2000)]
        
        full_list.extend(chunks)     
    return list(df.hastag.unique())[0] , full_list

#==============================================================================
# Adhoc function to flatten my list
#==============================================================================
def flaten_my_list(mylist):
    a = []
    for values in mylist:
        a.append(' '.join(values))
    return a

#==============================================================================
# Used for creating seed list to ensure only correct word spellings are used for
# seed list and it used wordnet synsets library
#==============================================================================

def get_right_words(tmp):
    result = []
    for word in tmp:
        if wordnet.synsets(word):
            result.append(word)
        else:
            continue
    return result    


def wordCloudInput(token_list):
    word_cloud_list = ""
    for  line in token_list:
        word_cloud_list = word_cloud_list + line
    
    return word_cloud_list

def wordCloudInput_result(token_list):
    word_cloud_list = []
    for  line in token_list:
        for line2 in line:
            word_cloud_list.append(line2)
    
    return " ".join(word_cloud_list)


def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be",text):
            continue
        val = tmpDict.get(text,0)
        tmpDict[text.lower()] = val+1
    for key in tmpDict:
        fullTermsDict.add(key,tmpDict[key])
    return fullTermsDict

def makeImage(text):
    alice_mask = np.array(Image.open(image_path))

    wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask)
    # generate word cloud
    wc.generate_from_frequencies(text)
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
def get_count_vectorizer(flat_list_xx , no_features_xx ):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features_xx, stop_words='english')
    tf = tf_vectorizer.fit_transform(flat_list_xx)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf , tf_feature_names



#==============================================================================
# Function returns the seed list for for the Guided LDA topic modelling algorithm
#==============================================================================
def get_seed_list(flat_list_xx  ):
    tagged_sent = []
    for tags in flat_list_xx:
        tagged_sent.extend(tag.pos_tag(tags.split(sep = " ")))
    
    seed_topic_list =  list(set(word for word, tag in tagged_sent if tag in ('NN')))
    seed_topic_list = get_right_words(seed_topic_list)
    shuffle(seed_topic_list)
    cnt = round(len(seed_topic_list)/10)
    chunks_123 = [seed_topic_list[x:x+cnt] for x in range(0, len(seed_topic_list), cnt)]
    
    Vocab = []
    for line in flat_list_xx:
        Vocab.extend(line.split(sep = " ")) 
    word2id = dict((v, idx) for idx, v in enumerate(Vocab))
    seed_topics = {}
    for t_id, st in enumerate(chunks_123):
        for word in st:
            seed_topics[word2id[word]] = t_id
    
    return seed_topics

#==============================================================================
# Function return the modelled topics with gthe coherence value for
# a corupus using Latent Dirichlet Allocation (LDA)
#==============================================================================
def get_topics_using_LDA(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
    lda = LatentDirichletAllocation(n_topics=no_topics_xx, max_iter=500, learning_method='online', learning_offset=50.,random_state=1).fit(tf_xx)
    #display_topics(lda, tf_feature_names_xx, no_top_words_xx)
    lda_topic = get_list_of_topics(lda, tf_feature_names_xx, no_top_words_xx)
    lda_coherence = CoherenceModel(topics=lda_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
    return lda_topic , lda_coherence


#==============================================================================
# Function return the modelled topics with gthe coherence value for
# a corupus using Non-negative Matrix Factorisation (NMF)
#==============================================================================
def get_topics_using_NMF(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
    nmf = NMF(n_components=no_topics_xx, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf_xx)
    #display_topics(nmf, tf_feature_names_xx , no_top_words_xx)
    nmf_topic = get_list_of_topics(nmf, tf_feature_names_xx , no_top_words_xx)
    nmf_coherence = CoherenceModel(topics=nmf_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
    return nmf_topic , nmf_coherence


#==============================================================================
# Function return the modelled topics with gthe coherence value for
# a corupus using semi supervised Guided LDA without seeding
#==============================================================================
def get_topics_GLDA_without_seed(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
    gda_wthout_seed = guidedlda.GuidedLDA(n_topics=no_topics_xx, n_iter=500, random_state=1, refresh=100).fit(tf_xx)
    #display_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
    gda_topic = get_list_of_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
    gda_wthout_seed_coherence = CoherenceModel(topics=gda_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
    return gda_topic , gda_wthout_seed_coherence

#==============================================================================
# Function return the modelled topics with gthe coherence value for
# a corupus using semi supervised Guided LDA with seed topics
#==============================================================================
def get_topics_GLDA_with_seed(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
    flat_list = flaten_my_list(list_df_xx)
    seed_topics = get_seed_list(flat_list )
    glda_with_seed = guidedlda.GuidedLDA(n_topics=no_topics_xx, n_iter=500, random_state=1, refresh=100)
    glda_with_seed.fit(tf_xx, seed_topics=seed_topics, seed_confidence=0.15)   
    #display_topics(glda_with_seed ,  tf_feature_names_xx, no_top_words_xx)
    glda_topic = get_list_of_topics(glda_with_seed ,  tf_feature_names_xx, no_top_words_xx)
    glda_wth_seed_coherence = CoherenceModel(topics=glda_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
    return glda_topic , glda_wth_seed_coherence

def get_wordlist(hashtag):
    hashtag_dict = { 'HolocaustMemorialDay' : ['Holocaust', 'Memorial' ,'Day'],
                     'Obama'                : ['Obama'],
                     'TuesdayThoughts'      : ['Tuesday' , 'Thoughts'],
                     'iHeartAwards'         : ['Heart' , 'Awards'],
                     'FalconHeavy'          : ['Falcon' , 'Heavy'],
                     'MondayMotivation'     : ['Monday' , 'Motivation'],
                     'SundayFunday'         : ['Sunday' , 'Funday'],
                     'FridayReads'          : ['Friday' , 'Reads'],
                     'Trump'                : ['Trump'],
                     'ThursdayThoughts'     : ['Thursday' , 'Thoughts'],
                     'FinalFour'            : ['Final' , 'Four'],
                     'PressforProgress'     : ['Press' , 'Progress'],
                     'SuperBowl'            : ['Super' , 'Bowl'],
                     'BreakingNews'         : ['Breaking' , 'News']
                    }
    return hashtag_dict[hashtag]


def get_average_similirarity_matrix(hashtag , modeled_topics , model ):
    list1 =  get_wordlist(hashtag)
    list2 = [token for values in modeled_topics for token in values]
    list = []
    list_count = []
    
    for word1 in list1:
        for word2 in list2:
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            if wordFromList1 and wordFromList2:
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                list.append((word1 , word2 , s))
                if(s is not None):
                    list_count.append(s)
                else:
                    list_count.append(0)
                    
    h = sorted(list_count)
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  
    pl.plot(h,fit,'-o')
    pl.hist(h,normed=True)      #use this to draw histogram of your dat
    pl.show() 


#==============================================================================
#==============================================================================
#==============================================================================
# # # End of functions used in the body of the program
#==============================================================================
#==============================================================================
#==============================================================================



# dataframe to collect all sampled tweets
df_tweet = pd.DataFrame(columns = ['Date' , 'tweet' , 'hastag'] )
for file in dirFileList:
    if (file.endswith('.csv')):
        path = os.path.join(data_dir , file)
        columns = ['Date' , 'tweet' ]
        df = pd.read_csv(path,names=columns , header=None)
        df['hastag'] = file[9:-4]        
        df_tweet = df_tweet.append(df)

#==============================================================================
# End of Loading the tweets file into the df_tweet padaframe
#==============================================================================        

df_tweet = df_tweet.drop_duplicates(subset=['Date', 'tweet' , 'hastag'], keep=False)
#to get the  hastags with tweet over 1000
a = df_tweet.hastag.value_counts() >= 1000
a = list(a.index[a])

#return only tweets above 1000 count
df_tweet = df_tweet.loc[df_tweet['hastag'].isin(a)]

#==============================================================================
# Tweet Preprocessing
#==============================================================================
df_tweet['clean_tweet'] =  df_tweet.apply(lambda x : clean(strip_all_entities(remove_hyperlink(x['tweet'] ))), axis=1)

#==============================================================================
# Sentiment Analysis using TextBlod Library
#==============================================================================
df_tweet['sentiment'] =  df_tweet.apply(lambda x : get_tweet_sentiment(x['clean_tweet'] ), axis=1)


#==============================================================================
# using word cloud for visualisation
# 
#==============================================================================

for hashtag in list(df_tweet.hastag.unique()):
    print('... WordCloud Token Visualisation for {} ....'.format(hashtag))
    temp = list(set(df_tweet['clean_tweet'].loc[df_tweet['hastag'] == hashtag]))
    temp1 = wordCloudInput(temp)
    makeImage(getFrequencyDictForText(temp1))
    

#==============================================================================
# Topic Modelling
#  1. NMF
#  2. LDA
#  3.GuidedLDA with and without seeding
#==============================================================================

no_features = 2000
no_topics = 10
no_top_words = 10

test_hashtags = ['HolocaustMemorialDay',
                 'Obama', 
                 'TuesdayThoughts' , 
                 'FalconHeavy' , 
                 'iHeartAwards' ,
                 'MondayMotivation' , 
                 'SundayFunday' ,  
                 'FridayReads' , 
                 'Trump' , 
                 'ThursdayThoughts',
                 'FinalFour',
                 'PressforProgress',
                 'SuperBowl' ,
                 'BreakingNews'
                 ] 


df_topics_list = pd.DataFrame(index=test_hashtags , columns=['NMF' ,'LDA' , 'GLDA' , 'GLDA_SEED'])
df_topics_cohh = pd.DataFrame(index=test_hashtags , columns=['NMF' ,'LDA' , 'GLDA' , 'GLDA_SEED'])
df_topics_list = df_topics_list.astype('object')


    
for hashtag in test_hashtags:
    temp = df_tweet.loc[df_tweet['hastag'] == hashtag]
    title , list_df = grouping_date_sentiment(temp)

    flat_list = flaten_my_list(list_df)
    dictionary = Dictionary(list_df)
    
    #calling count vectorizer
    tf_hashtag , tf_feature_names_hashtag = get_count_vectorizer(flat_list , no_features )
    
    #for LDA
    try:
        lda_topic , lda_coherence = get_topics_using_LDA(no_topics , tf_hashtag , tf_feature_names_hashtag , no_top_words , list_df , dictionary )
        df_topics_list.loc[hashtag , 'LDA'] = lda_topic
        df_topics_cohh.loc[hashtag , 'LDA'] = lda_coherence
    except:
        pass
    
    #for nmf
    try:
        nmf_topic , nmf_coherence = get_topics_using_NMF(no_topics , tf_hashtag , tf_feature_names_hashtag , no_top_words , list_df , dictionary )
        df_topics_list.loc[hashtag , 'NMF'] = nmf_topic
        df_topics_cohh.loc[hashtag , 'NMF'] = nmf_coherence
    except:
        pass
    
    #get_topics_GLDA_without_seed
    try:
        glda_topic , glda_coherence = get_topics_GLDA_without_seed(no_topics , tf_hashtag , tf_feature_names_hashtag , no_top_words , list_df , dictionary )
        df_topics_list.loc[hashtag , 'GLDA'] = glda_topic
        df_topics_cohh.loc[hashtag , 'GLDA'] = glda_coherence
    except:
        pass
    
    
    #get_topics_GLDA_with_seed
    try:
        gldSeed_topic , gldSeed_coherence = get_topics_GLDA_with_seed(no_topics , tf_hashtag , tf_feature_names_hashtag , no_top_words , list_df , dictionary )
        df_topics_list.loc[hashtag , 'GLDA_SEED'] = gldSeed_topic
        df_topics_cohh.loc[hashtag , 'GLDA_SEED'] = gldSeed_coherence
    except:
        pass
    
    print('The below Result are for {} '.format(hashtag))
    print('Coherence Score for LDA {} '.format(lda_coherence))
    print('Coherence Score for nmf {} '.format(nmf_coherence))
    print('Coherence Score for glda without seed {} '.format(glda_coherence))
    print('Coherence Score for glda with seed {} '.format(gldSeed_coherence))
    print()
    print()    
    


result_model = ['LDA' , 'GLDA']
result_hastag = ['Trump' , 'HolocaustMemorialDay' ,  'ThursdayThoughts' , 'PressforProgress']


for hashtag in result_hastag :
    print('... WordCloud Token Visualisation for {} ....'.format(hashtag))
    
    
    list_topics = df_topics_list.loc[hashtag]
    for model in result_model:
        try:
            print('... WordCloud Token Visualisation for {} ....'.format(model))
            data = wordCloudInput_result(list_topics[model])
            makeImage(getFrequencyDictForText(data))
            get_average_similirarity_matrix(hashtag , list_topics[model] , model)
        except:
            pass
            

#==============================================================================
#              
# #==============================================================================
# #     plt.hist(list_count, normed=True, bins=100)
# #     plt.title(hashtag + " "+ model )
# #     plt.xlabel('wup_Similarity')
# #     plt.ylabel('Count')
# #     
# #     plt.show()
# #==============================================================================
#     
#         
# 
# 
# 
# def visualise_result(data , hashtag , model):    
#     plt.hist(temp1, normed=True, bins=40)
#     plt.title(hashtag + " "+ model )
#     plt.xlabel('wup_Similarity')
#     plt.ylabel('Count')
#     
#     plt.show()
# 
# 
# 
# 
# def visualise_result(data , hashtag , model):    
#     plt.hist(temp1, normed=True, bins=40)
#     plt.title(hashtag + " "+ model )
#     plt.xlabel('wup_Similarity')
#     plt.ylabel('Count')
#     
#     plt.show()
# 
# 
# 
# 
# 
# 
# 
#     
#     
#     temp1 = wordCloudInput(temp)
#     makeImage(getFrequencyDictForText(temp1))
# 
# a = df_topics_list.loc[hashtag]['NMF']
# 
# 
# 
# import numpy as np
# import scipy.stats as stats
# import pylab as pl
# 
# h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
#      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
#      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted
# 
# fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
# 
# pl.plot(h,fit,'-o')
# 
# pl.hist(h,normed=True)      #use this to draw histogram of your data
# 
# pl.show()
# 
# 
# 
# 
# 
#         
#         
#    for word1 in list1:
#     for word2 in list2:
#         wordFromList1 = wordnet.synsets(word1)
#         wordFromList2 = wordnet.synsets(word2)
#         if wordFromList1 and wordFromList2: #Thanks to @alexis' note
#             s = wordFromList1[0].wup_similarity(wordFromList2[0])
#             if(s != 'None'):
#                 list.append((word1 , word2 , s))
#             else:
#                 
# 
# 
# 
# test_hashtags = ['HolocaustMemorialDay',
#                  'Obama', 
#                  'TuesdayThoughts' , 
#                  'FalconHeavy' , 
#                  'iHeartAwards' ,
#                  'MondayMotivation' , 
#                  'SundayFunday' ,  
#                  'FridayReads' , 
#                  'Trump' , 
#                  'ThursdayThoughts',
#                  'FinalFour',
#                  'PressforProgress',
#                  'SuperBowl' ,
#                  'BreakingNews'
#                  ] 
# 
# 
# 
# 
# 
#     
# gda_model = guidedlda.GuidedLDA(n_topics=no_topics, n_iter=200, random_state=1, refresh=50).fit(tf_hashtag)
# display_topics(gda_model ,  tf_feature_names_hashtag, no_top_words)
# gda_topic = get_list_of_topics(gda_model, tf_feature_names_hashtag, no_top_words)
# 
# 
# CoherenceModel(topics=gda_topic, texts=list_df, dictionary=dictionary, window_size=100).get_coherence()
# 
# 
# 
# 
# def get_topics_GLDA_without_seed(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
#     gda_wthout_seed = guidedlda.GuidedLDA(n_topics=no_topics_xx, n_iter=500, random_state=1, refresh=100).fit(tf_xx)
#     #display_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
#     gda_topic = get_list_of_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
#     gda_wthout_seed_coherence = CoherenceModel(topics=gda_wthout_seed, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
#     return gda_topic , gda_wthout_seed_coherence
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# #==============================================================================
# # 
# # 
# # 
# # title , list_df = grouping_date_sentiment(df_iheart)
# # 
# # 
# # no_topics = 10
# # no_top_words = 15
# # 
# # flat_list = flaten_my_list(list_df)
# # dictionary = Dictionary(list_df)   
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # def get_count_vectorizer(flat_list_xx , no_features_xx ):
# #     tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features_xx, stop_words='english')
# #     tf = tf_vectorizer.fit_transform(flat_list_xx)
# #     tf_feature_names = tf_vectorizer.get_feature_names()
# #     return tf , tf_feature_names
# # 
# # 
# # tf , tf_feature_names = get_count_vectorizer(flat_list , no_features )
# # 
# # 
# # 
# # 
# # 
# # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# # tf = tf_vectorizer.fit_transform(flat_list)
# # tf_feature_names = tf_vectorizer.get_feature_names()
# # 
# # tf , tf_feature_names = get_count_vectorizer()
# # 
# # 
# # 
# # 
# # #==============================================================================
# # # Function returns the seed list for for the Guided LDA topic modelling algorithm
# # #==============================================================================
# # def get_seed_list(flat_list_xx):
# #     tagged_sent = []
# #     for tags in flat_list_xx:
# #         tagged_sent.extend(tag.pos_tag(tags.split(sep = " ")))
# #     
# #     seed_topic_list =  list(set(word for word, tag in tagged_sent if tag in ('NN')))
# #     seed_topic_list = get_right_words(seed_topic_list)
# #     shuffle(seed_topic_list)
# #     cnt = round(len(seed_topic_list)/10)
# #     chunks_123 = [seed_topic_list[x:x+cnt] for x in range(0, len(seed_topic_list), cnt)]
# #     
# #     Vocab = []
# #     for line in flat_list_xx:
# #         Vocab.extend(line.split(sep = " ")) 
# #     word2id = dict((v, idx) for idx, v in enumerate(Vocab))
# #     seed_topics = {}
# #     for t_id, st in enumerate(chunks_123):
# #         for word in st:
# #             seed_topics[word2id[word]] = t_id
# #     
# #     return seed_topics
# # 
# # #==============================================================================
# # # Function return the modelled topics with gthe coherence value for
# # # a corupus using Latent Dirichlet Allocation (LDA)
# # #==============================================================================
# # def get_topics_using_LDA(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
# #     lda = LatentDirichletAllocation(n_topics=no_topics_xx, max_iter=500, learning_method='online', learning_offset=50.,random_state=1).fit(tf_xx)
# #     #display_topics(lda, tf_feature_names_xx, no_top_words_xx)
# #     lda_topic = get_list_of_topics(lda, tf_feature_names_xx, no_top_words_xx)
# #     lda_coherence = CoherenceModel(topics=lda_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
# #     return lda_topic , lda_coherence
# # 
# # 
# # #==============================================================================
# # # Function return the modelled topics with gthe coherence value for
# # # a corupus using Non-negative Matrix Factorisation (NMF)
# # #==============================================================================
# # def get_topics_using_NMF(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
# #     nmf = NMF(n_components=no_topics_xx, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf_xx)
# #     #display_topics(nmf, tf_feature_names_xx , no_top_words_xx)
# #     nmf_topic = get_list_of_topics(nmf, tf_feature_names_xx , no_top_words_xx)
# #     nmf_coherence = CoherenceModel(topics=nmf_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
# #     return nmf_topic , nmf_coherence
# # 
# # 
# # #==============================================================================
# # # Function return the modelled topics with gthe coherence value for
# # # a corupus using semi supervised Guided LDA without seeding
# # #==============================================================================
# # def get_topics_GLDA_without_seed(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
# #     gda_wthout_seed = guidedlda.GuidedLDA(n_topics=no_topics_xx, n_iter=500, random_state=1, refresh=100).fit(tf_xx)
# #     #display_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
# #     gda_topic = get_list_of_topics(gda_wthout_seed ,  tf_feature_names_xx, no_top_words_xx)
# #     gda_wthout_seed_coherence = CoherenceModel(topics=gda_wthout_seed, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
# #     return gda_topic , gda_wthout_seed_coherence
# # 
# # #==============================================================================
# # # Function return the modelled topics with gthe coherence value for
# # # a corupus using semi supervised Guided LDA with seed topics
# # #==============================================================================
# # def get_topics_GLDA_with_seed(no_topics_xx , tf_xx , tf_feature_names_xx , no_top_words_xx , list_df_xx , dictionary_xx ):
# #     flat_list = flaten_my_list(list_df_xx)
# #     seed_topics = get_seed_list(flat_list)
# #     glda_with_seed = guidedlda.GuidedLDA(n_topics=no_topics_xx, n_iter=500, random_state=1, refresh=100)
# #     glda_with_seed.fit(tf_xx, seed_topics=seed_topics, seed_confidence=0.15)   
# #     #display_topics(glda_with_seed ,  tf_feature_names_xx, no_top_words_xx)
# #     glda_topic = get_list_of_topics(glda_with_seed ,  tf_feature_names_xx, no_top_words_xx)
# #     glda_wth_seed_coherence = CoherenceModel(topics=glda_topic, texts=list_df_xx , dictionary=dictionary_xx, window_size=100).get_coherence()
# #     return glda_topic , glda_wth_seed_coherence
# # 
# # 
# # 
# # 
# # 
# # 
# #==============================================================================
# 
# 
# 
# 
#==============================================================================
