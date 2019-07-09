# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:03:33 2018

@author: Olashile Adebimpe
"""

import os
import sys
import math
import random
import time
import csv
import numpy as np
import pandas as pd

#==============================================================================
# Importing the twitter API
#==============================================================================
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

#==============================================================================
# Authenticathion codes
#==============================================================================
consumer_key = 'pu9KOFEEWkpOAaHCypQaxy4gv'
consumer_secret = 'vBR3970CqwEfmT8F6KdrdmbRbrbuKmJdER9b0RO5XW9ktYtxv3'
access_token = '209622320-1kXbHfZ6xOaJ95XAhmj6kqHP2XFDT3OHeXt02U7y'
access_secret = 'nwho49ViUw49jtlQ5oh7FLM4bnkR1mD87O8K1a9fHE0ih'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth,wait_on_rate_limit=True)

#==============================================================================
# End of Twitter Authetication.
#==============================================================================


# to get the current  absolute path where this python program is running from
root_dir = os.path.abspath('.')

#to get the path that has the raw input data file
data_dir = os.path.join(root_dir, 'data')

#ThursdayThoughts
#Olympics2018

#list of all hastages to get daily 
#==============================================================================
# hastag_list =  [  'SaferInternetDay' , 'GetMeHotIn4Words' ,
#                 'MondayMotivaton' ,'TuesdayThoughts' , 
#                 'FalconHeavy' , 'SpaceX', 'WhyImSingle' ,
#                 'TheLaunch' , 'BCWineChat' , 'CBBUS' , 'iHeartAwards' , 'SuperBowl' ,]
#==============================================================================
hastag_list =  [str(sys.argv[1])]

#  cd .\Python\Projects\AdvanceNLP\  python Tweets_Extraction.py WednesdayWisdom


#[ 'oscars2018' , 'Oscars' , 'CanadaSupportsScience' , 'Airplane' , 'HopeWorld' ]


 # ['Grammys2018' ,'BreakingNews' , 'HolocaustMemorialDay' , 'BBNaija']
print("extracting tweet for"+ str(hastag_list))
    

def get_tweets(hastag):
    csvFile = open(os.path.join(data_dir , time.strftime("%Y%m%d_")+ hastag+ ".csv") , 'a')
    
    csvWriter = csv.writer(csvFile)
    count = 0 ;
    print ("Getting tweets for " + hastag )
    for tweet in tweepy.Cursor(api.search,q="#"+hastag+ " -filter:retweets", #count=10000,
                           lang="en",
                           until= "2018-03-25",
                           since = "2018-02-01"
                           ).items():
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
        count = count+1
        
        print(str(count) + ": "+ str(tweet.text.encode('utf-8')))
    
    
    print ("Total Tweets saved for #" + str(hastag) + " is " + str(count) )  
       
        
 # gets all tweets and save them to a folder and csv.       
for each_tags in  hastag_list:
    get_tweets(each_tags)














#==============================================================================
# 
# #####United Airlines
# # Open/Create a file to append data
# csvFile = open(os.path.join(data_dir , time.strftime("%Y%m%d")+"BreakingNews.csv") , 'a')
# #Use csv Writer
# csvWriter = csv.writer(csvFile)
# 
# for tweet in tweepy.Cursor(api.search,q="#BreakingNews", #count=10000,
#                            lang="en",
#                            since="2018-01-01").items():
#     #print (tweet.created_at, tweet.text)
#     csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
#==============================================================================
    




    
    

#==============================================================================
# we can read our own timeline (i.e. our Twitter homepage) with
#==============================================================================
#==============================================================================
# for status in tweepy.Cursor(api.home_timeline).items(10):
#     # Process a single status
#     print(status.text)
#==============================================================================


#==============================================================================
# The below code gathers all the new tweets about the provided has tags
#==============================================================================
#==============================================================================
# class MyListener(StreamListener):
#  
#     def on_data(self, data):
#         try:
#             with open('python.json', 'a') as f:
#                 f.write(data)
#                 return True
#         except BaseException as e:
#             print("Error on_data: %s" % str(e))
#         return True
#  
#     def on_error(self, status):
#         print(status)
#         return True
#  
# twitter_stream = Stream(auth, MyListener())
# twitter_stream.filter(track=['#python'])
# 
#==============================================================================






