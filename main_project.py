#Required libraries

# plotly, tweepy, pyspellchecker, nltk,pickle,collections,pillow,textblob


import tweepy
from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import re
import string
import itertools
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os 
import pickle
import plotly.figure_factory as ff
import glob
import json
from string import punctuation
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
   
ps = PorterStemmer()
spell = SpellChecker()

with open("contraction_dict.pickle", 'rb') as handle:
        contraction_dict = pickle.load(handle)    

with open('abbr.json','r') as file:
    abbr_dict=json.loads(file.read())
    
    
    
spelling_corrected=[]

def remove_mentions(txt):
    word_tokens = word_tokenize(txt)
    filtered_sentence=[]
    for word in word_tokens:
        if "@" not in word:
            filtered_sentence.append(word)
    return " ".join(filtered_sentence)
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", txt).split())

stop_words = set(stopwords.words('english')) 


def remove_stop_words(txt):
    word_tokens = word_tokenize(txt)     
    filtered_sentence=[]
    for word in word_tokens:
            if not word in stop_words:
                filtered_sentence.append(word)
    return " ".join(filtered_sentence)

def remove_custom_words(txt):
    
    word_tokens = word_tokenize(txt) 
    custom_words=["rt","COVID19","pfizer","pfizerbiontech","moderna","janssen","https","j","co","amp"]
    filtered_sentence=[]
    for word in word_tokens:
        if not word in custom_words:
            filtered_sentence.append(word)
    return " ".join(filtered_sentence)

def spell_checker(txt):    
    word_tokens = word_tokenize(txt)  
    temp=""
    for word in word_tokens:
        if word in english_vocab:
            temp+=" "+ word
        else:
            if word not in spelling_corrected:
                temp+=" "+ spell.correction(word)
                spelling_corrected.append(word)
    return temp

def stem(txt):    
    word_tokens = word_tokenize(txt) 
    word_tokens=[ps.stem(word) for word in word_tokens]    
    return " ".join(word_tokens)

def remove_numbers_puncs(txt):
    puncsets = set(punctuation)
    word_tokens = word_tokenize(txt)
    word_tokens=[word for word in word_tokens if word not in puncsets]
    
    filtered_sentence=[]
    for word in word_tokens:
        if not word.isdigit():
            filtered_sentence.append(word)
    return " ".join(filtered_sentence)

def expand_contractions(line):
    line_split=line.split()
    for i in line.split():
        if i in contraction_dict:            
            line_split[line_split.index(i)]=contraction_dict[i]
        if i in abbr_dict :
            
            line_split[line_split.index(i)]=abbr_dict[i]
    return ' '.join(line_split)



def tweet_sentiment(tweet):
    analysis= TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    subjectivity=analysis.sentiment.subjectivity
    polarity_label=""
    if analysis.sentiment.polarity > 0:
            polarity_label= 'positive'
    elif analysis.sentiment.polarity == 0:
            polarity_label = 'neutral'
    else:
            polarity_label = 'negative'
            
    return polarity,polarity_label,subjectivity
def create_dataframe(tweetlist):
    polarity=[]
    polarity_label=[]
    subjectivity=[]
    for tweet in tweetlist:
        temp_polarity,temp_polarity_label,temp_subjectivity=tweet_sentiment(tweet)
        polarity.append(temp_polarity)
        polarity_label.append(temp_polarity_label)
        subjectivity.append(temp_subjectivity)
    df = pd.DataFrame(list(zip(tweetlist, polarity,polarity_label,subjectivity)),
                   columns =['Tweet', 'polarity','polarity_label','subjectivity'])
    return df


def create_pie_chart_sentiment(final_df,name):
    sum_positive_sentiment=len(final_df.loc[final_df["polarity"]>0])
    sum_negative_sentiment=len(final_df.loc[final_df["polarity"]<0])
    sum_neutral_sentiment=len(final_df.loc[final_df["polarity"]==0])
    total_sentiment=len(final_df)

    percentage_positive_sentiment=round(sum_positive_sentiment/total_sentiment * 100,2)
    percentage_negative_sentiment=round(sum_negative_sentiment/total_sentiment * 100,2)
    percentage_neutral_sentiment=round(sum_neutral_sentiment/total_sentiment * 100,2)


    percentage_df=pd.DataFrame()
    percentage_df["Labels"]=["Positive","Negative","Neutral"]
    percentage_df["Percentage"]=[percentage_positive_sentiment,percentage_negative_sentiment,percentage_neutral_sentiment]


    fig = go.Figure(data=[go.Pie(labels=["Positive","Negative","Neutral"]
                                 , values=[percentage_positive_sentiment,percentage_negative_sentiment,percentage_neutral_sentiment], 
                                 textinfo='label+percent',
                                 insidetextorientation='radial',showlegend=False
                                )])

    fig.update_layout(title="Pie Chart of polarity of tweets")
    fig.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
    fig.write_image("results/"+name+"_sentiment.jpeg")
    return percentage_positive_sentiment,percentage_negative_sentiment,percentage_neutral_sentiment
    
def create_pie_chart_subjectivity(final_df,name):
    sum_obj_subjectivity=len(final_df.loc[final_df["subjectivity"]==0])
    sum_sub_subjectivity=len(final_df.loc[final_df["subjectivity"]==1])
    sum_neutral_subjectivity=len(final_df)-(sum_obj_subjectivity+sum_sub_subjectivity)
    total_subjectivity=len(final_df)

    percentage_obj_subjectivity=round(sum_obj_subjectivity/total_subjectivity * 100,2)
    percentage_sub_subjectivity=round(sum_sub_subjectivity/total_subjectivity * 100,2)
    percentage_neutral_subjectivity=round(sum_neutral_subjectivity/total_subjectivity * 100,2)

    percentage_df=pd.DataFrame()
    percentage_df["Labels"]=["Objective","Subjective","Neutral"]
    percentage_df["Percentage"]=[percentage_obj_subjectivity,percentage_sub_subjectivity,percentage_neutral_subjectivity]
    

    fig = go.Figure(data=[go.Pie(labels=["Objective","Subjective","Neutral"], 
                                 values=[percentage_obj_subjectivity,percentage_sub_subjectivity,percentage_neutral_subjectivity], 
                                 textinfo='label+percent',
                                 insidetextorientation='radial',showlegend=False
                                )])
    fig.update_layout(title="Pie Chart of Subjectivity of tweets")
    fig.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
    fig.write_image("results/"+name+"_subjectivity.jpeg")
    return percentage_sub_subjectivity,percentage_obj_subjectivity,percentage_neutral_subjectivity


def normalization(x):
    return (x-min(x))/(max(x)-min(x))

def create_wordcloud(text,name):
    mask = np.array(Image.new("RGB",size=(800,800)))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    mask = mask,
    max_words=5000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(str(text))
    wc.to_file("results/"+name+"wordcloud.jpg")
    
    
total_dict=[]
with open("data/tweets.pickle", 'rb') as handle:
    total_dict = pickle.load(handle)
retrieved_tweets_list=[]
retrieved_tweets_list_no_stem=[]
retrieved_tweets_dict={"COVID-19 vaccine":[],"Pfizer-BioNTech":[],"Moderna":[],"Janssen":[]}
for items in total_dict:
    for item in items:
        tweets=list(set(items[item]))
        processed_items=[remove_mentions(tweet) for tweet in tweets]
        processed_items=[remove_url(tweet) for tweet in processed_items]           
        processed_items=[" ".join(tweet.lower().split()) for tweet in processed_items]            
        processed_items=[remove_numbers_puncs(tweet) for tweet in processed_items]            
        processed_items=[remove_stop_words(tweet) for tweet in processed_items]            
        processed_items=[remove_custom_words(tweet) for tweet in processed_items]            
        processed_items=[expand_contractions(tweet) for tweet in processed_items]
# I have commented this spellchecker for submission as this is a time consuming process
# However, for the results I used this module
#You can uncomment the below line to run the spellchecker module
#         processed_items=[spell_checker(tweet) for tweet in processed_items]           
        processed_items_stem=[stem(tweet) for tweet in processed_items]
        retrieved_tweets_list.extend(processed_items_stem)
        retrieved_tweets_list_no_stem.extend(processed_items)
        retrieved_tweets_dict[item].extend(processed_items_stem)
retrieved_tweets_list=list(set(retrieved_tweets_list))
retrieved_tweets_list_no_stem=list(set(retrieved_tweets_list_no_stem))
retrieved_tweets_dict={k:list(set(j)) for k,j in retrieved_tweets_dict.items()}


# this code is to get retrieve the tweets from Tweeter using Tweepy api
# I have commented this out as I have already stored the tweets in tweets.pickle and used that for the analysis
# you can run this code if you want to retrrive the tweets yourself
# consumerKey = "vbL0UYzGXU2CNBVsWzaCA6GGF"
# consumerSecret = "ssbldkt84qh9OpmdHCSz86HP9glqWfQH1nyFoGrQzdijrHvxeh"
# accessToken = "253084088-Vc4D5TEfuYlT0z2sIM5JWAkVsrp4mjmyqkt7Cjia"
# accessTokenSecret = "3arkG44HU8wc0QQpsvhvIkUwG77ktHDO3w0q0CsTKrlXo"
# auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
# auth.set_access_token(accessToken, accessTokenSecret)
# api = tweepy.API(auth,wait_on_rate_limit=True)
# keywords=["COVID-19","Pfizer-BioNTech","Moderna","Janssen"]
# total_counts=50

# retrieved_tweets_list=[]
# retrieved_tweets_list_no_stem=[]
# retrieved_tweets_dict={"COVID-19":[],"Pfizer-BioNTech":[],"Moderna":[],"Janssen":[]}
# for keyword in keywords:
#     temp=[]
#     tweet_list=[]
#     tweets=tweepy.Cursor(api.search, q=keyword,lang="en").items(total_counts)
#     tweet_list=[tweet.text for tweet in tweets if not tweet.retweeted]
#     processed_items=[remove_url(tweet) for tweet in tweet_list]           
#     processed_items=[" ".join(tweet.lower().split()) for tweet in processed_items]            
#     processed_items=[remove_numbers_puncs(tweet) for tweet in processed_items]            
#     processed_items=[remove_stop_words(tweet) for tweet in processed_items]            
#     processed_items=[remove_custom_words(tweet) for tweet in processed_items]            
#     processed_items=[expand_contractions(tweet) for tweet in processed_items]            
#     processed_items=[spell_checker(tweet) for tweet in processed_items]           
#     processed_items_stem=[stem(tweet) for tweet in processed_items]
#     retrieved_tweets_list.extend(processed_items_stem)
#     retrieved_tweets_list_no_stem.extend(processed_items)
#     retrieved_tweets_dict[item].extend(processed_items_stem)
# retrieved_tweets_list=list(set(retrieved_tweets_list))
# retrieved_tweets_list_no_stem=list(set(retrieved_tweets_list_no_stem))
# retrieved_tweets_dict={k:list(set(j)) for k,j in retrieved_tweets_dict.items()}

df_total=create_dataframe(retrieved_tweets_list)
df_total_not_stemmed=create_dataframe(retrieved_tweets_list_no_stem)
df_covid=create_dataframe(retrieved_tweets_dict["COVID-19 vaccine"])
df_pfizer=create_dataframe(retrieved_tweets_dict["Pfizer-BioNTech"])
df_moderna=create_dataframe(retrieved_tweets_dict["Moderna"])
df_janssen=create_dataframe(retrieved_tweets_dict["Janssen"])

pos_total,neg_total,neu_total=create_pie_chart_sentiment(df_total,"total")
pos_covid,neg_covid,neu_covid=create_pie_chart_sentiment(df_covid,"covid")
pos_pfizer,neg_pfizer,neu_pfizer=create_pie_chart_sentiment(df_pfizer,"pfizer")
pos_moderna,neg_moderna,neu_moderna=create_pie_chart_sentiment(df_moderna,"moderna")
pos_janssen,neg_janssen,neu_janssen=create_pie_chart_sentiment(df_janssen,"janssen")

sub_total,obj_total,neu1_total=create_pie_chart_subjectivity(df_total,"total")
sub_covid,obj_covid,neu1_covid=create_pie_chart_subjectivity(df_covid,"covid")
sub_pfizer,obj_pfizer,neu1_pfizer=create_pie_chart_subjectivity(df_pfizer,"pfizer")
sub_moderna,obj_moderna,neu1_moderna=create_pie_chart_subjectivity(df_moderna,"moderna")
sub_janssen,obj_janssen,neu1_janssen=create_pie_chart_subjectivity(df_janssen,"janssen")


create_wordcloud(df_total["Tweet"].values,"total")
create_wordcloud(df_total.loc[df_total["polarity"]>0]["Tweet"].values,"positive")
create_wordcloud(df_total.loc[df_total["polarity"]==0]["Tweet"].values,"neutral")
create_wordcloud(df_total.loc[df_total["polarity"]<0]["Tweet"].values,"negative")

final_df_no_zero_polarity=df_total.loc[df_total["polarity"]!=0]
hist_data = [normalization(final_df_no_zero_polarity["polarity"].values), 
             normalization(df_total["subjectivity"].values)]

group_labels = ['Polarity', 'Subjectivity']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
fig.update_layout(title="Distribution of polarity and subjectivity values for the tweets (Min-Max Normalized)")
fig.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
# fig.show()
fig.write_image("results/hist.jpeg")


labels=["Combined","COVID-19 vaccine","Pfizer-BioNTech","Moderna","Janssen"]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=labels, y=[obj_total,obj_covid,obj_pfizer,obj_moderna,obj_janssen],text=[obj_total,obj_covid,obj_pfizer,obj_moderna,obj_janssen],textposition='auto',),
    go.Bar(name='Neutral', x=labels, y=[neu_total,neu_covid,neu_pfizer,neu_moderna,neu_janssen],text=[neu_total,neu_covid,neu_pfizer,neu_moderna,neu_janssen],textposition='auto',),
    go.Bar(name='Negative', x=labels, y=[neg_total,neg_covid,neg_pfizer,neg_moderna,neg_janssen],text=[neg_total,neg_covid,neg_pfizer,neg_moderna,neg_janssen],textposition='auto',)
])

fig.update_layout(barmode='group')
fig.update_layout(bargap=0.5)

fig.update_layout(title="Bar Chart of polarity of tweets")
fig.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
# fig.show()
fig.write_image("results/bar1.jpeg")


fig1 = go.Figure(data=[
    go.Bar(name='Objectivity', x=labels, y=[obj_total,obj_covid,obj_pfizer,obj_moderna,obj_janssen],text=[obj_total,obj_covid,obj_pfizer,obj_moderna,obj_janssen],textposition='auto',),
    go.Bar(name='Neutral', x=labels, y=[neu1_total,neu1_covid,neu1_pfizer,neu1_moderna,neu1_janssen],text=[neu1_total,neu1_covid,neu1_pfizer,neu1_moderna,neu1_janssen],textposition='auto',),
    go.Bar(name='Subjectivity', x=labels, y=[sub_total,sub_covid,sub_pfizer,sub_moderna,sub_janssen],text=[sub_total,sub_covid,sub_pfizer,sub_moderna,sub_janssen],textposition='auto',)
])

fig1.update_layout(barmode='group')
fig1.update_layout(bargap=0.5)

# fig1.show()
fig1.update_layout(title="Bar Chart of subjectivity of tweets")
fig1.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig1.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
# fig1.show()
fig1.write_image("results/bar2.jpeg")


words_in_tweet = [tweet.lower().split() for tweet in df_total_not_stemmed["Tweet"].values]
# words_in_tweet[:2]
all_words_no_urls = list(itertools.chain(*words_in_tweet))
counts_no_urls = collections.Counter(all_words_no_urls)
clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(20),
                             columns=['words', 'count'])
clean_tweets_no_urls=clean_tweets_no_urls.sort_values(["count"])
fig=px.bar(clean_tweets_no_urls,x="count",y="words",orientation='h',text="count")
fig.update_layout(bargap=0.6)
fig.update_layout(title="20 Most Frequent Words")
fig.update_yaxes(title="")
fig.update_layout(
    title={
        
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout(
    font_family="Arial",
    font_color="Red",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
# fig.show()

fig.write_image("results/freqwords.jpeg")


from nrclex import NRCLex
text_object = NRCLex(retrieved_tweets_list[0])
emotion_list=[]
for tweet in retrieved_tweets_list:
    text_object = NRCLex(tweet)
    emotion_list.extend(text_object.affect_list)
emotion_list_processed=[]
for emotion in emotion_list:
    if emotion != "positive" and emotion !="negative":
        emotion_list_processed.append(emotion)
from collections import Counter
a=dict(Counter(emotion_list_processed))
a={k: v for k, v in sorted(a.items(), key=lambda item: item[1])}
count=[]
emotions=[]
for k in a :
    emotions.append(k)
    count.append(a[k])


fig = go.Figure(go.Pie(
            values=count,
            labels=emotions,sort=False,
     textinfo='label+percent',
     hole=.3,showlegend=False
            ))

fig.update_layout(title="Pie Chart of NRC Emotions")
fig.update_layout(
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout(
    font_family="Arial",
    font_color="blue",
    font_size=13,
    title_font_family="Times New Roman",
    title_font_color="Black",
    legend_title_font_color="green"
)
# fig.show()
fig.write_image("results/emotions.jpeg")