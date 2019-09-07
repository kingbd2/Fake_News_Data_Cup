# NLTK 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re

### GENERAL FUNCTIONS ###
# General function to create new column named column_name in a dataframe (df) after running function on column in df
def create_append_feature(df, new_column_name, function, column):
    df[str(new_column_name)] = df.apply(lambda x: function(x[column]), axis=1)

def split_featureset():
    return

def export_data():
    return

### DATA PREPARATION FUNCTIONS ###
# Filter text to remove punctuation and stopwords
def remove_stopwords(text):
    stop_words = list(set(stopwords.words('english')))
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text)
    return [w for w in text if not w in stop_words]

# Get word count of text with stopwords removed
def get_word_count(text):
    word_list = remove_stopwords(text)
    return len(word_list)

# Get token count of tokenized text
def get_token_count(text):
    return len(word_tokenize(text))

def get_brevity_score(text):
    word_count_no_stopwords = get_word_count(text)
    token_count = get_token_count(text)
    return (word_count_no_stopwords / token_count)

def get_sentiment_nltk_vader(text):
    sentence = ' '.join(word for word in text)
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    return ss['pos'], ss['neg'], ss['neu'], ss['compound']