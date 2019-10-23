# NLTK 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Spacy 
import spacy
import en_core_web_sm
import en_core_web_md
    
# Other packages
import re
import pandas as pd
import numpy as np
import json

### GENERAL FUNCTIONS ###
# General function to create new column named column_name in a dataframe (df) after running function on column in df
def create_append_feature(source_df, source_df_column, new_column_name, applied_function):
    column_list = []
    for text in source_df[source_df_column]:
        column_list.append(applied_function(text))
    source_df[new_column_name] = column_list
    return source_df

def load_train_data(json_file_path, output="list"):
    with open(json_file_path) as f:
        train_data = json.load(f)
    if output == "dataframe":
        train_df = pd.DataFrame.from_records(train_data)
        return train_df
    else:
        return train_data

def load_labels(df):
    return df


def split_featureset():
    print("hi")
    return

def export_data():
    return

### DATA PREPARATION FUNCTIONS ###
# Filter text to remove punctuation and stopwords
def remove_stopwords(text):
    stop_words = list(set(stopwords.words('english')))
    text_no_punc = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text_no_punc)
    return [w for w in text_tokens if not w in stop_words]
    
# Get word count of text with stopwords removed
def get_word_count(text):
    word_list = remove_stopwords(text)
    return len(word_list)

# Get token count of tokenized text
def get_token_count(text):
    return len(word_tokenize(text))

# Get sentence count of tokenized text
def get_sentence_count(text):
    sent_tokenize_list = sent_tokenize(text)
    return len(sent_tokenize_list)

def get_brevity_score(text):
    word_count_no_stopwords = get_word_count(text)
    token_count = get_token_count(text)
    return (word_count_no_stopwords / token_count)

def get_sentiment_nltk_vader(text):
    sid = SentimentIntensityAnalyzer()
    tokens_list = word_tokenize(text)
    sentence = ' '.join(word for word in tokens_list)
    # print(sentence)
    ss = sid.polarity_scores(sentence)
    return [ss['pos'], ss['neg'], ss['neu'], ss['compound']]


class spacy_functions:
    nlp_small = en_core_web_sm.load()
    nlp_medium = en_core_web_md.load()
    def __init__(self, text):
        self.text_small = self.nlp_small(text)
        self.text_med = self.nlp_medium(text)

    def tokenize(self):
        for token in self.text_small:
            print(token.text)
    # Text: The original word text.
    # Lemma: The base form of the word.
    # POS: The simple part-of-speech tag.
    # Tag: The detailed part-of-speech tag.
    # Dep: Syntactic dependency, i.e. the relation between tokens.
    # Shape: The word shape – capitalization, punctuation, digits.
    # is alpha: Is the token an alpha character?
    # is stop: Is the token part of a stop list, i.e. the most common words of the language?
    
    # https://spacy.io/api/annotation#pos-tagging
    def annotate(self):
        for token in self.text_small:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

    # https://spacy.io/api/annotation#named-entities
    def entities(self):
        for ent in self.text_small.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

    # Text: The original token text.
    # has vector: Does the token have a vector representation?
    # Vector norm: The L2 norm of the token’s vector (the square root of the sum of the values squared)
    # OOV: Out-of-vocabulary
    def word_similarities(self):
        for token in self.text_med:
            print(token.text, token.has_vector, token.vector_norm, token.is_oov)

def combine_texts(train_df, article_df):
    for i, (claim) in enumerate(zip(train_df.related_articles)):
        texts=[]
        for k, article_id in enumerate(claim[0]):
            target = article_df.loc[article_df['id'] == article_id]
            if target.empty:
                continue
            texts.append(target['text'].values[0])
            text_list = list(texts)
        text_compiled = ''.join(text_list)
        train_df.iloc[i]['texts_compiled'] = text_compiled


def create_article_metric (train_df, article_df, applied_summarization_function, new_column_name='new_column',  include_var=True):
    feature_mean = []
    feature_var = []
    for i, (claim) in enumerate(zip(train_df.related_articles)):
        feature_ = np.zeros([len(claim[0]), 1])
        for k, article_id in enumerate(claim[0]):
            target = article_df.loc[article_df['id'] == article_id]
            if target.empty:
                continue
            # Need to replace brevity score with abstracted target measure in article_df
            feature_[k-1, 0] = target[new_column_name].values
        feature_mean.append(np.mean(feature_))
        feature_var.append(np.var(feature_))
    if include_var:
        train_df['feature_mean'] = feature_mean
        train_df['feature_var'] = feature_var
        return train_df
    else:
        train_df['feature_mean'] = feature_mean
        return train_df