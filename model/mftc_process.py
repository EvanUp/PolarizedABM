## This is a messy script used to calculate correlations, sentiment, and train the logistic regression

import pandas as pd 
import numpy as np
import networkx as nx
from ast import literal_eval
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('data/mftc_final.csv').drop(columns='Unnamed: 0')
df = df[['tweet_id', 'tweet_text', 'annotations']]
df['annots'] = df.apply(lambda x: literal_eval(x['annotations']), axis = 1)
annotations = df.annots.tolist()

def reset_mc():
    moral_counter = {'purity':0, 
                    'authority':0, 
                    'fairness':0, 
                    'degradation':0, 
                    'care':0, 
                    'loyalty':0, 
                    'nh':0, 
                    'subversion':0, 
                    'non-moral':0, 
                    'cheating':0, 
                    'harm':0, 
                    'betrayal':0, 
                    'nm':0}
    return moral_counter

cache = []
for i in annotations:
    label = [j['annotation'].split(',') for j in i]
    flat_labels = [item for items in label for item in items]
    cache.append(flat_labels)


def count_morals(cache):
    moral_count_dicts = []
    for annotation in cache:
        moral_counter = reset_mc()
        for i in annotation:
            moral_counter[i] += 1
        moral_count_dicts.append(moral_counter)
    morals = pd.DataFrame(moral_count_dicts)
    morals['non-moral'] = morals['non-moral'] + morals['nm']
    morals = morals.drop(columns = 'nm')
    return morals
morals = count_morals(cache)
final = pd.concat([df, morals], axis = 1)

text = final.tweet_text.tolist()
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    sentiment_labels = []
    for sentence in text:
        vs = analyzer.polarity_scores(sentence)
        if vs['compound'] > 0.05:
            sentiment_labels.append(1)
        elif vs['compound'] < -0.05:
            sentiment_labels.append(-1)
        else:
            sentiment_labels.append(0)
    return sentiment_labels
sl = get_vader_sentiment(text)
final['sentiment'] = sl


## get virtue/vice correlations
df = pd.read_csv('data/mtfc.csv')
sentiment = df[['sentiment']].copy()
pos = np.where(sentiment==1, 1, 0)
neg = np.where(sentiment==-1, 1, 0)
sentiment['pos'] = pos
sentiment['neg'] = neg
df = df.drop(columns = ['tweet_id', 'tweet_text', 'annotations', 'annots', 'sentiment', 'nh', 'non-moral'])
normalized = df.div(df.sum(axis=1), axis=0).fillna(0)
binarized_df = normalized
#binarized_df = np.where(df>0, 1, 0)
#binarized_df = pd.DataFrame(binarized_df)
#binarized_df.columns = df.columns
binarized_df.corrwith(sentiment['pos'])
binarized_df.corrwith(sentiment['neg'])

binarized_df['virtue'] = (binarized_df['purity'] + binarized_df['authority'] + 
                        binarized_df['fairness'] +
                        binarized_df['care'] + binarized_df['loyalty']
).fillna(0)

binarized_df['vice'] = (binarized_df['subversion'] + binarized_df['cheating'] + 
                        binarized_df['harm'] + binarized_df['betrayal'] +
                        binarized_df['degradation']
).fillna(0)

vv = binarized_df[['virtue', 'vice']]
vv.corrwith(sentiment['pos'])
vv.corrwith(sentiment['neg'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
binarized_df = binarized_df.drop(columns = ['virtue', 'vice'])
X_train, X_test, y_train, y_test = train_test_split(binarized_df, sentiment, test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train['pos'])
predictions = logmodel.predict(X_test)
pos_coef = logmodel.coef_[0]

from sklearn.metrics import classification_report
print(classification_report(y_test['pos'],predictions))

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train['neg'])
predictions = logmodel.predict(X_test)
neg_coef = logmodel.coef_[0]

from sklearn.metrics import classification_report
print(classification_report(y_test['neg'],predictions))

## sentiment
logmodel = LogisticRegression()
no_neutrals = binarized_df.copy()
no_neutrals['sentiment'] = sentiment['sentiment']
no_neutrals = no_neutrals[no_neutrals['sentiment']!=0].copy()
no_neutrals = no_neutrals.dropna()
newsent = no_neutrals['sentiment']
no_neutrals = no_neutrals.drop(columns = 'sentiment')

X_train, X_test, y_train, y_test = train_test_split(no_neutrals, newsent, test_size=0.30, 
                                                    random_state=101)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
sentiment_coef = logmodel.coef_[0]

print(classification_report(y_test,predictions))

ex = pd.DataFrame((X_test.iloc[0])).T
logmodel.predict(np.random.rand(1,10))

pickle.dump(logmodel, open('moral2emotelr.pkl', 'wb'))
model = pickle.load(open('moral2emotelr.pkl', 'rb'))

mean_cache = []
for i in range(1000):
    a = np.random.rand(10000, 10)
    apred = logmodel.predict(a)
    mean_cache.append(np.mean(apred))
    

