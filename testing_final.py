import sys
import _pickle as cPickle
import pandas as pd
import numpy as np
import textstat
import glob
import statistics
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn                        import metrics, svm
from sklearn.svm                    import SVC
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import sys
import _pickle as cPickle



def clean_text(text):
        ## Remove puncuation
        #text = text.translate(string.punctuation)

        ## Convert words to lower case and split them
        text = text.lower().split()

        ## Remove stop words
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

        text = " ".join(text)

        ## Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        ## Stemming
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

        return text

def testing_tfidf_fe(s):
    s=clean_text(s)
    df= pd.DataFrame(columns=['text'])
    df.loc[0]=[s]

    with open('my_dumped_classifier.pkl', 'rb') as fid:
        gnb_loaded = cPickle.load(fid)
    f=open("glove.6B.100d.txt","r",encoding="utf8")

    glove=[]
    r=f.readlines()
    stop_words=stopwords.words("english")

    for i in r:
        if(i.split()[0] not in stop_words):
            glove.append(i.split()[0])

    f.close()

    transform_functions = [

            lambda x: x.count(" ")/len(x.split()),
            lambda x: x.count(".")/len(x.split()),
            lambda x: x.count("!")/len(x.split()),
            lambda x: x.count("?")/len(x.split()),
            lambda x: x.count("-")/len(x.split()),
            lambda x: x.count(",")/len(x.split()),
            lambda x: x.count("$")/len(x.split()),
            lambda x: x.count("(")/len(x.split()),
            lambda x: len(x) / (x.count(" ") + 1),
            lambda x: x.count(" ") / (x.count(".") + 1),
            lambda x: len(re.findall("\d", x)),
            lambda x: len(re.findall("[A-Z]", x)),
            lambda x: textstat.flesch_reading_ease(x),
            lambda x: textstat.smog_index(x),
            lambda x: textstat.flesch_kincaid_grade(x),
            lambda x: textstat.coleman_liau_index(x),
            lambda x: textstat.automated_readability_index(x),
            lambda x: textstat.dale_chall_readability_score(x),
            lambda x: textstat.difficult_words(x),
            lambda x: textstat.linsear_write_formula(x),
            lambda x: textstat.gunning_fog(x),
        ]

    transformer = TfidfTransformer(smooth_idf=True)
    count_vectorizer = CountVectorizer(ngram_range=(2,3),vocabulary=glove)
    counts = count_vectorizer.fit_transform(df['text'].values)
    tfidf = transformer.fit_transform(counts)

    columns = []

    for func in transform_functions:
        columns.append(df["text"].apply(func))


    meta = np.asarray(columns).T

    features = np.hstack([ meta,tfidf.todense()])
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        gnb_loaded = cPickle.load(fid)
        ans=gnb_loaded.predict(features)[0]
        if(ans==0):
            print("Article is legitimate according to classifier")
        else:
            print("Article is fake according to classifiers")
        return ans







