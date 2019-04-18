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
from Fact_Checking import embeddings_index

df = pd.DataFrame(columns=['text', 'label'])

path=sys.argv[1]
#path="./training"

i=0
file_list = glob.glob(path+"/celebrityDataset/fake/*.txt")
for file_name in file_list:
    file=open(file_name,"r", encoding="utf8")
    a=file.read()
    df.loc[i]=[a,1]
    i=i+1

file_list = glob.glob(path+"/celebrityDataset/legit/*.txt")
for file_name in file_list:
    file=open(file_name,"r", encoding="utf8")
    a=file.read()
    df.loc[i]=[a,0]
    i=i+1

file_list = glob.glob(path+"/fakeNewsDataset/fake/*.txt")
for file_name in file_list:
    file=open(file_name,"r", encoding="utf8")
    a=file.read()
    df.loc[i]=[a,1]
    i=i+1

file_list = glob.glob(path+"/fakeNewsDataset/legit/*.txt")
for file_name in file_list:
    file=open(file_name,"r", encoding="utf8")
    a=file.read()
    df.loc[i]=[a,0]
    i=i+1

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

# apply the above function to df['text']

df['text'] = df['text'].map(lambda x: clean_text(x))


stop_words=stopwords.words("english")
#embeddings_index=dict()
#f=open("glove.6B.100d.txt","r",encoding="utf8")
glove=[]
#r=f.readlines()
for i in embeddings_index.keys():
    if(i not in stop_words):
        glove.append(i)

#f.close()


#tfidf
transformer = TfidfTransformer(smooth_idf=True)
count_vectorizer = CountVectorizer(ngram_range=(2,3),vocabulary=glove)
counts = count_vectorizer.fit_transform(df['text'].values)
tfidf = transformer.fit_transform(counts)


target=df['label'].values.astype('int')
selector = SelectKBest(chi2, k=1000)
selector.fit(tfidf, target)
top_words = selector.get_support().nonzero()

# Pick only the most informative columns in the data.
chi_matrix = tfidf[:,top_words[0]]


# In[150]:


# Our list of functions to apply.
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

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(df["text"].apply(func))

# Convert the meta features to a numpy array.
meta = np.asarray(columns).T



##features = np.hstack([ meta,chi_matrix.todense()])
features = np.hstack([ meta,tfidf.todense()])
##features=tfidf


# In[152]:


targets = df['label'].values
targets=targets.astype('int')

###split in samples
clf=LogisticRegression(C=1e5)
clf.fit(features,targets)

with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)




