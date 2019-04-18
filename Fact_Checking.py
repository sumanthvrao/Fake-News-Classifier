import numpy as np
import pandas as pd
import keras
import os
import json
import os
import random
import nltk
import http.client
import urllib.parse
import json
import re
import string

from scipy import spatial
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, SemanticRolesOptions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
stop_words = set(stopwords.words('english'))
translator = str.maketrans(dict.fromkeys(list(",")))
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize("was appointed",pos="v")
wordnet_lemmatizer.lemmatize("may have",pos="v")

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,"glove.6B.100d.txt")

embeddings_index=dict()
f=open(path)
df = pd.DataFrame(columns=["text"])
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype="float32")
    embeddings_index[word]=coefs
f.close()


#Given a document, returns a list of facts that need to be validated
def fact_extraction(doc): #Type: Str
    facts=[]
    doc=doc.translate(translator)
    response = natural_language_understanding.analyze(
        text=doc,
        features=Features(
        semantic_roles=SemanticRolesOptions(entities=True)
        )).get_result()

    for i in response['semantic_roles']:
        if('subject' in i.keys() and 'object' in i.keys()):
            if('entities' in i['subject'].keys() and 'entities' in i['object'].keys()):
                d=i['subject']['entities']
                e=i['object']['entities']
                if(d==[] or e==[]):
                    continue
                sentence=i['sentence']
                print(sentence)
                for add in sentence.split("."):
                    if(add!=''):
                        out=re.sub(r'[^\w\d\s\.]+', '', add)
                        facts.append(out.strip()+".")
    return facts

# subscription key
subscriptionKey = "46ca951ab6b946ad928c3fd285e7a8ce"
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/search"

# Performs a Bing Web search and returns the results.
def web_search(term):
    def BingWebSearch(search):

        headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
        conn = http.client.HTTPSConnection(host)
        query = urllib.parse.quote(search)

        # Filtering value webpages only and filtering count
        conn.request("GET", path + "?q=" + query+"&responseFilter=Webpages" + "&answerCount=10&count=10"+"&setLang=EN"+"&promote=Webpages", headers=headers)
        response = conn.getresponse()
        headers = [k + ": " + v for (k, v) in response.getheaders()
                    if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
        return headers, response.read().decode("utf8")

    # List of urls
    url=[]

    # List of webpage title
    names=[]

    # String of contents
    snippet=[]

    if len(subscriptionKey) == 32:
        headers, result = BingWebSearch(term)
        ans = json.loads(result)
        if "webPages" not in ans.keys():
            return []
        webpages = ans["webPages"]["value"]
        for hits in webpages:
            url.append(hits["url"])
            names.append(hits["name"])
            # TODO: fix snippet to have more information
            clean = hits["snippet"]
            out = re.sub(r'[^\w\d\s\.]+', '', clean)
            snippet.append(out)
        return snippet

    else:

        print("Invalid Bing Search API subscription key!")


df = pd.DataFrame(columns=["text"])

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey=os.environ["IAM_API_KEY"],
    url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
)

def cos_sim(x1,x2):
    try:
        return 1-spatial.distance.cosine(x1,x2)
    except:
        return 0

def triplet_return(text):
    global df
    if(len(text.split())<4):
        return None
    tup_negated_list=[]
    try:
        response = natural_language_understanding.analyze(
        text=text,language="English", features=Features(semantic_roles=SemanticRolesOptions(limit=10, entities=True, keywords=True))).get_result()
    except:
        return tup_negated_list
    # tup: 3 entity tuple with (subject, object, action)


    semantic_roles_list = response["semantic_roles"]
    for semantic_roles in semantic_roles_list:
        try:
            subject_str=""
            object_str=""
            if("keywords" in semantic_roles["subject"]):
                for i in semantic_roles["subject"]["keywords"]:
                    subject_str=subject_str+i["text"]+" "
            else:
                subject_str=semantic_roles["subject"]["text"]
            if("keywords" in semantic_roles["object"]):
                for i in semantic_roles["object"]["keywords"]:
                    object_str=object_str+i["text"]+" "
            else:
                object_str=semantic_roles["object"]["text"]
            tup = (subject_str, object_str, semantic_roles["action"]["text"])
            negated = False
            if("negated" in semantic_roles["action"]["verb"].keys()):
                negated=True
            print((tup, negated))  #Uncomment to view file.
            tup_negated_list.append((tup,negated))
        except:
            pass
    return tup_negated_list


def fact_check(testing): # Type : List[str]
    global df

    all_search_result = [] # Type : List[List[str]] #Contains tuples of form (triplet, Boolean)
    all_test_result = [] # Type: List[List[str]] #Contains test triplets

    for test in testing:
        print("test",test)
        tup_negated_list = triplet_return(test)
        print("tup negated list",tup_negated_list)
        if(tup_negated_list is None):
            continue

        # call to Bing API.
        search_result_list = web_search(test)  # Type: List[str]
        if(search_result_list ==[]):
            print("bing search for",test,"failed")
            continue
        search_result_list = [search_result_list[i].split(".") for i in range(0,len(search_result_list))]
        search_output =[]

        for ele in search_result_list:
            for e in ele:
                if(e !='' and len(e.split(" ")) > 4):
                    search_output.append(e)

        triplet_result_list=[]
        #Contains triplets for every sentence
        for ele in search_output:
            temp = triplet_return(ele)
            if (temp is not None):
                triplet_result_list.extend(temp)

        all_search_result.append(triplet_result_list)
        all_test_result.append(tup_negated_list)

        for (test_tup,negated_tup) in tup_negated_list:
            # # call to Bing API.
            # search_result_list = web_search(test)  # Type: List[str]
            # if(search_result_list ==[]):
            #     print("bing search for",test,"failed")
            #     continue
            # search_result_list = [search_result_list[i].split(".") for i in range(0,len(search_result_list))]
            # search_output =[]

            # for ele in search_result_list:
            #     for e in ele:
            #         if(e !='' and len(e.split(" ")) > 4):
            #             search_output.append(e)

            # triplet_result_list=[]
            # #Contains triplets for every sentence
            # for ele in search_output:
            #     temp = triplet_return(ele)
            #     if (temp is not None):
            #         triplet_result_list.extend(temp)

            # all_search_result.append(triplet_result_list)
            flag = 0
            for j in range(len(triplet_result_list)):
                # Objects

                # Removing stop words from triplets.
                filtered_sentence1 = []
                # Indexing object and cleaning object-For a given search result sentence
                for w in triplet_result_list[j][0][1].split():
                    if w not in stop_words:
                        filtered_sentence1.append(wordnet_lemmatizer.lemmatize(w.lower().strip(),pos="n"))
                filtered_sentence1 = " ".join(filtered_sentence1)

                # Indexing object and cleaning object -For test sentence
                filtered_sentence2 = []
                for w in test_tup[1].split():
                    if w not in stop_words:
                        filtered_sentence2.append(wordnet_lemmatizer.lemmatize(w.lower().strip(),pos="n"))
                filtered_sentence2 = " ".join(filtered_sentence2)
                try:
                    # Adding words to embedding matrix.
                    df=df.append({"text":filtered_sentence1},ignore_index=True)
                    df=df.append({"text":filtered_sentence2},ignore_index=True)

                    # Not cleaning for the action
                    l=[wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="v") for word in triplet_result_list[j][0][2].lower().split()]
                    df=df.append({"text":" ".join(l)},ignore_index=True)
                    l=[wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="v") for word in test_tup[2].lower().split()]
                    df=df.append({"text":" ".join(l)},ignore_index=True)

                    #Add subjects to embedding matrix
                    df=df.append({"text":triplet_result_list[j][0][0].lower().strip()},ignore_index=True)
                    df=df.append({"text":test_tup[0].lower().strip()},ignore_index=True)
                    #print("")
                except:
                    print("except")
                    pass

    vocabulary_size = len(embeddings_index)
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df['text'])
    print("df",df["text"])
    sequences = tokenizer.texts_to_sequences(df['text'])
    data = pad_sequences(sequences, maxlen=50)

    embedding_matrix = np.zeros((vocabulary_size, 100))

    print(tokenizer.word_index.items())
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            print("SHIT")
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                print(word,"has embedding vector")
                embedding_matrix[index] = embedding_vector
            else:
                print(word,"not in glove")

    search_index=0
    true_value=0
    false_value=0
    for test in all_test_result:
        tup_negated_list = test
        for (test_tup,negated_tup) in tup_negated_list:
            flag = 0
            triplet_result_list=all_search_result[search_index]
            for j in range(len(triplet_result_list)):
                #Objects
                filtered_sentence1 = []
                for w in triplet_result_list[j][0][1].lower().split():
                    if w not in stop_words:
                        filtered_sentence1.append(w.lower().strip())
                # new object is this
                filtered_sentence1 = " ".join(filtered_sentence1)


                filtered_sentence2 = []
                for w in test_tup[1].lower().split():
                    if w not in stop_words:
                        filtered_sentence2.append(w.lower().strip())

                # New test object
                filtered_sentence2 = " ".join(filtered_sentence2)

                # Subject
                w111=np.zeros(100)
                count=0
                for word in triplet_result_list[j][0][0].lower().split():
                    w111+=embedding_matrix[tokenizer.word_index[word.lower().strip()]]
                    count+=1
                if(count==0):
                    count=1
                w111=w111/count

                w222=np.zeros(100)
                count=0
                for word in test_tup[0].split():
                    print(word)
                    print("-----------------")
                    w222+=embedding_matrix[tokenizer.word_index[word.lower().strip()]]
                    count+=1
                if(count==0):
                    count=1
                w222=w222/count


                w1=np.zeros(100)
                count=0
                for word in filtered_sentence1.split():
                    word=wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="n")
                    w1+=embedding_matrix[tokenizer.word_index[word]]
                    count+=1
                if(count==0):
                    count=1
                w1=w1/count

                w2=np.zeros(100)
                count=0
                for word in filtered_sentence2.split():
                    word=wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="n")
                    w2+=embedding_matrix[tokenizer.word_index[word]]
                    count+=1
                if(count==0):
                    count=1
                w2=w2/count

                #Actions
                w11=np.zeros(100)
                count=0
                for word in triplet_result_list[j][0][2].split():
                    w11+=embedding_matrix[tokenizer.word_index[wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="v")]]
                    count+=1
                if(count==0):
                    count=1
                w11=w11/count

                w22=np.zeros(100)
                count=0
                for word in test_tup[2].split():
                    w22+=embedding_matrix[tokenizer.word_index[wordnet_lemmatizer.lemmatize(word.lower().strip(),pos="v")]]
                    count+=1
                if(count==0):
                    count=1
                w22=w22/count

                print(cos_sim(w111,w222))
                print(cos_sim(w1,w2))
                print(cos_sim(w11,w22))
                print("------------------")
                try:
                    # print(" subject",triplet_result_list[j][0][0]," ::: ",test_tup[0])
                    # print(" Object", triplet_result_list[j][0][1]," ::: ",test_tup[1])
                    # print(" Action", wordnet_lemmatizer.lemmatize(triplet_result_list[j][0][2],pos="v")," ::: ",wordnet_lemmatizer.lemmatize(test_tup[2],pos="v"))

                    if cos_sim(w1,w2)>=0.7 and cos_sim(w11,w22)>=0.3 and cos_sim(w111,w222)>=0.70:
                        print("MATCHED subject",triplet_result_list[j][0][0]," ::: ",test_tup[0])
                        print("MATCHED Object", triplet_result_list[j][0][1]," ::: ",test_tup[1])
                        print("MATCHED Action", triplet_result_list[j][0][2]," ::: ",test_tup[2])
                        if ((negated_tup is True and triplet_result_list[j][1] is False) and (negated_tup is False and triplet_result_list[j][1] is True)):
                            flag=0
                            break
                        else:
                            flag=1
                            break
                except:
                    print("Cannot find match!")
                    continue
            print("===================================")

            if flag==1 :
                true_value+=1
                print ("The fact mentioned above is True.")
            else:
                false_value+=1
                print ("The fact mentioned above is False.")
        search_index+=1
    try:
        accuracy = true_value/(true_value+false_value)
    except ZeroDivisionError:
        accuracy = 0.0
    print("Accuracy of predicting Fact is True : ",accuracy)
    #return accuracy
    if(accuracy >= 0.5):
        return 0 # Article is Real
    else:
        return 1 # Article is Fake

if __name__ == "__main__":
    testing = ["Trump is the President Of USA.","Mount Everest is 400m tall."]
    fact_check(testing)

