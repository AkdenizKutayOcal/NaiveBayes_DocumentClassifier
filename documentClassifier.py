# ----------------------------------
# Akdeniz Kutay Öçal
# ----------------------------------

import re
import string
import pandas as pd

import numpy as np
from sklearn.datasets import fetch_20newsgroups  # load data from sklearn dataset

data = fetch_20newsgroups()
categories = data.target_names
# Training the data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories
test = fetch_20newsgroups(subset='test', categories=categories)

def preProcessing(text):

    # Convert text to lowercase
    outText = text.lower()

    # Remove numbers
    outText = re.sub(r'\d+', '', outText)

    # Remove punctuation
    outText = outText.translate(str.maketrans("", "", string.punctuation))

    # Remove whitespaces
    outText = outText.strip()

    # Remove stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(outText)
    outText = [i for i in tokens if not i in stop_words]

    # Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in outText:
        result.append(lemmatizer.lemmatize(word))

    return result

def createDict(list):
# creates a dictionary from given list of documents in a sorted way

    dict = {}

    for text in list:
        for word in preProcessing(text):
            dict[word] = dict.get(word,0)+1

    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def dictToFile(filename,dict):
    # used to store vocabulary in a file
    # used to speed up the process
    with open(filename, "w", encoding="utf-8") as file:

        for i,(k,v) in enumerate(dict.items()):
            file.write(str(i)+" "+str(k)+" "+str(v)+"\n")

def dictFromFile(filename):
    # loads vocabulary as a dictionary but only loads 5000 of them
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        itemsList = []
        for line in lines:
            line = line[:-1]
            items = line.split(" ")
            if(int(items[0])>4999):
                break
            else:
                itemsList.append(items)

        dict = {item[1]:int(item[2]) for item in itemsList}

    return dict

def frequencyCalculation(data,features):
    # returns frequency of words in each text in data list
    freqList = []
    for text in data:
        dict = {}
        words = preProcessing(text)
        for key in features:
            if key in words:
                dict[key] = dict.get(key, 0) + 1
            else:
                dict[key] = 0
        freqList.append(list(dict.values()))
    return freqList

def yGeneration(data_list):
# generates y list contains catagories for given data
    y = []
    target_list = list(data_list.target_names)

    for text in data_list.target:
        y.append(target_list[text])

    return y

def calculateClassPriors():
# Class prior calculator used for question 2
    classes = []
    total = len(train.target)
    for label in range(0,len(train.target_names)):
        labelTotal = 0
        for i in train.target:
            if(i==label):
                labelTotal+=1
        classes.append(labelTotal/total)
    for i in range(0,len(classes)):
        print(train.target_names[i]," ",classes[i])
    print(sum(classes))

# Codes that are used during implementation but not necessary
# To sped up process dataframes are stored in a file

# file_Train = "trainDict.txt"
# file_Test = "testDict.txt"
file_Vocab = "vocab.txt"

#dict_vocab = createDict((train.data+test.data))
#dictToFile(file_Vocab,dict_vocab)
# dict_train = createDict(train)
# dict_test = createDict(test)

#
# dictToFile(file_Train,dict_train)
# dictToFile(file_Test,dict_test)

dict_vocab = dictFromFile(file_Vocab)
features = list(dict_vocab.keys())

#df_train = pd.DataFrame(frequencyCalculation(train.data,features), columns = features)
#df_test = pd.DataFrame(frequencyCalculation(test.data,features), columns = features)
#df_train.to_pickle("df_train.pkl")
#df_test.to_pickle("df_test.pkl")

df_train = pd.read_pickle("df_train.pkl")
df_test = pd.read_pickle("df_test.pkl")
y_train = yGeneration(train)
y_test = yGeneration(test)

# Build in Naive Bayes Algorithm implementation
#
# from sklearn.naive_bayes import MultinomialNB
# mnb = MultinomialNB()
# mnb.fit(df_train.values,y_train)
# y_pred = mnb.predict(df_test.values)
#
# # confusion matrix and accuracy
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as ac
#
# cm = cm(y_test,y_pred)
# print("Accuracy=",ac(y_test,y_pred))
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fig, ax = plt.subplots(figsize=(15, 10))
# sns.heatmap(cm, annot=True, cmap = "Set3", fmt ="d",
# xticklabels=list(test.target_names), yticklabels=list(test.target_names))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
#

def fit(x,y):
    dict={}
    dict["total_data"]=x[:,:].sum()   # key containing the total no. of words in all the classes
    labels = set(y)

    for i in labels:                        # iterating over all categories
        dict[i] = {}
        indexs = y == i
        x_withLabel = x[indexs]                # x_current contain only those documents which has i as its category
        dict[i]["total_count"] = x_withLabel[:,:].sum()   # key containing the no. of words in ith category
        for j in range(0,5000):               # iterating over all features
            dict[i][j] = x_withLabel[:,j].sum()      # for each feature counting the total no. of words
    return dict

def probability(d, x, k):

    p_label = (d[k]["total_count"])/(d["total_data"])
    total_prob = (10.0**300) * p_label

    # calculating the probability of only those words which are present in the vocabulary
    for j in range(0,5000):         # iterating features
        xj=x[j]
        if xj != 0:  # id the word isn't in the current document ignored
            pay = d[k][j]+1       # total no. of words in feature j when category is k along with laplace smoothing
            payda = d[k]["total_count"]+len(features)
            p_word = pay/payda
            total_prob *= p_word
    return total_prob

def predictsingle(d,x):
    # Calls probability function for all labels and returns the one with the highest probability

    labels = d.keys()
    highest_prob = -10000
    best_label = -1

    for key in labels:                # iterating over each category and passing them to the probability function
        if key == "total_data":
            continue

        temp_prob = probability(d, x, key)  # probability of the test document with the given label

        if temp_prob > highest_prob:  # finding the label with highest probability
            best_label = key
            highest_prob = temp_prob

    return best_label

def predict(d,x_test):
    # passing each test text to predictsingle function one by one and returns list of labels
    y_pred = []
    for x in x_test:
        y_pred.append(predictsingle(d,x))

    return y_pred

y_indexs = train.target
y_indexs_test = test.target
dict_fit = fit(df_train.values,y_indexs)
#calculateClassPriors()

def indicators(dict_fit):
# Used for question 5
    indicators = []
    print(dict_fit)
    for key in dict_fit.keys():

        words = []
        if(key=="total_data"):
            continue
        print("\n",train.target_names[key],":")
        sorted_dict = {k: v for k, v in sorted(dict_fit[key].items(), key=lambda item: item[1], reverse=True)}
        for i,key in enumerate(sorted_dict.keys()):
            if(3<i<25):
                print(features[int(key)],end=" ")

#indicators(dict_fit)

y_p = predict(dict_fit,df_test.values)

cm = cm(y_indexs_test,y_p)
print("Accuracy=",ac(y_indexs_test,y_p))

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(cm, annot=True, cmap = "Set3", fmt ="d",
xticklabels=list(test.target_names), yticklabels=list(test.target_names))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
