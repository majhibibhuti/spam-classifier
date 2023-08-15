# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math


# %%
list_dir_spam = os.listdir(os.getcwd()+"/dataset/spam")
list_dir_ham = os.listdir(os.getcwd()+"/dataset/ham")
spam_path = os.getcwd()+"/dataset/spam/"
ham_path = os.getcwd()+"/dataset/ham/"

# %%
len_spam = len(list_dir_spam)
len_ham = len(list_dir_ham)

# %%
len_spam

# %%
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# %%
def preprocessFile(filename):
	words = []
	with open(filename, "r", errors="ignore") as file:
		filedata = file.readlines()
		for line in filedata:
			sent = decontracted(line)
			sent = sent.replace('\\r', ' ')
			sent = sent.replace('\\"', ' ')
			sent = sent.replace('\\n', ' ')
			sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
			sent = ' '.join(e.lower() for e in sent.split())
			words += list(sent.strip().split())
	return words

# %%
words = {}
for filename in list_dir_spam:
    for word in preprocessFile(spam_path + filename):
        if(word not in words):
            words[word] = 0
for filename in list_dir_ham:
    for word in preprocessFile(ham_path + filename):
        if(word not in words):
            words[word] = 0

# %%
list_words = []
for word in words:
    list_words.append(word)

# %%
spamwords=words.copy()
for filename in list_dir_spam:
    for word in preprocessFile("/Users/crysiswar999/Documents/Assignment 3/dataset/spam/" + filename):
        spamwords[word] += 1
for x in spamwords:
    spamwords[x] +=1
len_spam +=1
count = 0
for word in spamwords:
    if(spamwords[word] != 0):
        count+=1
temp_spamwords = spamwords.copy()
for word in spamwords:
    spamwords[word] /= count

print(spamwords)


# %%
hamwords=words.copy()
for filename in list_dir_ham:
    for word in preprocessFile("/Users/crysiswar999/Documents/Assignment 3/dataset/ham/" + filename):
        hamwords[word] += 1
for x in hamwords:
    hamwords[x] += 1
len_ham += 1
count = 0
for word in hamwords:
    if(hamwords[word] != 0):
        count+=1
temp_hamwords = hamwords.copy()
for word in hamwords:
    hamwords[word] /= count
print(hamwords)


# %%
phat = len_spam/(len_spam + len_ham)
phat

# %%
dict(sorted(spamwords.items(), key=lambda kv: (kv[1], kv[0]))[::-1])


# %%
def ComputeLabel(xtest):
    test = preprocessFile(xtest)
    feature = []
    for x in test:
        if(x not in words):
            # spamwords[x] = 1/(len(spamwords)+1)
            # hamwords[x] = 1/(len(hamwords)+1)
            temp_hamwords = hamwords.copy()
            temp_spamwords = spamwords.copy()
            words[x] = 0
            temp_spamwords[x] = 0
            for y in temp_spamwords:
                temp_spamwords[y] += 1
            for y in temp_spamwords:
                spamwords[y] = temp_spamwords[y]/len(temp_spamwords)
            temp_hamwords[x] = 0
            for y in temp_hamwords:
                temp_hamwords[y] += 1
            for y in temp_hamwords:
                hamwords[y] = temp_hamwords[y]/len(temp_hamwords)
    temp_spamwords = dict(sorted(spamwords.items(), key=lambda kv: (kv[1], kv[0]))[::-1])
    temp_hamwords = dict(sorted(hamwords.items(), key=lambda kv: (kv[1], kv[0]))[::-1])
            
    for x in temp_spamwords:
        if(x in test):
            feature.append(1)
        else:
            feature.append(0)
    probham  = 1
    probspam = 1

    # for x in list_words:
    #     if(x in test):
    #         probspam *= spamwords[x]
    #         probham *= hamwords[x]
    #     else:
    #         probspam *= (1-spamwords[x])
    #         probham *= (1-hamwords[x])
    #     print(probham, probspam)
    list_words = []
    for x in temp_spamwords:
        list_words.append(x)
    for x in range(len(feature)):
        probspam  *= ((temp_spamwords[list_words[x]]**feature[x])*((1-temp_spamwords[list_words[x]])**(1-feature[x])))
        # probspam  += (math.log(temp_spamwords[list_words[x]])*feature[x])+(math.log(1-temp_spamwords[list_words[x]])*(1-feature[x]))

        # if(probspam < 1e-310):
        #     break
    feature = []
    for x in temp_hamwords:
        if(x in test):
            feature.append(1)
        else:
            feature.append(0)
    list_words = []
    for x in temp_hamwords:
        list_words.append(x)
    for x in range(len(feature)):
        probham  *= ((temp_hamwords[list_words[x]]**feature[x])*((1-temp_hamwords[list_words[x]])**(1-feature[x])))
        # probham  += (math.log(temp_hamwords[list_words[x]])**feature[x])+(math.log(1-temp_hamwords[list_words[x]])*(1-feature[x]))

        # if(probham < 1e-310):
        #     break
    probham *= (0.5)
    probspam *= (0.5)
    if(probham > probspam):
        print("0")
        return 0
    else:
        print("+1")
        return 1
    


# %%
#place your test emails in test folder 
currentpath = os.getcwd()+"/test/"
list_dir_test = os.listdir(os.getcwd()+"/test/")
spam = 0
ham = 0
for filename in list_dir_test:
    res = ComputeLabel(currentpath + filename)
    if(res==0):
        ham += 1
    spam += res

# %%
print("Percentage of spam emails in the test folder : " + spam/len(list_dir_test)*100)

# %% [markdown]
# ## SVM classifier for spam detection.

# %%
from sklearn import svm
currentpath = os.getcwd()+"/dataset/ham/"
list_dir_test = os.listdir(os.getcwd()+"/dataset/ham/")
xtrain = []
ytrain = []
for filename in list_dir_test:
    test = preprocessFile(currentpath + filename)
    feature = []
    for x in list_words:
        if(x in test):
            feature.append(1)
        else:
            feature.append(0)
    xtrain.append(feature)
    ytrain.append(0)
currentpath = os.getcwd()+"/dataset/spam/"
list_dir_test = os.listdir(os.getcwd()+"/dataset/spam/")
for filename in list_dir_test:
    test = preprocessFile(currentpath + filename)
    feature = []
    for x in list_words:
        if(x in test):
            feature.append(1)
        else:
            feature.append(0)
    xtrain.append(feature)
    ytrain.append(1)


    
    


# %%
from sklearn import svm
#Place your test emails in test folder.
currentpath = os.getcwd()+"/test/"
list_dir_test = os.listdir(os.getcwd()+"/test/")
xtest = []
for filename in list_dir_test:
    test = preprocessFile(currentpath + filename)
    feature = []
    for x in list_words:
        if(x in test):
            feature.append(1)
        else:
            feature.append(0)
    xtest.append(feature)



# %%
X = xtrain
y = ytrain

C = 0.1
svc = svm.SVC(C = C, kernel = 'linear')
svc.fit(X, y)
p = svc.predict(X)

print('Training Accuracy: {0:.2f}%'.format(np.mean((p == y).astype(int)) * 100))

# %%
Labels = svc.predict(xtest)
#place your true labes in ytest list
ytest = []
print('Testing Accuracy: {0:.2f}%'.format(np.mean((labels == ytest).astype(int)) * 100))


