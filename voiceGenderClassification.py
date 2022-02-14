import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
set = pd.read_csv("voicedataset.csv")
#print(set)
set.label = [1 if each == "female" else 0 for each in set.label]
y =set.label.values
#print(y)

xdata = set.drop(["label"],axis=1)

x = (xdata-np.min(xdata)) /(np.max(xdata)).values

xtrain, xtest, ytrain, ytest = train_test_split (x,y,test_size=0.2,random_state = 30)
svm = SVC (random_state=42)

svm.fit(xtrain, ytrain)

f=svm.predict(xtest)
#plot a graph of f against the x test 
# f predicts that in which class does each datapoint of the test data belongs to. either 1 or 0.
#print(f)
#print(xtest)

accscore = accuracy_score(ytest, f)
print(accscore)
recscore = recall_score(ytest, f)
print(recscore)
f1score = f1_score (ytest, f)
print(f1_score)
kappascore=cohen_kappa_score(ytest,f)
print(kappascore)
prescore = precision_score(ytest, f)
print(prescore)
