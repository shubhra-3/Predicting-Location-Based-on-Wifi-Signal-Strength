
#datafrae sets column name alphabetically
#so we need to work on it

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
colnames=['Id','1','2','3','4','5','6','7']
testset = pd.read_csv('test.csv', names=colnames, header=None)
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values
test = testset.iloc[:, 1:8].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test = sc.transform(test)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)

# Predicting the Test set results
y_predLR = classifierLR.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predLR, normalize = False )

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(y_test, y_predLR)

predictLR = classifierLR.predict(test)
solution = pd.read_csv('sampleSubmission.csv')
solution = pd.DataFrame({'Category':predictLR, 'Id':testset['Id']})
solution.to_csv('sampleSubmission.csv', sep=',', encoding='utf-8', header=True, columns=["Id","Category"])



#K Nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
for i in range(50) :
    i=i+1
    classifierKNN = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    classifierKNN.fit(X_train, y_train)

    y_predKNN = classifierKNN.predict(X_test)
    print  ('accuracy_score ', accuracy_score(y_test, y_predKNN, normalize = False ))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmKNN = confusion_matrix(y_test, y_predKNN)
"""
confusion matrix
each row stands for actual class and each column for predicted class 
"""

#naive bayes
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB();
classifierNB.fit(X_train, y_train)

y_predNB = classifierNB.predict(X_test)
predNB = classifierNB.predict(test)
accuracy_score(y_test, y_predNB, normalize = False )
solution = pd.read_csv('sampleSubmission.csv')
solution = pd.DataFrame({'Room':predNB, 'Id':testset['Id']})
solution.to_csv('sampleSubmission.csv',index = False, sep=',',  header=True, columns=["Id","Room"])

cmNB = confusion_matrix(y_test , y_predNB)

#support vector machine
from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'linear' , random_state = 0)
classifierSVC.fit(X_train , y_train)

y_predSVC = classifierSVC.predict(X_test)
accuracy_score(y_test, y_predSVC, normalize = False )

cmSVC = confusion_matrix(y_test , y_predSVC)
 
#decision tree model
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier( criterion = "entropy",random_state=0)
classifierDT.fit(X_train,y_train)

y_predDT = classifierDT.predict(X_test)
accuracy_score(y_test, y_predDT, normalize = False )

cmDT = confusion_matrix(y_test , y_predDT)
