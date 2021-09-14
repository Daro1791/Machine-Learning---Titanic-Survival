#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Let's use machine learning to create models that predicts which passengers survived the Titanic shipwreck.
# We are going to use the data from Kaggle page (https://www.kaggle.com/c/titanic). Here we have file train.cvs, which 
# we will use to train the model, and file test.cvs, which we will use to make predictions.


# In[2]:


# 1. let’s import the libraries we are going to use for read and processing the data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from pandas.plotting import scatter_matrix

from sklearn.tree import DecisionTreeClassifier     
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC                         
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance # for naive bayes feature important


from sklearn.tree import export_graphviz
import graphviz


# In[3]:


# We are using pandas to load the data.
df = pd.read_csv('train.csv')
prediction = pd.read_csv('test.csv')


# In[4]:


# 2. Summarize the Dataset
print('df')
print('Head')
print(df.head())
print('Tail')
print(df.tail())
print('Shape:', df.shape)
print('prediction')
print('Head')
print(prediction.head())
print('Tail')
print(prediction.tail())
print('Shape:', prediction.shape)


# In[5]:


# We can see that the dataframe df has 891 rows and 12 columns. The columns are labeled and on the Kaggle page we can find 
# a description of each of them, which is the following:
# survival: Survival (0 = No, 1 = Yes)
# pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# sex: Sex
# Age: Age in years
# sibsp: number of siblings / spouses aboard the Titanic
# parch: number of parents / children aboard the Titanic
# ticket: Ticket number
# fare: Passenger fare
# cabin: Cabin number
# embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
# The dataframe "prediction" has 418 rows and 11 columns. The missing column is the one corresponding to "survival".

# We see if there is missing data
print("Missing data")
print(df.isnull().sum())
# In percentage
print("Missing data in percentage")
print(round(df.isnull().sum()/df.shape[0]*100,2))


# In[6]:


# we can see that most of the data from Cabin is missing, and a non-negligible amount of data from Age. 
# Although we assume that Cabin's data will be irrelevant
# Let's know the type of data
df.dtypes


# In[7]:


# We do a statistical analysis of the data
# count: number of non-null data in column
# mean: column mean value
# std: column desviation standart 
# min: minimum column value
# 25 %: percentile
# 50 %: percentile
# 75 %: percentile
# max: mmaximum column value
# unique: number of distinct objects in the column
# top: data that is repeated the most
# freq: the number of times the most repeated data appears
df.describe(include="all")


# In[8]:


# Print a concise summary of a DataFrame.
df.info


# In[9]:


# 3 - data processing
# Working with missing data

# Before we make a copy of the original data
dfCopy = df.copy()

# From the Age column we take the mean and round it
Meandf = round(df["Age"].mean()) 
print("Meandf", Meandf) 
Meanprediction = round(prediction["Age"].mean())
print("Meanprediction", Meanprediction) 

# We replace the data from Age column with that mean
df["Age"] = df["Age"].replace(np.nan, Meandf)
prediction["Age"] = prediction["Age"].replace(np.nan, Meanprediction)

# Change categorical variables to numeric
df['Sex'].replace(['female', 'male'], [0, 1], inplace = True)
prediction['Sex'].replace(['female', 'male'], [0, 1], inplace = True)

df['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace = True)
prediction['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace = True)

# We create several groups according to age bands
# Bands: 0-12, 9-21, 22-40, 41-60, 61-100
bins = [0, 12, 21, 40, 60, 100]
names = ['1', '2', '3', '4', '5']
df['Age'] = pd.cut(df['Age'], bins, labels = names)
prediction['Age'] = pd.cut(prediction['Age'], bins, labels = names)

# The Cabin column is removed as it has a lot of lost data
df.drop(['Cabin'], axis = 1, inplace=True)
prediction.drop(['Cabin'], axis = 1, inplace=True)

# We delete the columns that we think are not necessary for the analysis (PassengerId, Name, Ticket)
df = df.drop(['PassengerId','Name','Ticket'], axis=1)
prediction = prediction.drop(['Name','Ticket'], axis=1)

# Rows with missing data are deleted
df.dropna(axis=0, how='any', inplace=True)
prediction.dropna(axis=0, how='any', inplace=True)

# Let's check the data one more time
print("Missing date df:")
print(pd.isnull(df).sum())
print("Missing date prediction:")
print(pd.isnull(prediction).sum())
print("df shape:", df.shape)
print("prediction shape:", prediction.shape)
print("df head:", df.head())
print("prediction head:", prediction.head())

# The data is ready


# In[10]:


# Finding the number of people survived and not survived
df['Survived'].value_counts()


# In[11]:


# Extend that with some visualizations.
# Plots of each individual variable. We can create box and whisker plots of each.

# Make a box-and-whisker plot from DataFrame columns, optionally grouped by some other columns. 
# A box plot is a method for graphically depicting groups of numerical data through their quartiles. 
# The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2). 
# The whiskers extend from the edges of box to show the range of the data. By default, they extend no more than 
# 1.5 * IQR (IQR = Q3 - Q1) from the edges of the box, ending at the farthest data point within that interval. 
# Outliers are plotted as separate dots.

df.plot(kind='box', subplots=True, layout=(6,2), figsize=(15,15), fontsize=15)
plt.savefig('whisker.jpg')
plt.show()


# In[12]:


# histograms (https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.hist.html#:~:text=hist()%20%2C%20on%20each%20series,in%20one%20histogram%20per%20column.&text=The%20pandas%20object%20holding%20the%20data.&text=If%20passed%2C%20will%20be%20used,to%20a%20subset%20of%20columns.&text=If%20passed%2C%20then%20used%20to%20form%20histograms%20for%20separate%20groups.)
df.hist(layout=(6,2), xlabelsize=12, ylabelsize=12, figsize=(15,15))
plt.savefig('histograms.jpg')
plt.show()


# In[13]:


# scatter plot matrix
scatter_matrix(df, figsize=(60,60))
plt.savefig('scatter.jpg')
plt.show()


# In[14]:


# 4 - Create a Validation Dataset
# We separate the column with the information of the survivors
X = np.array(df.drop(['Survived'], 1))
y = np.array(df['Survived'])


# In[15]:


# We split out df into training data for preparing the models and testing data that we will use for testing them.
# We choose 20 % for test and the rest for train
# We divided the original data into training and test sets was to use the test set as a way to estimate how well the 
# model trained on the training data would generalize to new previously unseen data. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[16]:


# MACHINE LEARNING MODELS
# We will use validation curves and stratified k-fold cross validation to estimate model accuracy.
Metrics = pd.DataFrame()

## Decision Tree
# Decision trees are easy to use and understand and are often a good exploratory method if you're interested in getting 
# a better idea about what the influential features are in your dataset. 
decision = DecisionTreeClassifier()

# Validation curve
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
param_range = np.arange(1, 16)
train_scores, test_scores = validation_curve(decision, X, y, param_name="max_depth", param_range=param_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


print("TRAIN SCORES:")
print(train_scores)
print("TRAIN SCORES MEAN:")
print(train_scores_mean)

print("TEST SCORES:")
print(test_scores)
print("TEST SCORES MEAN:")
print(test_scores_mean)

plt.title("Validation Curve with Decision Tree")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("DecitionTreeValCurve.jpg")
plt.show()


# In[17]:


# From Validation Curve, we choose Max_Depth = 3
Max_Depth = 3
# Cross-validation
decision = DecisionTreeClassifier(max_depth=Max_Depth) # max_depth, min_samples_leaf, random_state, criterion
Cross_val = cross_val_score(decision, X, y, cv=5)
print("Cross validation score:", Cross_val)
print("Cross validation mean", np.mean(Cross_val))

# Individual
decision.fit(X_train, y_train)
y_pred = decision.predict(X_test)
print('Decision Tree Scores:')
print('Train set', round(decision.score(X_train, y_train), 2))
print('Test set', round(decision.score(X_test, y_test), 2))

# Metrics
matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') # Confusion Matrix [TN FP] ; [FN, TP]
print(matrix)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: TN+TP / TN+TP+FN+FP
print('Accuracy:', round(accuracy, 2))

precision = precision_score(y_test, y_pred) # Precision: TP / TP+FP
print('Precision', round(precision,2))

recall = recall_score(y_test, y_pred) # Recall: TP / TP+FN
print('Recall:', round(recall, 2))

f1 = f1_score(y_test, y_pred) # F1 Score (precision+recall): 2TP / 2TP+FN+FP
print('F1 Score:', round(f1, 2))

# Saving these measures in dataframe
Metrics['Decition Trees'] = [accuracy, precision, recall, f1]

Metrics.index = ['Accuracy', 'Precision', 'Recall', 'F1_Score']


# In[18]:


export_graphviz(decision, out_file = 'titanic.dot', class_names = np.array(['No Survival', 'Survival']),
               feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], impurity=False, filled=True)
with open('titanic.dot') as f:
    dot_graph = f.read()
graph = graphviz.Source(dot_graph, format = "png")
graph


# In[19]:


# Saving graph
graph.render("decision_tree-" + str(Max_Depth) + "deph")


# In[20]:


# Feature importance
plt.barh(range(X.shape[1]), decision.feature_importances_)
plt.yticks(np.arange(X.shape[1]), df.columns[1:])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("DecisionTreeFeatureImportance.jpg")
plt.show


# In[21]:


## Random Forest
# Random forests are widely used in practice and achieve very good results on a wide variety of problems.
# One disadvantage of using a single decision tree was that decision trees tend to be prone to overfitting the 
# training data. As its name would suggest, a random forest creates lots of individual decision trees on a 
# training set, often on the order of tens or hundreds of trees. The idea is that each of the individual trees in a 
# random forest should do reasonably well at predicting the target values in the training set but should also be 
#constructed to be different in some way from the other trees in the forest.

forest = RandomForestClassifier(criterion = 'entropy')

# Validation curve
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
param_range = np.arange(1, 16)
train_scores, test_scores = validation_curve(forest, X, y, param_name="max_depth", param_range=param_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


print("TRAIN SCORES:")
print(train_scores)
print("TRAIN SCORES MEAN:")
print(train_scores_mean)

print("TEST SCORES:")
print(test_scores)
print("TEST SCORES MEAN:")
print(test_scores_mean)

plt.title("Validation Curve with Random Forest")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("RandomForestValCurve.jpg")
plt.show()


# In[22]:


# From Validation Curve, we choose Max_Depth = 5
Max_Depth = 5
# Cross-validation
forest = RandomForestClassifier(max_depth=Max_Depth) 
Cross_val = cross_val_score(decision, X, y, cv=5)
print("Cross validation score:", Cross_val)
print("Cross validation mean", np.mean(Cross_val))

# Individual
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Random Forest Scores:')
print('Train set', round(forest.score(X_train, y_train), 2))
print('Test set', round(forest.score(X_test, y_test), 2))

# Metrics
matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') # Confusion Matrix [TN FP] ; [FN, TP]
print(matrix)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: TN+TP / TN+TP+FN+FP
print('Accuracy:', round(accuracy, 2))

precision = precision_score(y_test, y_pred) # Precision: TP / TP+FP
print('Precision', round(precision,2))

recall = recall_score(y_test, y_pred) # Recall: TP / TP+FN
print('Recall:', round(recall, 2))

f1 = f1_score(y_test, y_pred) # F1 Score (precision+recall): 2TP / 2TP+FN+FP
print('F1 Score:', round(f1, 2))

# Saving these measures in dataframe
Metrics['Random Forest'] = [accuracy, precision, recall, f1]


# In[23]:


# Feature importance
plt.barh(range(X.shape[1]), forest.feature_importances_)
plt.yticks(np.arange(X.shape[1]), df.columns[1:])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("RandomForestFeatureImportance.jpg")
plt.show


# In[24]:


## Logistic Regression
# Logistic regression can be seen as a kind of generalized linear model.
# However, unlike ordinary linear regression, in it's most basic form logistic repressions target value is a 
# binary variable instead of a continuous value.
logistic = LogisticRegression(max_iter=200)
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print('Logistic Regression:')
print('Train set', round(logistic.score(X_train, y_train), 2))
print('Test set', round(logistic.score(X_test, y_test), 2))

# Metrics
matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') # Confusion Matrix [TN FP] ; [FN, TP]
print(matrix)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: TN+TP / TN+TP+FN+FP
print('Accuracy:', round(accuracy, 2))

precision = precision_score(y_test, y_pred) # Precision: TP / TP+FP
print('Precision', round(precision,2))

recall = recall_score(y_test, y_pred) # Recall: TP / TP+FN
print('Recall:', round(recall, 2))

f1 = f1_score(y_test, y_pred) # F1 Score (precision+recall): 2TP / 2TP+FN+FP
print('F1 Score:', round(f1, 2))

# Saving these measures in dataframe
Metrics['Logistic Regression'] = [accuracy, precision, recall, f1]


# In[25]:


# Feature importance
# get importance
importance = logistic.coef_[0]
# plot feature importance
plt.barh(range(X.shape[1]), importance)
plt.yticks(np.arange(X.shape[1]), df.columns[1:])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("LogisticRegressionFeatureImportance.jpg")
plt.show


# In[26]:


## Naive Bayes
# Naive Bayes classifiers are called naive because informally, they make the simplifying assumption that each feature 
# of an instance is independent of all the others, given the class.
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)
print('Naive Bayes:')
print('Train set', round(NB.score(X_train, y_train), 2))
print('Test set', round(NB.score(X_test, y_test), 2))

# Metrics
matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') # Confusion Matrix [TN FP] ; [FN, TP]
print(matrix)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: TN+TP / TN+TP+FN+FP
print('Accuracy:', round(accuracy, 2))

precision = precision_score(y_test, y_pred) # Precision: TP / TP+FP
print('Precision', round(precision,2))

recall = recall_score(y_test, y_pred) # Recall: TP / TP+FN
print('Recall:', round(recall, 2))

f1 = f1_score(y_test, y_pred) # F1 Score (precision+recall): 2TP / 2TP+FN+FP
print('F1 Score:', round(f1, 2))

# Saving these measures in dataframe
Metrics['Naive Bayes'] = [accuracy, precision, recall, f1]

# Feature Importance
imps = permutation_importance(NB, X_test, y_test)
plt.barh(range(X.shape[1]), imps.importances_mean)
plt.yticks(np.arange(X.shape[1]), df.columns[1:])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("NaiveBayesFeatureImportance.jpg")
plt.show


# In[27]:


## Support Vector Machines
# So you can see that by defining this concept of margin that sort of quantifies the degree to which the classifier 
# can split the classes into two regions that have some amount of separation between them. We can actually do a search 
# for the classifier that has the maximum margin. This maximum margin classifier is called the Linear Support Vector 
# Machine, also known as an LSVM or a support vector machine with linear kernel.

# Cfloat, default=1.0 Regularization parameter. The strength of the regularization is inversely proportional to C. 
# Must be strictly positive. The penalty is a squared l2 penalty.

# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
# Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
# ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute 
# the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
support = SVC()

# Validation curve
param_range = ['linear', 'poly', 'rbf', 'sigmoid']
train_scores, test_scores = validation_curve(support, X, y, param_name="kernel", param_range=param_range, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

print("TRAIN SCORES:")
print(train_scores)
print("TRAIN SCORES MEAN:")
print(train_scores_mean)

print("TEST SCORES:")
print(test_scores)
print("TEST SCORES MEAN:")
print(test_scores_mean)

plt.title("Validation Curve with SVM")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("SVMValCurve.jpg")
plt.show()


# In[28]:


support = SVC(kernel='linear') 
support.fit(X_train, y_train)
y_pred = support.predict(X_test)
print('Support Vector Machines:')
print('Train set', round(support.score(X_train, y_train), 2))
print('Test set', round(support.score(X_test, y_test), 2))

# Metrics
matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') # Confusion Matrix [TN FP] ; [FN, TP]
print(matrix)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: TN+TP / TN+TP+FN+FP
print('Accuracy:', round(accuracy, 2))

precision = precision_score(y_test, y_pred) # Precision: TP / TP+FP
print('Precision', round(precision,2))

recall = recall_score(y_test, y_pred) # Recall: TP / TP+FN
print('Recall:', round(recall, 2))

f1 = f1_score(y_test, y_pred) # F1 Score (precision+recall): 2TP / 2TP+FN+FP
print('F1 Score:', round(f1, 2))

# Saving these measures in dataframe
Metrics['SVM'] = [accuracy, precision, recall, f1]

# Feature importance
# get importance
importance = support.coef_[0]
# plot feature importance
plt.barh(range(X.shape[1]), importance)
plt.yticks(np.arange(X.shape[1]), df.columns[1:])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("SupportVectorMachinesFeatureImportance.jpg")
plt.show


# In[29]:


Metrics


# In[32]:


# We choose Random Forest for make predictions 
ids = prediction['PassengerId']

predictRanFor = forest.predict(prediction.drop('PassengerId', axis=1))
RanForest = pd.DataFrame({'PassengerId' : ids, 'Survived': predictRanFor})
print('Random Forest Prediction:')
print(RanForest.head())


# In[33]:


plt.scatter(RanForest['PassengerId'], RanForest['Survived'], label='linear')
plt.xlabel('PassengerId')
plt.ylabel('Survived')
plt.savefig("FinalPrediction.jpg")


# In[ ]:




