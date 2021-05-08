#!/usr/bin/env python
# coding: utf-8

# In[173]:


import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


INFILE = "C:\\Users\\ejl9900\\Desktop\\Data Science\\MSDS422\\Unit 01 (Week 1 & 2)\\Assignment\\HMEQ_Loss.csv"

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"


# In[3]:


df = pd.read_csv( INFILE )
dt = df.dtypes
print( dt )


# In[4]:


'''
CREATE OBJECT and NUMBER List
'''

objList = []
numList = []
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )
    if dt[i] in (["float64","int64"]) : numList.append( i )


# In[5]:


print(" OBJECTS ")
print(" ------- ")
for i in objList :
    print( i )
print(" ------- ")


# In[6]:


print(" NUMBERS ")
print(" ------- ")
for i in numList :
    print( i )
print(" ------- ")


# In[7]:


'''
EXPLORE THE CATEGORICAL / OBJECT VARIABLES
'''


for i in objList :
    print(" Class = ", i )
    g = df.groupby( i )
    print( g[i].count() )
    x = g[ TARGET_F ].mean()
    print( "Loan Default Prob", x )
    print( " ................. ")
    x = g[ TARGET_A ].mean()
    print( "Loss Amount", x )
    print(" ===============\n\n\n ")


# In[8]:


'''
CHECK FOR MISSING DATA
'''

for i in objList :
    print( i )
    print( df[i].unique() )
    g = df.groupby( i )
    print( g[i].count() )
    print( "MOST COMMON = ", df[i].mode()[0] )   
    print( "MISSING = ", df[i].isna().sum() )
    print( "\n\n")


# In[9]:


"""
FOR OBJECTS - FILL IN MISSING WITH THE MODE
"""
for i in objList :
    if df[i].isna().sum() == 0 : continue
    print( i ) 
    print("HAS MISSING")
    NAME = "IMP_"+i
    print( NAME ) 
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna(df[NAME].mode()[0] )
    print( "variable",i," has this many missing", df[i].isna().sum() )
    print( "variable",NAME," has this many missing", df[NAME].isna().sum() )
    g = df.groupby( NAME )
    print( g[NAME].count() )
    print( "\n\n")
    df = df.drop( i, axis=1 )


# In[10]:


dt = df.dtypes
objList = []
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )


# In[11]:


print(" OBJECTS ")
print(" ------- ")
for i in objList :
    print( i )
print(" ------- ")


# In[12]:


'''
EXPLORE OBJECT VARIABLES ONCE AGAIN AFTER FILLING IN MISSING WITH MODE
'''

for i in objList :
    print(" Class = ", i )
    print( df[i].unique() )
    g = df.groupby( i )
    x = g[ TARGET_F ].mean()
    print( "Loan Default Prob", x )
    print( " ................. ")
    x = g[ TARGET_A ].mean()
    x = g[ TARGET_A ].median()
    print( "Loss Amount", x )
    print(" ===============\n\n\n ")


# In[13]:


"""
CREATE PIE CHARTS FOR OBJECTS
"""

for i in objList :
    x = df[ i ].value_counts(dropna=False)
    #print( x )
    theLabels = x.axes[0].tolist()
    #print( theLabels )
    theSlices = list(x)
    #print( theSlices ) 
    plt.pie( theSlices,
             labels=theLabels,
             startangle = 90,
             shadow=True,
             autopct="%1.1f%%")
    plt.title("Pie Chart: " + i)
    plt.show()


# In[14]:


"""
CREATE HISTOGRAMS FOR NUMBERS
"""

for i in numList :
    plt.hist( df[ i ] )
    plt.xlabel( i )
    plt.show()


# In[15]:


'''
FOR NUMBER VARIABLES - IMPUTE USING THE MEDIAN
'''

for i in numList :
    if df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    #print(i)
    print( df[i].isna().sum() )
    print( FLAG )
    print( IMP )
    print(" ------- ")
    df[ FLAG ] = df[i].isna() + 0
    df[ IMP ] = df[ i ]
    df.loc[ df[IMP].isna(), IMP ] = df[i].median()
    df = df.drop( i, axis=1 )


# In[16]:


"""
FILL IN MISSING WITH THE CATEGORY "MISSING".  ALSO CREATE IMP_ VARIABLES
"""
for i in objList :
    if df[i].isna().sum() == 0 : continue
    NAME = "IMP_"+i
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    g = df.groupby( NAME )
    df = df.drop( i, axis=1 )


# In[17]:


df.head().T


# In[18]:


dt = df.dtypes
objList = []
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )


# In[19]:


"""
CHECK FOR RELATIONSHIPS BETWEEN JOB AND LOAN AMOUNT
"""

g = df.groupby("IMP_JOB")
i = "LOAN"
print( g[i].median() )


# In[20]:


"""
CHECK FOR RELATIONSHIPS BETWEEN JOB AND VALUE OF HOUSE
"""

g = df.groupby("IMP_JOB")
i = "IMP_VALUE"
print( g[i].median() )


# In[21]:


"""
CHECK FOR RELATIONSHIPS BETWEEN JOB AND MORTGAGE DUE
"""

g = df.groupby("IMP_JOB")
i = "IMP_MORTDUE"
print( g[i].median() )


# In[22]:


"""
CHECK FOR RELATIONSHIPS BETWEEN DEFAULT LOAN FLAG AND MORTGAGE DUE
"""

g = df.groupby("TARGET_BAD_FLAG")
i = "IMP_MORTDUE"
print( g[i].median() )


# In[23]:


'''
USE ONE HOT ENCODING TO CONVERT STRING OBJECT VALUES TO NUMERICAL VALUES, THEN DROP OBJECT VARIABLES
'''

for i in objList :
    thePrefix = "z_" + i
    y = pd.get_dummies( df[i], prefix=thePrefix, drop_first=True )   
    y = pd.get_dummies( df[i], prefix=thePrefix )   
    df = pd.concat( [df, y], axis=1 )
    df = df.drop( i, axis=1 )


# In[24]:


df.head().T


# In[25]:


'''
GET SUMMARY STATS FOR EACH VARIABLE
'''

df.describe().T


# In[181]:


#Assignment 2 starts here
#First make a copy of the df
X = df.copy()

#drop target variables from copy since we don't want to use them as predictors
X = X.drop( TARGET_F, axis=1 )
X = X.drop( TARGET_A, axis=1 )
X.head().T


# In[182]:


#STORE TARGET VARIABLES IN pred variable.  Use var TO PREDICT pred
pred = df[ [TARGET_F, TARGET_A] ]
pred.head().T


# In[259]:


#Now we need to split data, 80/20 split for training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(var, pred, train_size=0.8, test_size=0.2, random_state=3)

#in case we want random samples all the time
#X_train, X_test, Y_train, Y_test = train_test_split(var, pred, train_size=0.8, test_size=0.2)

#Breakdown of datasets:
print( "DATASETS" )
print( "TRAINING = ", X_train.shape )
print( "TEST = ", X_test.shape )


# In[260]:


X_test.head().T


# In[261]:


#record nums between var_test and pred_test should be the same
Y_test.head().T


# In[262]:


#First is Decision Tree, import packages from sklearn

from sklearn import tree
from sklearn.tree import _tree
import sklearn.metrics as metrics


# In[428]:


#Calculate loan default probability using decision tree
#try with 4 levels first
#ld1_Tree = tree.DecisionTreeClassifier( max_depth=4 )
ld1_Tree = tree.DecisionTreeClassifier( max_depth=4, min_samples_leaf=200 )

ld1_Tree = ld1_Tree.fit( X_train, Y_train[ TARGET_F ] )


# In[429]:


#Now predict
pred_train_run = ld1_Tree.predict(X_train)
pred_test_run = ld1_Tree.predict(X_test)

#And calculate and display accuracy - try with different samples and levels of tree
print("Accuracy of Train:", metrics.accuracy_score(Y_train[TARGET_F], pred_train_run))
print("Accuracy of Test:", metrics.accuracy_score(Y_test[TARGET_F], pred_test_run))


# In[343]:


#Now create ROC curve
#first get probabilities of loan not defaulting, defaulting for the X_train dataset
probs = ld1_Tree.predict_proba(X_train)
probs[0:10]


# In[344]:


#grab second value of array, which is prob of defaulting
def_probs = probs[:,1]

#now get ROC curve metrics for defaulting probs including false and true positive rates
fpr_train, tpr_train, threshold = metrics.roc_curve( pred_train[TARGET_F], def_probs)


# In[345]:


#check false pos rates for training data set
fpr_train[0:5]


# In[346]:


#check true pos rates for training data set
tpr_train[0:5]


# In[347]:


#find area under the curve
roc_auc_train = metrics.auc(fpr_train, tpr_train)


# In[348]:


#do the same for the test data set:
probs = ld1_Tree.predict_proba(var_test)
def_probs = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( pred_test[TARGET_F], def_probs)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

#save copy of data for later
fpr_tree = fpr_test
tpr_tree = tpr_test
auc_tree = roc_auc_test

#now create the ROC curve
plt.title('Decision Tree ROC Curve')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC Train DS = %0.2f' % roc_auc_train, color = 'blue')
plt.plot(fpr_test, tpr_test, 'b', label = 'AUC Test DS = %0.2f' % roc_auc_test, color="red")
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc = 'lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()


# In[271]:


#View the decision tree using GraphViz
#first get list of variables:
feature_cols = list(var.columns.values)

#then create a GraphViz text file and use GVEdit to view it
tree.export_graphviz(ld1_Tree, out_file='dec_tree.txt',filled=True, rounded=True, feature_names = feature_cols,impurity=False,class_names=["Good","Bad"])


# In[272]:


#Now check which variables can be considered predictors of loan defaulting

#this function can get the var names from the decision tree
def getTreeVars( TREE, varNames ) :
    tree_ = TREE.tree_
    varName = [ varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]

    nameSet = set()
    for i in tree_.feature :
        if i != _tree.TREE_UNDEFINED :
            nameSet.add( i )
    nameList = list( nameSet )
    parameter_list = list()
    for i in nameList :
        parameter_list.append( varNames[i] )
    return parameter_list


feature_cols = list( var.columns.values )
tree.export_graphviz(ld1_Tree,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( ld1_Tree, feature_cols ) 

for i in vars_tree_flag:
    print(i)  


# In[273]:


#Create decision tree for loss amounts for loans that defaulted.  Use regression model since amount is continuous

#if we have loss amounts that are missing (loan did not default), this will set the flag to false
F = ~ pred_train[ TARGET_A ].isna()
pred_train.head().T


# In[274]:


#subset F into X_train and Y_train which will only return true records where loan defaulted
W_train = X_train[F].copy()   #making a copy is better practice
Z_train = Y_train[F].copy()


# In[275]:


Z_train.head().T 


# In[276]:


#do the same for the test data set:

F = ~ Y_test[ TARGET_A ].isna()
W_test = X_test[F].copy() 
Z_test = Y_test[F].copy()


# In[526]:


#create decision tree model to predict loss amounts for loan defaults

amt_m01_Tree = tree.DecisionTreeRegressor( max_depth= 4 )
#amt_m01_Tree = tree.DecisionTreeRegressor( min_samples_leaf = 10 )
amt_m01_Tree = amt_m01_Tree.fit( W_train, Z_train[TARGET_A] )

#predict loss amounts
Z_Pred_train = amt_m01_Tree.predict(W_train)
Z_Pred_test = amt_m01_Tree.predict(W_test)


# In[527]:


#now calculate and print RMSE

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("TREE RMSE Train:", RMSE_TRAIN )
print("TREE RMSE Test:", RMSE_TEST )

#save RMSE results of test
RMSE_TREE = RMSE_TEST


# In[279]:


#once again get variable names that are predictive of losses and create graphviz

feature_cols = list( X.columns.values )
vars_tree_amt = getTreeVars( amt_m01_Tree, feature_cols ) 
tree.export_graphviz(amt_m01_Tree,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )


# In[280]:


#check the varaibles that are predictive of loss amount
for i in vars_tree_amt:
    print(i)


# In[287]:


#Now create random forest to predict prob of defaulting

from sklearn.ensemble import RandomForestRegressor  #used for random forest
from sklearn.ensemble import RandomForestClassifier 

from operator import itemgetter


# In[554]:


#Random forest

ld1_RF = RandomForestClassifier( n_estimators = 25, random_state=1 ) #n_estimators is number of trees to build
#ld1_RF = RandomForestClassifier( n_estimators = 25, random_state=1, class_weight = 'balanced_subsample' )
ld1_RF = ld1_RF.fit( X_train, Y_train[ TARGET_F ] )

Y_Pred_train = ld1_RF.predict(X_train)
Y_Pred_test = ld1_RF.predict(X_test)

print("\n=============\n")
print("RANDOM FOREST\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")


# In[292]:


#now create ROC curve of random forest

#this will grab important variables from random forest:
def getEnsembleTreeVars( ENSTREE, varNames ) :
    importance = ENSTREE.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index :
        imp_val = importance[i]
        if imp_val > np.average( ENSTREE.feature_importances_ ) :
            v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
            theList.append( ( varNames[i], v ) )
    theList = sorted(theList,key=itemgetter(1),reverse=True)
    return theList

probs = ld1_RF.predict_proba(X_train) #predict probability scores for training ds
p1 = probs[:,1]  #get prob that loan defaults
fpr_train, tpr_train, threshold = metrics.roc_curve( Y_train[TARGET_F], p1)  #metrics of ROC curve
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = ld1_RF.predict_proba(X_test)  #predict probability scores for test ds
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

#save data - this is good practice
fpr_RF = fpr_test
tpr_RF = tpr_test
auc_RF = roc_auc_test


#print ROC curve
plt.title('Random Forest ROC CURVE')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC TRAIN = %0.2f' % roc_auc_train, color='blue')
plt.plot(fpr_test, tpr_test, 'b', label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[293]:


#get list of features from random forest
feature_cols = list( X.columns.values )
vars_RF_flag = getEnsembleTreeVars( ld1_RF, feature_cols )

#print them out
for i in vars_RF_flag :
    print( i )


# In[576]:


#now create random forest to predict loss amounts

amt_m01_RF = RandomForestRegressor(n_estimators = 50, random_state=1) #50 decision trees
amt_m01_RF = amt_m01_RF.fit( W_train, Z_train[TARGET_A] )

Z_Pred_train = amt_m01_RF.predict(W_train)
Z_Pred_test = amt_m01_RF.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))  #root mean square error of loss
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("RF RMSE Train:", RMSE_TRAIN )
print("RF RMSE Test:", RMSE_TEST )

RMSE_RF = RMSE_TEST  #save data for later

feature_cols = list( X.columns.values )  #get variables that are most predictive
vars_RF_amt = getEnsembleTreeVars( amt_m01_RF, feature_cols )


# In[565]:


#list predictive variables for loss amounts according to random forest model

for i in vars_RF_amt :
    print( i )


# In[300]:


#Gradient boosting packages

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingClassifier


# In[593]:


#Gradient Boosting

#fm01_GB = GradientBoostingClassifier( random_state=1 )
fm01_GB = GradientBoostingClassifier( random_state=1, learning_rate = .1 )
fm01_GB = fm01_GB.fit( X_train, Y_train[ TARGET_F ] )

Y_Pred_train = fm01_GB.predict(X_train)
Y_Pred_test = fm01_GB.predict(X_test)

print("\n=============\n")
print("GRADIENT BOOSTING\n")
print("Probability of default")
print("Accuracy Train:",metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")

probs = fm01_GB.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve( Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = fm01_GB.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve( Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_GB = fpr_test
tpr_GB = tpr_test
auc_GB = roc_auc_test


feature_cols = list( X.columns.values )
vars_GB_flag = getEnsembleTreeVars( fm01_GB, feature_cols )


# In[308]:


#get predictive variables of defaulting based on gradient boosting

for i in vars_GB_flag :
    print(i)


# In[310]:


#create ROC curve

plt.title('GB ROC CURVE')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC TRAIN = %0.2f' % roc_auc_train, color = 'blue')
plt.plot(fpr_test, tpr_test, 'b', label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[612]:


#check accurarcy of gradient boosting model regarding loss amounts

amt_m01_GB = GradientBoostingRegressor(random_state=1)  #can also set depth of tree with other arguments
#amt_m01_GB = GradientBoostingRegressor(random_state=1, learning_rate = .4)
amt_m01_GB = amt_m01_GB.fit( W_train, Z_train[TARGET_A] )

Z_Pred_train = amt_m01_GB.predict(W_train)
Z_Pred_test = amt_m01_GB.predict(W_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

print("GB RMSE Train:", RMSE_TRAIN )
print("GB RMSE Test:", RMSE_TEST )

RMSE_GB = RMSE_TEST

feature_cols = list( X.columns.values )
vars_GB_amt = getEnsembleTreeVars( amt_m01_GB, feature_cols )


for i in vars_GB_amt :
    print(i)


# In[316]:


#Now compare all 3 models by creating ROC curve of all 3


plt.title('ALL MODELS ROC CURVE')
plt.plot(fpr_tree, tpr_tree, 'b', label = 'AUC TREE = %0.2f' % auc_tree, color="red")  #decision tree
plt.plot(fpr_RF, tpr_RF, 'b', label = 'AUC RF = %0.2f' % auc_RF, color="green")  #random forest
plt.plot(fpr_GB, tpr_GB, 'b', label = 'AUC GB = %0.2f' % auc_GB, color="blue")  #gradient boosting
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("Root Mean Square Average For Losses")
print("TREE", RMSE_TREE)
print("RF", RMSE_RF)
print("GB", RMSE_GB)


# In[ ]:




