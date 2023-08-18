import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import accuracy_score

import pickle

import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.ensemble import  RandomForestClassifier
import  warnings
warnings.filterwarnings('ignore')

student=pd.read_csv('D:\pythonProject1Dropout\dataset.csv')

student.shape
student.columns
student.sample(4)
student.head(5)
student.info()

#Check nulls and duplicates
print(student.isnull().sum())
print(student.duplicated().sum())

#Convert the Target to numeric
student['Target'].unique()
student['Target']=student['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})

#Check the Target column
student
student.dtypes
#Learn the data mathematically
student.describe()

#Find the correlation of Target with other numeric columns
student.corr()['Target']
fig=px.imshow(student)
fig.show()

#New df considering relevant input and output columns
student_df=student.iloc[:,[1,11,13,14,15,16,17,20,22,23,26,28,29,34]]

student_df.head()

student_df.info()
sns.heatmap(student_df)

#Exploratory Data Analysis on student_df
#The count of dropouts, enrolled and graduates
student_df['Target'].value_counts()

#plot the above values
x=student_df['Target'].value_counts().index
y=student_df['Target'].value_counts().values

df=pd.DataFrame({
    'Target':x,
    'Count_T':y
})

fig=px.pie(df,
           names='Target',
           values='Count_T',
           title='The count of dropouts, enrolled and graduates')
fig.update_traces(labels=['Graduate','Dropout','Enrolled'],hole=0.4,textinfo='value+label',pull=[0,0.2,0.1])
fig.show()

#The corelation of Target with the rest
student_df.corr()['Target']

#Plot the 2nd sem and 1st sem and differentiate Target bu color
fig=px.scatter(student_df,
               x='Curricular units 1st sem (approved)',
               y='Curricular units 2nd sem (approved)',
               color='Target')
fig.show()


fig=px.scatter(student_df,
               x='Curricular units 1st sem (grade)',
               y='Curricular units 2nd sem (grade)',
               color='Target')
fig.show()

fig=px.scatter(student_df,
               x='Curricular units 1st sem (enrolled)',
               y='Curricular units 2nd sem (enrolled)',
               color='Target')
fig.show()

fig=px.box(student_df,y='Age at enrollment')
fig.show()

#Distribution of age of sudents at the time of enrollment
sns.histplot(data=student_df['Age at enrollment'],kde=True)

#Plot histogram for interactive figure
px.histogram(student_df['Age at enrollment'],x='Age at enrollment',color_discrete_sequence=['red'])

#Extract input and output columns
X=student_df.iloc[:,0:13]
y=student_df.iloc[:,-1]
X

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Training the model

#Logistic Regression

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
#Without Scaling
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("LogisticRegression")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(max_iter=1000,tol=1e-3)
#Without Scaling
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("SGDClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Preceptron

from sklearn.linear_model import  Perceptron
#The same as SGDClassifier(loss="perceptron",eta0=1,learning_rate="constant",penalty=None)
clf=Perceptron(tol=1e-3,random_state=0)
#Without Scaling
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Perceptron")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Logistic Regression CV

from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=5,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("LogisticRegressionCV")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Decision Tree Classifier

from  sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("DecisionTreeClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_depth=10,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Support Vecotor Machines

from sklearn.svm import SVC
#clf=SVC(gamma='auto')
svc=SVC()
parameters={'kernel':('linear','rbf'),'C':[1,10]}
clf=GridSearchCV(svc,parameters)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("SVC")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#NuSVC

from sklearn.svm import NuSVC
clf=NuSVC()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("NuSVC")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Linear SVC

from sklearn.svm import LinearSVC
clf=LinearSVC(random_state=0,tol=1e-5)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("LinearSVC")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Naive Bayes

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("GaussionNB")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

from sklearn.naive_bayes import MultinomialNB
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("MultinomialNB")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("BernoulliNB")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())


#K Nearest Neightbours
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("KNeighborsClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())

#Select Random Forest with Cross Validation
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_depth=10,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())
print("Precision Score;",precision_score(y_test,y_pred,average='macro'))
print("Recall Score:",recall_score(y_test,y_pred,average='macro'))
print("F1 Score:",f1_score(y_test,y_pred,average='macro'))

param_grid={
    'bootstrap':[False,True],
    'max_depth':[5,8,10,20],
    'max_features':[3,4,5,None],
    'min_samples_split':[2,10,12],
    'n_estimators':[100,200,300]
}

rfc=RandomForestClassifier()

clf=GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5,n_jobs=-1,verbose=1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(clf.best_params_)

clf=RandomForestClassifier(bootstrap=False,max_depth=10,max_features=3,
                           min_samples_split=12,
                           n_estimators=100,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestClassifier")
print("Without Scaling and without CV:",accuracy_score(y_test,y_pred))
scores=cross_val_score(clf,X_train,y_train,cv=10)
print("Without Scaling and With CV:",scores.mean())
print("Precision Score;",precision_score(y_test,y_pred,average='macro'))
print("Recall Score:",recall_score(y_test,y_pred,average='macro'))
print("F1 Score:",f1_score(y_test,y_pred,average='macro'))