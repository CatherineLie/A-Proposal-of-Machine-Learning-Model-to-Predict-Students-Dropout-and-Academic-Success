import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.ensemble import  RandomForestClassifier
import  warnings
warnings.filterwarnings('ignore')

import  os
for dirname,_, filenames in os.walk('D:\pythonProject1Dropout'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

df=pd.read_csv('dataset.csv')
df.head()#Display the first few rows of the dataset.

df.isna().sum()#Calculate the number of missing value in each column.
df.duplicated().sum()#Calculate the number of duplicate records.
df.info()#Provide an overall overview of the dataset.

from pandas.io.formats.format import set_option
set_option('display.precision',2)#Display floating-point values in DataFrame to 2 decimal places.
df.describe()#Generate descriptive statistics of the numerical columns

corr=df.corr(numeric_only=True)#Calculate the correlation matrix of the DataFrame df and stored in corr.
fig,ax=plt.subplots(figsize=(80,60))#Create a figure and axes object for plotting the heatmap.
sns.heatmap(corr,cmap='coolwarm',annot=True,square=True)#Correlation matrix.
plt.title("Correlation Heatmap between Features")
plt.show()

sns.set_theme(style='darkgrid')
df.hist(bins=10,figsize=(80,60),grid=True,legend=None)
df.describe(include='all').loc['unique',:]#Generate a summary statistics table for df.
plt.show()

s_df=df.copy()#Copy data to modify.
s_df.shape#Return the shape.

s_df['Target']=LabelEncoder().fit_transform((s_df['Target']))#Encode the label.
s_df.loc[:,'Target'].value_counts()##Retrieve the value counts.

z_scores=np.abs(stats.zscore((s_df)))#Deviations a data point is away from the mean.
outliers=np.where(z_scores>3.2)#Identify the outliers in s_df.
outliers#Contains the indices of the outliers.

out_df=s_df.drop(s_df.index[outliers[0]])#Remove the rows containing outliers from s_df.
s_df=out_df.reset_index(drop=True)#Reset the index of the updated Dataframe out_df.
s_df.shape#Dimension.
s_df

s_df.drop(s_df[s_df["Target"]==1].index,inplace=True)#Drop the rows from s_df where Target==1.
s_df.loc[:,'Target'].value_counts()

s_df.loc[:,'Target'].value_counts()

x=s_df.drop(columns=['Target'],axis=1)
z=s_df["Target"]

scaler=StandardScaler()
scaled=scaler.fit_transform(x)#Calculate the mean and standard deviation.

X_trian, X_test, y_train, y_test=train_test_split(scaled, z, test_size=0.2, random_state=10)
logreg=LogisticRegression()
logreg.fit(X_trian,y_train)
logreg.score(X_test, y_test)

print('Training Accuracy:',logreg.score(X_trian,y_train))
print('Testing Accuracy:', logreg.score(X_test, y_test))

y_pred=logreg.predict(X_test)
print('\nCLASSFICATION REPORT\n')
print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
