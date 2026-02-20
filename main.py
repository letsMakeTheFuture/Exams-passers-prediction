import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('/kaggle/input/exams-passers/train (1).csv')
test = pd.read_csv('/kaggle/input/exams-passers/test (2).csv')
print(train.head())
print(train.info())
X =train.drop(['id','pass'],axis =1)
y = train['pass']
X_test = test.drop(['id'],axis=1)

X=pd.get_dummies(X)
X_test=pd.get_dummies(X_test)
X,X_test = X.align(X_test,join='left',axis=1,fill_value=0)
print ("preprocessiog complete text is now numbers")

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)
predictions = model.predict(X_test)

submission = pd.DataFrame({'id':test['id'],'pass':predictions})

submission.to_csv('submission.csv',index=False)
print ("we made it ")

print(submission.head())
print(submission['pass'].value_counts())
