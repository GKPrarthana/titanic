import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

data = pd.read_csv('titanic.csv')

#when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)'
data.isnull().sum()
#num_col['Age'].fillna(num_col['Age'].median(),inplace=True)
data.fillna({'Age':data['Age'].median()},inplace=True)
data['Age'].isnull().sum()

data['Travelalone'] = np.where((data['SibSp'] + data['Parch']) > 0, 0, 1).astype('int64')
data = data.drop(columns=['SibSp','Parch','PassengerId','Name','Ticket','Cabin'])
data.head()

data.fillna({'Embarked':data['Embarked'].mode()},inplace=True)
data.isnull().sum()

num_col = data.select_dtypes(include=['float64','int64'])
cat_col = data.select_dtypes(include=['object'])

num_col.columns.tolist()
cat_col.columns.tolist()

plt.figure(figsize=(15,10))
sns.boxplot(num_col)
plt.title('Boxplot')
plt.xticks(rotation=90)
plt.show()

num_col.head()
#encoding
data.head()
data = pd.get_dummies(data, columns=['Sex','Embarked'])
data.head()
data.info()

scale_X = StandardScaler()

X = pd.DataFrame(scale_X.fit_transform(data.drop(['Survived'],axis=1),),columns=data.columns.drop(['Survived']))
X.head()
X.columns.tolist()

y = data['Survived']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train.head()

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train,y_train)

y_pred_lr = log_reg_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_lr)*100
print(f'Accuracy of Logistic Regression Model: {round(accuracy,2)} %')

from sklearn.model_selection import cross_val_score, StratifiedKFold
cv_method = StratifiedKFold(n_splits=3)
#cross validate the LR model
cv_score_lr = cross_val_score(
                            log_reg_model,
                            X_train, y_train,
                            cv = cv_method,
                            n_jobs=2,
                            scoring='accuracy'
                        )
print(f'Scores(cross validate) for Logistic Regression model: \n{cv_score_lr}')
print(f'CrossValMeans: {round(cv_score_lr.mean(),3)*100}%')
print(f'CrossValStandard Deviation: {round(cv_score_lr.std(),3)}')

#hyper parameter tunning LR
params_lr = {'tol': [0.0001,0.0002,0.0003],
             'C': [0.01,0.1,1,10,100],
             'intercept_scaling': [1,2,3,4]
            }
gridsearchcv_lr = GridSearchCV(estimator=log_reg_model,
                               param_grid=params_lr,
                               cv=cv_method,
                               verbose=1,
                               n_jobs=2,
                               scoring='accuracy',
                               return_train_score=True
                               )
#fit model with train data
gridsearchcv_lr.fit(X_train,y_train)

best_estimator_lr = gridsearchcv_lr.best_estimator_
print(f'Best estimator for LR Model: \n {best_estimator_lr}')

best_params_lr =gridsearchcv_lr.best_params_
print(f'Best parameter values for LR model: {round(gridsearchcv_lr.best_score_, 3)*100}%')

#check model performance LR
print('Classification Report')
print(classification_report(y_test,y_pred_lr))
#confution matrix LR
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_lr))

ax = plt.subplot()
cf = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cf, annot=True, fmt='0.3g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['not survived','survived']);
plt.show()