import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data = pd.read_csv('titanic.csv')

num_col = data.select_dtypes(include=['int64','float64'])
cat_col = data.select_dtypes(include=['object'])

data.shape
data.head()
data.info()
data.describe()
data.isnull().sum()

plt.figure(figsize=(15,10))
sns.boxplot(data)
plt.title('Nonscaled Boxplot')
plt.xticks(rotation=90)
plt.show()

cor_matrix = num_col.corr()
sns.heatmap(cor_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Marix')
plt.show()

data.columns.tolist()

data.hist()
plt.show()

sns.pairplot(data=num_col, markers='.')
plt.show()

scat_mat = scatter_matrix(data) #same shit pairplot
plt.show()

data['Survived'].value_counts()
data['Survived'].value_counts().plot(kind='pie',autopct='%1.0f%%', labels=['not survived', 'survived'])
plt.title('Pie chart of survived and not survived')
plt.show()

data['Age'].value_counts()
data['Age'] = data['Age'].round()
data['Age'].head()
data.describe()

data['Pclass'].value_counts()
data['Pclass'].value_counts().plot(kind='pie',autopct='%1.0f%%', labels=['1st class', '2nd class', '3rd class'])
plt.title('Pie chart of classes')
plt.show()

data['Embarked'].value_counts()
data['Embarked'].value_counts().plot(kind='pie',autopct='%1.0f%%',labels=['S','C','Q'])
plt.title('Pie chart of Embarked')
plt.show()

#stacked bar of survival based on sex(hehe:D)
df = (data
      .groupby('Sex')['Survived']
      .value_counts(normalize=True)
      .mul(100)
      .round(2)
      .unstack())
df