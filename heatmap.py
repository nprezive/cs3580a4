import pandas as pd
import seaborn as sns; 
import matplotlib.pyplot as plt
from   pandas.plotting import parallel_coordinates
import numpy as np

np.random.seed(0)
sns.set(style="ticks", color_codes=True)

data = pd.read_csv('train.csv', sep=',')

femaleMask = data['Sex'] == 'female'
data.loc[femaleMask,'Sex'] = 1
#or in one line:
data.loc[data['Sex'] == 'male','Sex'] = 0

data['AgeBucket'] = 1000

print(data['Age'].describe())

#note: All NA's will be '1000'
data.loc[data['Age'] <= 20,'AgeBucket'] = 20
data.loc[(data['Age'] > 20) & (data['Age'] <= 40),'AgeBucket'] = 40
data.loc[(data['Age'] > 40) & (data['Age'] <= 60),'AgeBucket'] = 60
data.loc[data['Age'] > 60,'AgeBucket'] = 80

#table = data.pivot_table(values='Survived',index=['Sex'], columns=['Pclass'])
table = data.pivot_table(values='Survived',index=['Sex'], columns=['Pclass','AgeBucket'])

print(table)

sns.heatmap(table, annot=True)#, fmt="g", cmap='viridis')


#uniform_data = np.random.rand(10, 12)
#ax = sns.heatmap(uniform_data)
#ax = sns.heatmap(iris)
plt.show()
#sns.heatmap(data)

#replace 'female' with '1' and 'male' with 0:
#femaleMask = data['Sex'] == 'female'
#data.loc[femaleMask,'Sex'] = 1
#or in one line:
#data.loc[data['Sex'] == 'male','Sex'] = 0

#print(data['Sex'].describe())
#print(data['Fare'].describe())
#print(data['Pclass'].describe())
#print(data['Age'][0:10])

#data['Age'].fillna(0, inplace=True)

#print(data['Age'][0:10])

#since the current parallel_coordinates only does one column of normalization, we will manually normalize:

#data.loc[data['Sex'] == 1,'Sex'] = 500
#data.loc[data['Pclass'] == 2,'Pclass'] = 250
#data.loc[data['Pclass'] == 3,'Pclass'] = 500

#subset = data[['Survived','Sex','Pclass','Age']]#,'Fare']]
#plt.figure()
#g = sns.pairplot(subset)

#parallel_coordinates(subset, 'Survived')