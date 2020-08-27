import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
#load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target 
df.columns = [
    'sepal length','sepal width','petal length','petal width','label'
]
df.label.value_counts()
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],label = '0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
#plt.show()
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1],data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])