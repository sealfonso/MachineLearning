import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
'exec(%matplotlib inline)'

my_data = pd.read_csv("drug200.csv", delimiter =",")
my_data[0:5]


size_data = len(my_data)

x = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
x[0:5]

# Transforming categorical data into numerical
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:,1] = le_sex.transform(x[:,1]) # Posición 2 del arreglo x

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2]) # Posición 3 del arreglo x

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3]) # Posición 4 del arreglo x
x[0:5]

y = my_data["Drug"]
y[0:5]

x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

print(x_trainset.shape, y_trainset.shape)
print(x_testset.shape, y_testset.shape)


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree


drugTree.fit(x_trainset,y_trainset)


predTree = drugTree.predict(x_testset)


print(predTree[0:5])
print(y_testset[0:5])


print("Decission tree's accuracy: ", metrics.accuracy_score(y_testset, predTree))


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')