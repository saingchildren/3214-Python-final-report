from tree import *
from mlp import *
from knn import *
from forest import *
from dnn import *

tree_result, tree_cm = tree()
mlp_result, mlp_cm = mlp()
knn_result, knn_cm = knn()
forest_result, forest_cm = forest()
dnn_result, dnn_cm = dnn()

result = pd.concat([tree_result, 
                    mlp_result, 
                    knn_result, 
                    forest_result, 
                    dnn_result], axis=1)
print("---TREE FINAL CM---")
print(tree_cm)
print("---MLP FINAL CM---")
print(mlp_cm)
print("---KNN FINAL CM---")
print(knn_cm)
print("---FOREST FINAL CM---")
print(forest_cm)
print("---DNN FINAL CM---")
print(dnn_cm)

print("---COMPARE---")
print(result)
