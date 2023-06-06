from tree import *
from mlp import *
from knn import *
from forest import *
from dnn import *

# get all result
tree_result, tree_cm = tree()
mlp_result, mlp_cm = mlp()
knn_result, knn_cm = knn()
forest_result, forest_cm = forest()
dnn_result, dnn_cm = dnn()

"""
result: 四種指標
{
    "accuracy",
    "tpr",
    "precision",
    "f1_score"
}
"""
result = pd.concat([tree_result, 
                    mlp_result, 
                    knn_result, 
                    forest_result, 
                    dnn_result], axis=1)

print("---COMPARE---")
print(result)
