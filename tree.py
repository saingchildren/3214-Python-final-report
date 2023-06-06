import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from func import Counter, get_data


def tree():
    X, y = get_data()
    # 建立模型
    model = DecisionTreeClassifier()
    tree_counter = Counter()

    # 分成十折，輪流當測試資料
    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        # 訓練模型
        model.fit(X[train_index], y[train_index])
        # 使用模型預測
        pred = model.predict(X[val_index])
        # 產生混淆矩陣
        cm = confusion_matrix(y[val_index], pred)
        # 計算prc
        lr_precision, lr_recall, _ = precision_recall_curve(y[val_index], pred)
        # 計算f1
        lr_f1, lr_auc = f1_score(y[val_index], pred), auc(lr_recall, lr_precision)
        # 計算roc
        fpr, tpr, threshold = roc_curve(y[val_index], pred)
        roc_auc = auc(fpr, tpr)

        # 丟回counter加總
        tree_counter.add_cm(cm)
        tree_counter.cm_counter(cm)
        # print
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        tree_counter.draw_roc("tree", count, fpr, tpr, roc_auc)
        tree_counter.draw_prc("tree", count, lr_recall, lr_precision, lr_auc)
        count += 1

    return tree_counter.get_result("TREE"), tree_counter.get_total_cm()
