import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from func import Counter, get_data
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score


def tree():
    X, y = get_data()
    model = DecisionTreeClassifier()
    tree_counter = Counter()

    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        pred = model.predict(X[val_index])
        cm = confusion_matrix(y[val_index], pred)
        c_y_test, c_pred = tree_counter.change_data_form(y[val_index], pred)
        lr_precision, lr_recall, _ = precision_recall_curve(c_y_test, pred)
        lr_f1, lr_auc = f1_score(c_y_test, c_pred), auc(lr_recall, lr_precision)
        fpr, tpr, threshold = roc_curve(c_y_test, c_pred)
        roc_auc = auc(fpr, tpr)
        tree_counter.add_cm(cm)
        tree_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        tree_counter.draw_roc("tree", count, fpr, tpr, roc_auc)
        tree_counter.draw_prc("tree", count, lr_recall, lr_precision, lr_auc)
        count += 1

    return tree_counter.get_result("TREE"), tree_counter.get_total_cm()

tree()
