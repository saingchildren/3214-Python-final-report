import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from func import Counter, get_data


def mlp():
    X, y = get_data()
    model = MLPClassifier()
    mlp_counter = Counter()

    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        pred = model.predict(X[val_index])
        cm = confusion_matrix(y[val_index], pred)
        c_y_test, c_pred = mlp_counter.change_data_form(y[val_index], pred)
        lr_precision, lr_recall, _ = precision_recall_curve(c_y_test, pred)
        lr_f1, lr_auc = f1_score(c_y_test, c_pred), auc(lr_recall, lr_precision)
        fpr, tpr, threshold = roc_curve(c_y_test, c_pred)
        roc_auc = auc(fpr, tpr)
        mlp_counter.add_cm(cm)
        mlp_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        mlp_counter.draw_roc("mlp", count, fpr, tpr, roc_auc)
        mlp_counter.draw_prc("mlp", count, lr_recall, lr_precision, lr_auc)
        count += 1
    
    return mlp_counter.get_result("MLP"), mlp_counter.get_total_cm()
