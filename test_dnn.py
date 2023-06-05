import pandas as pd
import numpy as np
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from func import Counter, get_data


def dnn():
    X, y = get_data()
    y = pd.get_dummies(y).to_numpy()
    dnn_counter = Counter()

    model = Sequential()
    model.add(Dense(512, input_dim = 20, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        pred = model.predict(X[val_index])
        cm = confusion_matrix(y[val_index], pred)
        c_y_test, c_pred = dnn_counter.change_data_form(y[val_index], pred)
        lr_precision, lr_recall, _ = precision_recall_curve(c_y_test, pred)
        lr_f1, lr_auc = f1_score(c_y_test, c_pred), auc(lr_recall, lr_precision)
        fpr, tpr, threshold = roc_curve(c_y_test, c_pred)
        roc_auc = auc(fpr, tpr)
        dnn_counter.add_cm(cm)
        dnn_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        dnn_counter.draw_roc("tree", count, fpr, tpr, roc_auc)
        dnn_counter.draw_prc("tree", count, lr_recall, lr_precision, lr_auc)
        count += 1

    return dnn_counter.get_result("DNN"), dnn_counter.get_total_cm()

dnn()
