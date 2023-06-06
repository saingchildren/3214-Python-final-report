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

    # 建立模型
    model = Sequential()
    # 輸入層
    model.add(Dense(512, input_dim = 20, activation="relu"))
    # 隱藏層
    model.add(Dense(256, activation="relu"))
    # 輸出層
    model.add(Dense(2, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    # 分成十折，輪流當測試資料
    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        # 訓練模型
        model.fit(X[train_index], y[train_index])
        # 使用模型預測
        pred = model.predict(X[val_index])
        # 產生混淆矩陣
        cm = confusion_matrix(y[val_index].argmax(axis=1), pred.argmax(axis=1))
        # 計算prc
        lr_precision, lr_recall, _ = precision_recall_curve(y[val_index].argmax(axis=1), pred.argmax(axis=1))
        # 計算f1
        lr_f1, lr_auc = f1_score(y[val_index].argmax(axis=1), pred.argmax(axis=1)), auc(lr_recall, lr_precision)
        # 計算roc
        fpr, tpr, threshold = roc_curve(y[val_index].argmax(axis=1), pred.argmax(axis=1))
        roc_auc = auc(fpr, tpr)

        # 丟回counter加總
        dnn_counter.add_cm(cm)
        dnn_counter.cm_counter(cm)
        # print
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index].argmax(axis=1), pred.argmax(axis=1)))
        dnn_counter.draw_roc("dnn", count, fpr, tpr, roc_auc)
        dnn_counter.draw_prc("dnn", count, lr_recall, lr_precision, lr_auc)
        count += 1

    return dnn_counter.get_result("DNN"), dnn_counter.get_total_cm()
