import pandas as pd
import numpy as np
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from func import Counter, get_data


def dnn():
    X, y = get_data()
    y = pd.get_dummies(y).to_numpy()
    dnn_counter = Counter()

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
        model.fit(X[train_index], y[train_index], verbose=False, epochs=50)
        # 使用模型預測
        pred = model.predict(X[val_index])
        # 產生混淆矩陣
        report = classification_report(y[val_index].argmax(axis=1), pred.argmax(axis=1))
        cm = confusion_matrix(y[val_index].argmax(axis=1), pred.argmax(axis=1))
        dnn_counter.add_cm(cm)
        dnn_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(confusion_matrix(y[val_index].argmax(axis=1), pred.argmax(axis=1))))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index].argmax(axis=1), pred.argmax(axis=1)))
        count += 1
    return dnn_counter.get_result("DNN"), dnn_counter.get_total_cm()
