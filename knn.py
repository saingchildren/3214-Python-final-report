import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from func import Counter, get_data


def knn():
    X, y = get_data()
    model = KNeighborsClassifier()
    knn_counter = Counter()


    # 分成十折，輪流當測試資料
    kf = KFold(10, shuffle=True)
    count = 1

    for train_index, val_index in kf.split(X):
        # train_index: 九成的訓練資料
        # val_index: 一成拿來測試

        # 訓練模型
        model.fit(X[train_index], y[train_index])
        # 使用模型預測
        pred = model.predict(X[val_index])
        # 產生混淆矩陣
        report = classification_report(y[val_index], pred, output_dict=True)
        cm = confusion_matrix(y[val_index], pred)
        knn_counter.add_cm(cm)
        knn_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        count += 1

    return knn_counter.get_result("KNN"), knn_counter.get_total_cm()
