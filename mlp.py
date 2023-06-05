import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from func import Counter, get_data


def mlp():
    X, y = get_data()
    model = MLPClassifier()
    mlp_counter = Counter()

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
        mlp_counter.add_cm(cm)
        mlp_counter.cm_counter(cm)
        print(f"------ {count} confusion matrix------")
        print(pd.DataFrame(cm))
        print(f"------ {count} classification report------")
        print(classification_report(y[val_index], pred))
        count += 1
    
    return mlp_counter.get_result("MLP"), mlp_counter.get_total_cm()
