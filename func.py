import pandas as pd

class Counter():
    def __init__(self):
        self.score = {
            "accuracy": 0,
            "tpr": 0,
            "precision": 0
        }

        self.total_cm = pd.DataFrame([[0, 0], [0, 0]])

    def cm_counter(self, cf):
        # 計算精確率、真陽率、準確率
        TP = cf[0][0]
        FP = cf[0][1]
        FN = cf[1][0]
        TN = cf[1][1]
        # 陽性的樣本中有幾個是預測正確的
        precision = TP / (TP + FP)
        # 事實為真的樣本中有幾個是預測正確的
        tpr = TP / (TP + FN)
        # 預測正確的
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        self.score["precision"] += precision
        self.score["tpr"] += tpr
        self.score["accuracy"] += accuracy

    def add_cm(self, cm):
        self.total_cm += cm

    def get_total_cm(self):
        return self.total_cm

    def result_counter(self, scoreName, num):
        self.score[scoreName] += num

    def get_result(self, modelName):
        result = (pd.DataFrame([self.score]).T[0] / 10).rename(modelName)
        return result

def get_data():
    data = pd.read_csv("./mobile_train.csv")
    # 切出前20行
    features = list(data.columns[:20])

    X = data[features].to_numpy()
    y = data["price_range"].to_numpy()

    # return: 特徵值, 標籤
    return X, y
