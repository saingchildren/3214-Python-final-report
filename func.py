import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

class Counter():
    def __init__(self):
        self.score = {
            "accuracy": 0,
            "tpr": 0,
            "precision": 0,
            "f1_score": 0
        }
        self.workbook = openpyxl.load_workbook("score.xlsx")
        self.total_cm = pd.DataFrame([[0, 0], [0, 0]])
        self.roc = []
        self.prc = []

    def cm_counter(self, cf):
        TP = cf[0][0]
        FP = cf[0][1]
        FN = cf[1][0]
        TN = cf[1][1]
        # 陽性的樣本中有幾個是預測正確的
        precision = TP / (TP + FP)
        # 事實為真的樣本中有幾個是預測正確的
        tpr = TP / (TP + FN)
        # 正確的機率
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        # 同時考慮Precision和Recall的指標
        f1_score = (2 * tpr * precision) / (tpr + precision)
        sheet = self.workbook["score"]
        sheet.append([accuracy, tpr, precision, f1_score])
        # 插入至資料表
        self.workbook.save("score.xlsx")
        print(pd.DataFrame([accuracy, tpr, precision, f1_score]).T)
        self.score["precision"] += precision
        self.score["tpr"] += tpr
        self.score["accuracy"] += accuracy
        self.score["f1_score"] += f1_score

    def add_cm(self, cm):
        self.total_cm += cm

    def get_total_cm(self):
        return self.total_cm

    def get_result(self, modelName):
        result = (pd.DataFrame([self.score]).T[0] / 10).rename(modelName)
        # 插入至資料表
        sheet_roc = self.workbook["roc"]
        sheet_roc.append(self.roc)
        sheet_roc.append([])
        sheet_roc = self.workbook["prc"]
        sheet_roc.append(self.prc)
        sheet_roc.append([])
        self.workbook.save("score.xlsx")
        return result

    def draw_roc(self, modelName, count, fpr, tpr, roc_auc):
        # roc圖表
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color = "darkorange")
        lw = lw
        plt.plot([0, 1], [0, 1], color = "navy", lw = lw, linestyle = "--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{modelName}ROC Curve {count}")
        plt.legend()
        plt.savefig(f"./{modelName}_roc_img/roc{count}.png")
        plt.clf()
        self.roc.append(roc_auc)
        print("ROC_auc area=%.4f" %(roc_auc))

    def draw_prc(self, modelName, count, lr_recall, lr_precision, lr_auc):
        # prc圖表
        plt.plot([0,1],[1,0],color='navy',lw=2,linestyle='--')
        plt.plot(lr_recall,lr_precision,color="darkorange", lw=2,label='PRC curve (area = %0.4f)'%lr_auc)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PRC Curve")
        plt.legend()
        plt.savefig(f"./{modelName}_prc_img/prc{count}.png")
        plt.clf()
        self.prc.append(lr_auc)
        print("PRC_auc area=%.4f" %(lr_auc))

def get_data():
    data = pd.read_csv("./mobile_train.csv")
    features = list(data.columns[:20])

    # 前20行的特徵值
    X = data[features].to_numpy()
    # 最後一行是標籤
    y = data["price_range"].to_numpy()

    return X, y
