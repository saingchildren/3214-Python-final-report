# before run python all.py

> #### 1.要確保 score.xlsx 在資料夾內, 並同時確保有score、roc、prc三個資料表

> #### 2.每次要重新跑 all.py 時候必須確保 score.xlsx 內的三個資料表都為空

# after run python all.py

> #### score的score工作表的資料會是50列4行、以10列區分，分別是tree、mlp、knn、forest、dnn, 欄位分別是accuracy、tpr、precision、f1_score
> #### roc、prc的資料表都為5列10行，以列區分tree、mlp、knn、forest、dnn
