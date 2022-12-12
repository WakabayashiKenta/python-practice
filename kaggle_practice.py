#RMSE
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

y_true = [1.0,2.0,1.5,5.0,1.4]
y_pred = [1.0,1.9,1.3,3.4,1.2]

rmse = np.sqrt(mean_squared_error(y_true,y_pred))
print(rmse)
#0.7280109889280518

#混合行列
from sklearn.metrics import confusion_matrix

#0と1で表される二値分類の真の値と予測値
y_true = [1,0,1,1,0,1]
y_pred = [1,1,0,0,0,1]

#陽性はpositive 陰性はnegativeなのでtp,fpとかになる.予測がpositiveで真の値もpositiveは真陽性、予測がnegativeで真の値がpositiveは偽陰性
tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp,tn],
                              [fp,fn]])
print(confusion_matrix1)
#[[2 1]
# [1 2]]

#sklearnのモジュールを使う。ただしnumpyで作った上の混合行列とは並びが違う
confusion_matrix2 = confusion_matrix(y_true,y_pred)
print(confusion_matrix2)

#予測が正しい割合のaccuracyと誤っている割合のerror rateを求める
from sklearn.metrics import accuracy_score

y_true = [1,0,1,1,0,1]
y_pred = [1,1,0,0,0,1]

accuracy = accuracy_score(y_true,y_pred)

print(f'正答率は{accuracy},誤答率は{1-accuracy}です')
#正答率は0.5,誤答率は0.5です

#logloss
from sklearn.metrics import log_loss

y_true = [1,0,1,1,0,1]
y_pred = [0.1,0.2,0.8,0.9,0.2,0.9]

logloss = log_loss(y_true,y_pred)
print(logloss) 
#0.5304561297087212

from sklearn.metrics import log_loss

#三クラス分類の真の値と予測値
y_true = np.array([0,2,1,2,1])
y_pred = np.array([[0.68,0.32,0.00],
                   [0.00,0.00,1.00],
                   [0.60,0.40,0.00],
                   [0.00,0.00,1.00],
                   [0.28,0.12,0.60]])
logloss = log_loss(y_true,y_pred)
print(logloss)

from sklearn.metrics import f1_score

y_true = np.array([[1,1,0],
                   [1,0,0],
                   [1,1,1],
                   [0,1,1],
                   [0,0,1]])

y_pred = np.array([[1,0,1],
                   [0,1,0],
                   [1,0,1],
                   [0,0,1],
                   [0,0,1]])

#mean_f1ではレコードごとにF1-scoreを計算
mean_f1 = np.mean(f1_score(y_true[i,:],y_pred[i,:]) for i in range(len(y_true)))

#macro_f1ではクラスごとにF1-scoreを計算
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:,c],y_pred[:,c]) for c in range(len(n_class))])

#micro-f1ではレコード×クラスのペアごとにTP/TN/FP/FNを計算してF1-scoreを計算する
micro_f1 = f1_score(y_true.reshape(-1),y_pred.reshape(-1))

#sklearnのメソッドを使えば簡単
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

from sklearn.metrics import confusion_matrix, cohen_kappa_score

def quadratic_weight_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]): #0は縦方向
        for j in range(c_matrix.shape[1]): #１は横方向
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i,j]
            eij = c_matrix[i:j].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij
    return 1.0 - numer /denom

y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

c_matrix = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

kappa = quadratic_weight_kappa(c_matrix)

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')






#欠損値を表す文字が格納されているときの対処
train = pd.read_csv('train_csv', na_values=['',' NA', -1, 9999])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_x[num_cols])

train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])