#RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

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