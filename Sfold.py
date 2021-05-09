# coding:utf-8
import xlrd
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve   #可视化学习的整个过程
from sklearn.model_selection import cross_val_score  #交叉验证
import operator
import pandas as pd
'''
    进行S折交叉检验，并可视化学习过程
'''
# 加载数据
def readdata(dataname,sheetname):
    wb = xlrd.open_workbook(dataname)
    #按工作簿定位工作表
    sh = wb.sheet_by_name(sheetname)
    testset = []
    for i in range(1,sh.nrows):
        line=sh.row_values(i)[3:]
        testset.append(line)  
    data=np.array(testset)
    x = data[:, :-1].astype("float")
    y = data[:, -1].astype("float")
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.25, random_state=7)
    return train_x, test_x, train_y, test_y,x,y
train_x, test_x, train_y, test_y, X, Y=readdata('处理后数据20210426.xlsx','x')

model = svm.SVC(C=100,kernel='linear', decision_function_shape="ovr")
model.fit(train_x, train_y)
predict_y = model.predict(test_x)
#train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
#cv=10采取10折交叉验证
train_sizes,train_scores,test_scores=learning_curve(model,X=X,y=Y,train_sizes=np.linspace(0.1,1.0,4),cv=10)
#统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
#绘制效果
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='Cross-validation')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#lt.ylim([0.8,1.0])
plt.show()







