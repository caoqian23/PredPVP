from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix


def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds,shuffle=True, random_state=0)
    oof_train = np.zeros((ntrain,classnum))
    oof_test = np.zeros((ntest,classnum))

    for i,(train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index] # 数据
        kf_y_train = y_train[train_index] # 标签
        kf_X_test = X_train[test_index]  # k-fold的验证集
        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index,0] = clf.predict_proba(kf_X_test)[:,-1]
        oof_train[test_index,1] = clf.predict(kf_X_test)
        oof_test[:,0] += clf.predict_proba(X_test)[:,-1]
        oof_test[:,1] +=clf.predict(X_test)
    oof_test = oof_test/float(n_folds)
    for i in range(ntest-1):
        if oof_test[i,-1]>0.5:
            oof_test[i,-1]=1
        else:
            oof_test[i, -1]=0
    return oof_train, oof_test

def SelectModel(modelname):

    if modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn(n_neighbors=4)

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50)

    elif modelname == "XGboost":
        from xgboost.sklearn import XGBClassifier
        model = XGBClassifier(n_estimators=100)

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100)

    elif modelname == "EF":
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=150)

    elif modelname == "LightGBM":
        from lightgbm.sklearn import LGBMClassifier
        model = LGBMClassifier(n_estimators=180)

    elif modelname == "ANN":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(10,10))

    elif modelname == "SVM":
        from sklearn import svm
        model = svm.SVC(probability=True,C=32,gamma=0.0078125)
    else:
        pass
    return model


x1=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\training set\\norm\\norm_F_train.csv",delimiter=",")
x2=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\training set\\norm\\norm_Var_train.csv",delimiter=",")
x3=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\training set\\norm\\norm_RFECV_train.csv",delimiter=",")
y = [1 for i in range(int(x1.shape[0]/2))]  # 250个1
y.extend([0 for i in range(int(x1.shape[0]/2))])
y = np.array(y)
test_x1=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\test set\\norm_test\\norm_F_test.csv",delimiter=",")
test_x2=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\test set\\norm_test\\norm_Var_test.csv",delimiter=",")
test_x3=np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\test set\\norm_test\\norm_RFECV_test.csv",delimiter=",")
modelist = ['KNN', 'RF', 'XGboost', 'GBDT', 'EF', 'LightGBM', 'ANN', 'SVM']
newtrfeature_list = []
newtefeature_list = []
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train_ ,oof_test_= get_oof(clf=clf_first,n_folds=10,X_train=x1,y_train=y,X_test=test_x1)
    newtrfeature_list.append(oof_train_)
    newtefeature_list.append(oof_test_)
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train2_ ,oof_test2_= get_oof(clf=clf_first,n_folds=10,X_train=x2,y_train=y,X_test=test_x2)
    newtrfeature_list.append(oof_train2_)
    newtefeature_list.append(oof_test2_)
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train3_ ,oof_test3_= get_oof(clf=clf_first,n_folds=10,X_train=x3,y_train=y,X_test=test_x3)
    newtrfeature_list.append(oof_train3_)
    newtefeature_list.append(oof_test3_)

import functools
newtrfeature = functools.reduce(lambda x,y:np.concatenate((x,y),axis=1),newtrfeature_list)
newtefeature = functools.reduce(lambda x,y:np.concatenate((x,y),axis=1),newtefeature_list)

#np.savetxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\change5\\newfeature.csv", newfeature, delimiter=",")
#np.savetxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\model\\change5\\newtestdata.csv", newtestdata, delimiter=",")

testy = [1 for i in range(int(newtefeature.shape[0]/2))]  # 250个1
testy.extend([0 for i in range(int(newtefeature.shape[0]/2))])
testy = np.array(testy)

clf_second = LogisticRegression()
clf_second.fit(newtrfeature, y)
pred = clf_second.predict(newtefeature)
probas=clf_second.predict_proba(newtefeature)
accuracy = metrics.accuracy_score(testy, pred)
print(accuracy)

confusion = confusion_matrix(testy, pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print("ROC:{}".format(roc_auc_score(testy, probas[:, 1])))
print("SP:{}".format(TN / (TN + FP)))
print("SN:{}".format(TP / (TP + FN)))
n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
print("PRE:{}".format(TP / (TP + FP)))
print("MCC:{}".format(n))
print("F-score:{}".format((2 * TP) / (2 * TP + FP + FN)))
print("ACC:{}".format((TP + TN) / (TP + FP + TN + FN)))