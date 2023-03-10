import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold

train_x = np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\github\\training_all feature.csv", delimiter=",")
train_y = [1 for i in range(int(train_x.shape[0] / 2))]  # 250个1
train_y.extend([0 for i in range(int(train_x.shape[0] / 2))])
train_y = np.array(train_y)
#test_x = np.loadtxt("D:\\python\\pycharm\\PyCharm Community Edition 2020.3.5\\pythonProject\\github\\test_all feature.csv", delimiter=",")

def F_score():
    selector = SelectKBest(f_classif, k=500)
    selector.fit(train_x, train_y)  # 在训练集上训练
    transformed_train = selector.transform(train_x)  # 转换训练集
    #transformed_test = selector.transform(test_x)  # 转换测试集
    np.savetxt("F_score_train.csv", transformed_train, delimiter=",")
    #np.savetxt("F_score_test.csv", transformed_test, delimiter=",")

def VarianceSelection():
    selector = VarianceThreshold(threshold=0.00001)
    selector.fit(train_x)  # 在训练集上训练
    transformed_train = selector.transform(train_x)  # 转换训练集
    #transformed_test = selector.transform(test_x)
    np.savetxt("Var_train.csv", transformed_train, delimiter=",")
    #np.savetxt("Var_test.csv", transformed_test, delimiter=",")

def RFECV():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFECV
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    selector = RFECV(estimator=clf, step=1, cv=5)  # 使用5折交叉验证
    # 每一步我们仅删除一个变量
    selector = selector.fit(train_x, train_y)
    transformed_train = train_x[:, selector.support_]  # 转换训练集
    #transformed_test = test_x[:, selector.support_]  # 转换测试集
    np.savetxt("RFECV_train.csv", transformed_train, delimiter=",")
    #np.savetxt("RFECV_test.csv", transformed_test, delimiter=",")

F_score()
VarianceSelection()
RFECV()

