# coding=utf-8
from time import time
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from gensim.models import word2vec
import os
import sys

from vsm import getVSM

reload(sys)
sys.setdefaultencoding('utf-8')

log_path = 'log/log.txt'
data_path = "../data/"


# 读取原始训练和测试数据转化为id
def get_data(data_path=None):
    train_input_path = os.path.join(data_path, "train_input_data.txt")
    train_label_path = os.path.join(data_path, "train_label_data.txt")
    test_input_path = os.path.join(data_path, "test_input_data.txt")
    test_label_path = os.path.join(data_path, "test_label_data.txt")

    file = open(train_input_path, 'r')
    train_input_data = list(file.readlines())
    file.close()

    file = open(train_label_path, 'r')
    train_label_data = list(file.readlines())
    file.close()

    file = open(test_input_path, 'r')
    test_input_data = list(file.readlines())
    file.close()

    file = open(test_label_path, 'r')
    test_label_data = list(file.readlines())
    file.close()

    return train_input_data, train_label_data, test_input_data, test_label_data


# SVM分类
def SVMClassify(data, label):
    t0 = time()
    param_grid = {
        'C': [0.5],#, 8
        'kernel': ['rbf'],#'linear',
        'gamma': [2] #, 10
    }
    svm_clf = SVC(decision_function_shape='ovo')
    grid = RandomizedSearchCV(svm_clf, param_grid, cv=5, n_iter=1, random_state=5, scoring='precision_weighted',
                              refit=True)
    grid.fit(data, label)
    mylog = open(log_path, 'a')
    print '**************** SVM param ******************'
    mylog.write('**************** SVM param ******************\n')
    print '用时：', time() - t0
    mylog.write('用时：' + str(time() - t0) + '\n')
    print '准确率：', grid.best_score_
    mylog.write('准确率：' + str(grid.best_score_) + '\n')
    print '最优参数：', grid.best_params_
    mylog.write('最优参数：' + str(grid.best_params_) + '\n')

    joblib.dump(grid, 'model/svm_model.m')
    # return [grid.best_score_, svm_clf]


# 朴素贝叶斯
def NaiveBayesClassify(data, label):
    t0 = time()
    param_grid = {
        'alpha': [ 0.8, 0.9, 1]
    }
    NB_clf = MultinomialNB()
    grid = GridSearchCV(NB_clf, param_grid, cv=5, n_jobs=-1, scoring='precision_weighted', refit=True)
    grid.fit(data, label)

    mylog = open(log_path, 'a')
    print '**************** Navie Bayse param ******************'
    mylog.write('**************** Navie Bayse param ******************\n')
    print '用时：', time() - t0
    mylog.write('用时：' + str(time() - t0) + '\n')
    print '准确率：', grid.best_score_
    mylog.write('准确率：' + str(grid.best_score_) + '\n')
    print '最优参数：', grid.best_params_
    mylog.write('最优参数：' + str(grid.best_params_) + '\n')
    joblib.dump(grid, 'model/nb_model.m')
    # return [grid.best_score_, NB_clf]



if __name__ == '__main__':
    train_input_data, train_label_data, test_input_data, test_label_data = get_data(data_path)

    getVSM(train_input_data, train_label_data)
    tfidf = joblib.load('model/tfidf_model.m')
    x_train = tfidf.transform(train_input_data)

    # NaiveBayes 分类器
    NaiveBayesClassify(x_train, train_label_data)

    # svm 分类器
    SVMClassify(x_train, train_label_data)











    # 影响预测准确度的因素主要来自于：获取其周围文本的数量，其左右各一条，两条，三条？经测试（1000），三条（38.48%） < 两条（44.97%） < 一条（52.68%）
    # svc的kernel的取值，经测试当kernel取值为linear时，准确率更高
    # svm的参数C的设置：C

    # score = ‘weighted’效果更好
    # 使用训练数据集做测试的时候，效果和交叉验证的差不多
