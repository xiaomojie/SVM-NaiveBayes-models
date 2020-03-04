# coding=utf-8
from sklearn import metrics
from sklearn.externals import joblib
import os

# 读取原始训练和测试数据转化为id
def get_data(data_path=None):
    train_input_path = os.path.join(data_path, "train_input_data.txt")
    train_label_path = os.path.join(data_path, "train_label_data.txt")
    test_input_path = os.path.join(data_path, "test_input_data.txt")
    test_label_path = os.path.join(data_path, "test_label_data.txt")

    file = open(train_input_path, 'r')
    train_input_data = list(file.readlines())
    file.close()
    # print(len(train_input_data))

    file = open(train_label_path, 'r')
    train_label_data = list(file.readlines())
    file.close()

    file = open(test_input_path, 'r')
    test_input_data = list(file.readlines())
    file.close()
    # print(len(test_input_data))

    file = open(test_label_path, 'r')
    test_label_data = list(file.readlines())
    file.close()

    return train_input_data, train_label_data, test_input_data, test_label_data

def test(x_test, y_test):
    mylog = open(log_path, 'a')
    mylog.write('&&&&&&&&&&&&&&&&&&&&&&&test&&&&&&&&&&&&&&&&&&：\n')
    print 'NaiveBayes 分类器：'
    mylog.write('NaiveBayes 分类器：\n')
    nb_clf = joblib.load('model/nb_model.m')
    nb_predicted = nb_clf.predict(x_test)
    # print y_test
    # print nb_predicted
    # print metrics.accuracy_score(y_test, nb_predicted)
    # nb_tuple = metrics.precision_recall_fscore_support(y_test, nb_predicted, average=None)#, average='weighted')
    # print nb_tuple
    nb_tuple = metrics.classification_report(y_test, nb_predicted)
    print type(nb_tuple)
    print nb_tuple
    mylog.write(nb_tuple)
    # print nb_tuple
    # print '精确度：', nb_tuple[0]
    # print '召回率：', nb_tuple[1]
    # print 'F1值：', nb_tuple[2]
    # mylog.write('精确度：' + str(nb_tuple[0]) + '\n')
    # mylog.write('召回率：' + str(nb_tuple[1]) + '\n')
    # mylog.write('F1值：' + str(nb_tuple[2]) + '\n')

    # print 'KNN 分类器：'
    # mylog.write('KNN 分类器：\n')
    # knn_clf = joblib.load('model/knn_model.m')
    # knn_predicted = knn_clf.predict(x_test)
    # knn_tuple = metrics.precision_recall_fscore_support(y_test, knn_predicted, average='weighted')
    # print knn_tuple
    # print '精确度', knn_tuple[0]
    # print '召回率：', knn_tuple[1]
    # print 'F1值：', knn_tuple[2]
    # mylog.write('精确度：' + str(knn_tuple[0]) + '\n')
    # mylog.write('召回率：' + str(knn_tuple[1]) + '\n')
    # mylog.write('F1值：' + str(knn_tuple[2]) + '\n')
    #
    print 'SVM 分类器：'
    mylog.write('SVM 分类器：\n')
    svm_clf = joblib.load('model/svm_model.m')
    svm_predicted = svm_clf.predict(x_test)
    # # svm_tuple = metrics.precision_recall_fscore_support(y_test, svm_predicted, average='weighted')
    svm_tuple = metrics.classification_report(y_test, svm_predicted)
    # print type(svm_tuple)
    # print svm_tuple
    mylog.write(svm_tuple)
    # print svm_tuple
    # print '精确度', svm_tuple[0]
    # print '召回率：', svm_tuple[1]
    # print 'F1值：', svm_tuple[2]
    # mylog.write('精确度：' + str(svm_tuple[0]) + '\n')
    # mylog.write('召回率：' + str(svm_tuple[1]) + '\n')
    # mylog.write('F1值：' + str(svm_tuple[2]) + '\n')

log_path = 'log/log.txt'
data_path = "../data/"

if __name__ == '__main__':
    train_input_data, train_label_data, test_input_data, test_label_data = get_data(data_path)
    tfidf = joblib.load('model/tfidf_model.m')
    x_test = tfidf.transform(test_input_data)
    test(x_test, test_label_data)