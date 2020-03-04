# coding=utf-8
from sklearn import metrics
from sklearn.externals import joblib
import os

# 读取原始训练和测试数据转化为id
def get_data(data_path=None):
    test_input_data = []
    for i in range(7):
        test_input_path = os.path.join(data_path, str(i+1) + '/')
        num = len(os.listdir(test_input_path))
        for j in range(num):
            file = open(test_input_path + str(j+1), 'r')
            test_input_data += list(file.readlines())
            file.close()

    print(len(test_input_data))
    return test_input_data

def test(x_test, y_test):
    mylog = open(log_path, 'a')
    # mylog.write('&&&&&&&&&&&&&&&&&&&&&&&test&&&&&&&&&&&&&&&&&&：\n')
    print 'NaiveBayes 分类器：'
    mylog.write('NaiveBayes 分类器：\n')
    nb_clf = joblib.load('model/nb_model.m')
    nb_predicted = nb_clf.predict(x_test)
    # print y_test
    # print nb_predicted
    # nb_tuple = metrics.accuracy_score(y_test, nb_predicted)
    # nb_tuple = metrics.precision_recall_fscore_support(y_test, nb_predicted, average=None)#, average='weighted')
    # print nb_tuple
    nb_tuple = metrics.classification_report(y_test, nb_predicted)
    # print type(nb_tuple)
    print nb_tuple
    mylog.write(str(nb_tuple)+'\n')

    #
    print 'SVM 分类器：'
    mylog.write('SVM 分类器：\n')
    svm_clf = joblib.load('model/svm_model.m')
    svm_predicted = svm_clf.predict(x_test)
    # print y_test
    # print svm_predicted
    # svm_tuple = metrics.precision_recall_fscore_support(y_test, svm_predicted, average='weighted')
    # svm_tuple = metrics.accuracy_score(y_test, svm_predicted)

    svm_tuple = metrics.classification_report(y_test, svm_predicted)
    print svm_tuple
    mylog.write(str(svm_tuple)+'\n')


log_path = '../table/sci_log.txt'
data_path = "../table/"

if __name__ == '__main__':
    test_input_data = get_data(data_path)
    tfidf = joblib.load('model/tfidf_model.m')
    input = []
    label = []
    for test_input_i in test_input_data:
        temp = test_input_i.strip().split('\t')
        if len(temp[0].split(' ')) > 1:
            input.append(temp[0])
            label.append(temp[1]+ '\n')
        # print label
        # for i in range(len(test_input_i)):
        #     temp = test_input_i[i].strip().split('\t')
        #     input.append(temp[0])
        #     # label.append(temp[1] + '\n')
        #     label.append(temp[1] )
    print len(input)
    x_test = tfidf.transform(input)
    test(x_test, label)