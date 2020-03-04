# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline


def getVSM(content, label):
    # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # 卡方检验进行特征选择，选取9000个特征
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    text_clf = Pipeline(
        [('vect', CountVectorizer()), ('x_chi2', SelectKBest(chi2, k=2000)), ('tfidf', TfidfTransformer())])
    text_clf.fit(content, label)
    joblib.dump(text_clf, 'model/tfidf_model.m')
