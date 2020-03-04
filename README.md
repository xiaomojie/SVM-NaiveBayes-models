基于支持向量机(SVM)/朴素贝叶斯(NaiveBayes)的表格列标签标注

1. vsm.py: 将文本中的词语转换为词频矩阵，tfidf，卡方检验进行特征选择
2. tableUnderstanding.py: 模型训练，使用交叉验证和网格搜索来获取最佳参数
3. modelTest.py: 模型测试，验证SVM和NaiveBayes模型的有效性
4. tabletest.py: 表格测试，以表格列为输入进行测试
