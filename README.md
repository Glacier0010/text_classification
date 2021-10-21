# text_classification
基于PyTorch的文本分类模型
数据集：AG-news (from Kaggle)
网络结构：Embedding + BiLSTM + Dropout + ReLU + FC
词典：glove.6B.100d
优化：L2 regularization (weight decay)
结果：train_accuracy=0.99135,    test_accuracy=0.90053
