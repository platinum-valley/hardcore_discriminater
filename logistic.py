# coding: utf-8

import numpy as np
from function import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
"""
ロジスティック回帰を用いて，音声ファイルを2クラスに分類する
10-foldクロスバリデーションより、分類の精度をみる
ランダムに2クラス分類を行った場合と比較する

"""

csv_file_path = r"path/to/data.csv"
x_data, y_data = load_data(csv_file_path)

x_data = np.array(x_data)
y_data = np.array(y_data)

# データを10分割する
n_fold = 10
k_fold = KFold(n_fold, shuffle=True)

accuracy_list = []
train_accuracy_list = []
norm_accuracy_list = []
norm_train_accuracy_list = []
L1_accuracy_list = []
L1_train_accuracy_list = []
PCA_accuracy_list = []
PCA_train_accuracy_list = []
random_accuracy_list = []

for train_idx, test_idx in k_fold.split(x_data, y_data):
    train_x = x_data[train_idx]
    test_x = x_data[test_idx]
    train_y = y_data[train_idx]
    test_y = y_data[test_idx]

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr_norm = LogisticRegression(C=1000.0, random_state=0)
    lr_L1 = LogisticRegression(C=1000.0, penalty="l1")
    lr_pca = LogisticRegression(C=1000.0, random_state=0)

    mms = MinMaxScaler()
    train_x_norm = mms.fit_transform(train_x)
    test_x_norm = mms.fit_transform(test_x)

    pca = PCA(n_components=10)
    train_x_pca = pca.fit_transform(train_x)
    test_x_pca = pca.fit_transform(test_x)
    #print("not PCA", train_x.shape)
    #print("PCA", train_x_pca.shape)

    lr.fit(train_x, train_y)
    lr_norm.fit(train_x_norm, train_y)
    lr_L1.fit(train_x, train_y)
    lr_pca.fit(train_x_pca, train_y)

    # テストデータに対するクラス分類の精度を計算
    predict_test = lr.predict(test_x)
    predict_train = lr.predict(train_x)
    predict_test_norm = lr_norm.predict(test_x_norm)
    predict_train_norm = lr_norm.predict(train_x_norm)
    predict_test_L1 = lr_L1.predict(test_x)
    predict_train_L1 = lr_L1.predict(train_x)
    predict_test_pca = lr_pca.predict(test_x_pca)
    predict_train_pca = lr_pca.predict(train_x_pca)

    accuracy_list.append(accuracy_score(test_y, predict_test))
    train_accuracy_list.append(accuracy_score(train_y, predict_train))
    norm_accuracy_list.append(accuracy_score(test_y, predict_test_norm))
    norm_train_accuracy_list.append(accuracy_score(train_y, predict_train_norm))
    L1_accuracy_list.append(accuracy_score(test_y, predict_test_L1))
    L1_train_accuracy_list.append(accuracy_score(train_y, predict_train_L1))
    PCA_accuracy_list.append(accuracy_score(test_y, predict_test_pca))
    PCA_train_accuracy_list.append(accuracy_score(train_y, predict_train_pca))

    # ランダムにクラス分類する
    random = np.random.choice([-1, 1], predict_test.shape[0])
    random_accuracy_list.append(accuracy_score(random, predict_test))


print("Predict Accuracy:" + str(np.array(accuracy_list).mean(axis=0)))
print("Train Accuracy:" + str(np.array(train_accuracy_list).mean(axis=0)))
print("Predict Accuracy(norm):" + str(np.array(norm_accuracy_list).mean(axis=0)))
print("Train Accuracy(norm):" + str(np.array(norm_train_accuracy_list).mean(axis=0)))
print("Predict Accuracy(L1):" + str(np.array(L1_accuracy_list).mean(axis=0)))
print("Train Accuracy(L1):" + str(np.array(L1_train_accuracy_list).mean(axis=0)))
print("Random Accuracy:" + str(np.array(random_accuracy_list).mean(axis=0)))
print("Predict Accuracy(PCA):" + str(np. array(PCA_accuracy_list).mean(axis=0)))
print("Train Accuracy(PCA):" + str(np.array(PCA_train_accuracy_list).mean(axis=0)))



