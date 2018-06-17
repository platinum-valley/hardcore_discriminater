# coding: utf-8

import numpy as np
from function import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

"""
K近傍法を用いて，音声ファイルを2クラスに分類する
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
random_accuracy_list = []

for train_idx, test_idx in k_fold.split(x_data, y_data):
    train_x = x_data[train_idx]
    test_x = x_data[test_idx]
    train_y = y_data[train_idx]
    test_y = y_data[test_idx]

    knn = KNeighborsClassifier(n_neighbors=10, p=1, metric="minkowski")
    knn.fit(train_x, train_y)

    # テストデータに対するクラス分類の精度を計算
    predict_test = knn.predict(test_x)
    accuracy_list.append(accuracy_score(test_y, predict_test))

    # ランダムにクラス分類する
    random = np.random.choice([-1, 1], predict_test.shape[0])
    random_accuracy_list.append(accuracy_score(random, predict_test))

print("Predict Accuracy:" + str(np.array(accuracy_list).mean(axis=0)))
print("Random Accuracy:" + str(np.array(random_accuracy_list).mean(axis=0)))


