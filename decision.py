# coding: utf-8

import numpy as np
from function import load_data
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

"""
決定木を用いて，音声ファイルを2クラスに分類する
10-foldクロスバリデーションより、回帰の精度をみる
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

    tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)
    tree.fit(train_x, train_y)

    export_graphviz(tree, out_file="./tree.dot")
    # テストデータに対するクラス分類の精度を計算
    predict_test = tree.predict(test_x)
    accuracy_list.append(accuracy_score(test_y, predict_test))

    # ランダムにクラス分類する
    random = np.random.choice([-1, 1], predict_test.shape[0])
    random_accuracy_list.append(accuracy_score(random, predict_test))

print("Predict Accuracy:" + str(np.array(accuracy_list).mean(axis=0)))
print("Random Accuracy:" + str(np.array(random_accuracy_list).mean(axis=0)))


