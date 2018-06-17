# coding: utf-8

import numpy as np
from function import load_data
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

"""
SVMを用いて，音声ファイルを2クラスに分類する
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

    tuned_parameters = {"C": np.random.rand(10), "kernel": ["rbf"], "gamma": np.random.rand(100)}
    svc_grid = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring="%s" % "accuracy")
    svc_grid.fit(train_x, train_y)


    #print(svc_grid.cv_results_)
    #print(svc_grid.best_estimator_)
    #svc_best = svc_grid.best_estimator_
    svm = SVC(kernel="rbf", gamma=0.06, C=1.0, random_state=0, class_weight="balanced")
    #svm = SVC(C=1.0, random_state=0)

    svm.fit(train_x, train_y)

    # テストデータに対するクラス分類の精度を計算
    predict_test = svm.predict(test_x)
    predict_test = svc_grid.predict(test_x)
    accuracy_list.append(accuracy_score(test_y, predict_test))

    # ランダムにクラス分類する
    random = np.random.choice([-1, 1], predict_test.shape[0])
    random_accuracy_list.append(accuracy_score(random, predict_test))

print("Predict Accuracy:" + str(np.array(accuracy_list).mean(axis=0)))
print("Random Accuracy:" + str(np.array(random_accuracy_list).mean(axis=0)))


