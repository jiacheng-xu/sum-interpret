from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import logging

import random


def train_dt(feat_names, feat_array, model_prediction, modified_text):
    logging.info(f"size: {len(feat_array)}")
    combined_data_in_list = [f + [l] for f, l in zip(feat_array, model_prediction)]
    random.shuffle(combined_data_in_list)
    combined_data_in_np = np.asarray(combined_data_in_list)
    num_train_rows = int(len(combined_data_in_list) * 0.8)
    feat, label = combined_data_in_np[:, :-1], combined_data_in_np[:, -1:]
    logging.info(f"{feat[:5,:]}")

    logging.info(f"{sum(label[:500])/500}")
    feat_train, feat_test = feat[:num_train_rows, :], feat[num_train_rows:, :]
    lb_train, lb_test = label[:num_train_rows], label[num_train_rows:]
    df_feat_train = pd.DataFrame(np.array(feat_train),
                                 columns=feat_names)
    df_feat_test = pd.DataFrame(np.array(feat_test),
                                columns=feat_names)

    clf = DecisionTreeClassifier(random_state=1234, max_depth=3, max_leaf_nodes=6, min_impurity_decrease=0.01)
    model = clf.fit(df_feat_train, lb_train)
    text_representation = tree.export_text(clf, feature_names=df_feat_train.columns.to_list())
    logging.info("\n" + text_representation)

    fig = plt.figure(figsize=(100, 100))
    _ = tree.plot_tree(clf,
                       feature_names=df_feat_train.columns.to_list(),
                       filled=True)
    fig.savefig("decistion_tree.png")

    train_acc = clf.score(df_feat_train, lb_train)
    logging.info(f"---------TRAIN ACC: {train_acc}---------")

    test_acc = clf.score(df_feat_test, lb_test)
    logging.info(f"---------TEST ACC: {test_acc}---------")
