#!/usr/bin/env python

# coding: utf-8

#------------------------------------------------------------------------------------------

from __future__ import print_function

#------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

#------------------------------------------------------------------------------------------

pure_train_data = pd.read_csv("./data/train.data.cvs")

#------------------------------------------------------------------------------------------

train_data = pure_train_data  #[~pure_train_data.duplicated(["QID", "Y"])]

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

Y_s = train_data.loc[:, ["Y"]]
QID_s = train_data.loc[:, ["QID"]]

#------------------------------------------------------------------------------------------

X_dt = train_data.drop(["Y", "QID"], axis=1)

#------------------------------------------------------------------------------------------

info_dt = pd.concat([QID_s, Y_s], axis=1)
info_dt = info_dt.sort_values(["QID", "Y"], ascending=[True, False])


def numerate(group):
    group.loc[:, "num"] = range(1, len(group) + 1)
    return group


info_dt = info_dt.groupby("QID").apply(numerate)

info_dt.loc[:, "Y_norm"] = info_dt.Y / info_dt.Y.max()

info_dt.loc[:, "DCG"] = info_dt.apply(
    lambda x: (2**x.Y_norm - 1) / np.log2(x.num + 1), axis=1)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

unique_QID = pd.Series(info_dt.QID.unique())
test_QIDs = unique_QID.sample(frac=0.5)
train_QIDs = unique_QID[~unique_QID.isin(test_QIDs)]

#------------------------------------------------------------------------------------------

docs_grouped_by_qid = info_dt.groupby("QID")
qid_dcg_s = info_dt.groupby(
    ["QID", "num"]).DCG.first().reset_index().groupby("QID").DCG.sum()

#------------------------------------------------------------------------------------------

train_doc_ids = info_dt[info_dt.QID.isin(train_QIDs)].index
test_doc_ids = info_dt[info_dt.QID.isin(test_QIDs)].index

#------------------------------------------------------------------------------------------

train_X = X_dt.loc[train_doc_ids]
test_X = X_dt.loc[test_doc_ids]
train_qid_dcg = qid_dcg_s.loc[train_QIDs]
test_qid_dcg = qid_dcg_s.loc[test_QIDs]
train_docs_info = info_dt.loc[train_doc_ids]
test_docs_info = info_dt.loc[test_doc_ids]

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

import sys

#------------------------------------------------------------------------------------------

f_rel = lambda rel, pos: (2**rel - 1) / np.log2(pos + 1)
f_sigm = lambda si, sj, sign: -0.5 / (1.0 + np.exp(0.5 * (si - sj) * sign)) * sign


def compute_grad(curr_obj, obj, q_dcg):
    bigger = obj["Y_norm"] < curr_obj["Y_norm"]
    lower = obj["Y_norm"] > curr_obj["Y_norm"]
    if not (lower or bigger):
        return 0

    new_dcg = (q_dcg - curr_obj["DCG"] - obj["DCG"] + f_rel(
        curr_obj["Y_norm"], obj["num"]) + f_rel(obj["Y_norm"], curr_obj["num"])
               )

    ndcg_delta = np.abs(1 - new_dcg / q_dcg)

    sign = 1 if bigger else -1
    grad = f_sigm(0, 0, sign)
    lmbda = grad * ndcg_delta
    return -lmbda


#------------------------------------------------------------------------------------------

test_qid_idcg = test_docs_info.groupby("QID")["DCG"].sum()

#------------------------------------------------------------------------------------------


def compute_dcg(group):
    group = group.reset_index()
    dcg = ((2**group.Y_norm - 1) / np.log2(group.index + 2)).sum()
    return dcg


#------------------------------------------------------------------------------------------


def compute_lambda(rel, pred_rel, sigma=0.8):
    rel_sign = np.sign(rel[:, np.newaxis] - rel)
    pred_rel_diff = pred_rel[:, np.newaxis] - pred_rel
    pred_rel_sign = np.sign(pred_rel_diff)

    left_part = 0.5 * (1.0 - rel_sign
                       )  # === 0 т.к. мы рассматриваем только Sij = 1
    right_part = 1.0 / (1.0 + np.exp(sigma * pred_rel_diff * rel_sign))

    lambda_m = sigma * (-right_part) * rel_sign

    return lambda_m.sum(axis=1)


#------------------------------------------------------------------------------------------

import multiprocessing

#------------------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------------------


def compute_docs_grads(qid_group):
    qid_doc_ids = qid_group.index.tolist()
    qid_rels = qid_group["Y_norm"].as_matrix()
    qid_rels_pred = qid_group["pred"].as_matrix()

    qid_grad = -compute_lambda(qid_rels, qid_rels_pred)

    return zip(qid_doc_ids, qid_grad)


#------------------------------------------------------------------------------------------


class LambdaRank:
    def __init__(self, n_estimators, shrikage_rate=0.01):
        self.n_estimators = n_estimators
        self.trees = []
        self.shrikage_rate = shrikage_rate

    def predict(self, X):
        y = np.zeros(len(X))
        for tree in self.trees:
            y += tree.predict(X) * self.shrikage_rate

        return y

    def fit(self, _train_X, _train_doc_ids, _train_QIDs, _train_qid_dcg,
            _train_docs_info, test_X, test_docs_info, test_qid_idcg):
        base_tree = DecisionTreeRegressor(max_depth=4)
        base_tree.fit(_train_X.as_matrix(), np.tile(0, (len(_train_X), 1)))

        self.trees.append(base_tree)

        for i in range(self.n_estimators):
            print("Train estimator #%i" % (i + 1))

            ensemble_predict = self.predict(_train_X.as_matrix())
            _train_docs_info.loc[_train_doc_ids, "pred"] = ensemble_predict

            docs_grouped_by_qid = _train_docs_info.groupby("QID")

            docs_grads = []

            #             pool = multiprocessing.Pool(processes=8)
            #             docs_grads = []
            #             data_gen = (docs_grouped_by_qid.get_group(q_id) for q_id in _train_QIDs)
            #             for j, part in enumerate(pool.imap_unordered(compute_docs_grads, data_gen)):
            #                 docs_grads += part
            #             pool.close()

            for j, q_id in enumerate(_train_QIDs):
                sys.stdout.write("\r %s/%s" % (j + 1, len(train_QIDs)))

                qid_group = docs_grouped_by_qid.get_group(q_id)

                qid_doc_ids = qid_group.index.tolist()
                qid_rels = qid_group["Y_norm"].as_matrix()
                qid_rels_pred = qid_group["pred"].as_matrix()

                qid_grad = -compute_lambda(qid_rels, qid_rels_pred)

                docs_grads += zip(qid_doc_ids, qid_grad)
            print()
            tree = DecisionTreeRegressor(max_depth=4, min_samples_split=4)
            step_train_doc_ids, step_train_grad = zip(*docs_grads)
            tree = tree.fit(_train_X.loc[list(step_train_doc_ids)].as_matrix(),
                            np.asarray(step_train_grad))
            self.trees.append(tree)

            if i % 10 == 0:
                print("  validate")
                test_docs_info.loc[test_doc_ids, "test_pred_y"] = lrk.predict(
                    test_X.as_matrix())

                test_qid_dcg = test_docs_info.sample(frac=1).sort_values(
                    ["QID", "test_pred_y"],
                    ascending=[True, False]).groupby("QID").apply(compute_dcg)
                print("ndcg:", (test_qid_dcg / test_qid_idcg).mean())


lrk = LambdaRank(n_estimators=250, shrikage_rate=0.02)
lrk.fit(train_X, train_doc_ids, train_QIDs, train_qid_dcg, train_docs_info,
        test_X, test_docs_info, test_qid_idcg)

#------------------------------------------------------------------------------------------

import dill

with open("./simple_lambda_rank.model", 'wr') as fout:
    dill.dump(lrk, fout)

#------------------------------------------------------------------------------------------

sys.exit(0)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

print("  validate")
test_docs_info.loc[test_doc_ids, "test_pred_y"] = lrk.predict(test_X.as_matrix(
))
# test_docs_info.loc[test_doc_ids, "test_pred_y"] = np.random.normal(loc=0, scale=1.0, size=len(test_X))

test_qid_dcg = test_docs_info.sample(frac=1).sort_values(
    ["QID", "test_pred_y"],
    ascending=[True, False]).groupby("QID").apply(compute_dcg)
print("ndcg:", (test_qid_dcg / test_qid_idcg).mean())

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

testset = pd.read_csv("./data/testset.cvs")

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

Test_QID = testset["QID"]
Test_X = testset.drop(["Y", "QID"], axis=1)

#------------------------------------------------------------------------------------------

Test_rel = lrk.predict(Test_X)

#------------------------------------------------------------------------------------------

sumbission = Test_QID.to_frame().copy()
sumbission.loc[:, "rel"] = Test_rel
# sumbission.loc[:, "rel"] = np.random.normal(loc=0, scale=1.0, size=len(Test_X))

#------------------------------------------------------------------------------------------

sumbission.index.name = "DocumentId"  # ,QueryId

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

sumbission = sumbission.reset_index().sample(frac=1).sort_values(
    ["QID", "rel"], ascending=[True, False])
sumbission = sumbission.rename(columns={"QID": "QueryId"})
sumbission.loc[:, "DocumentId"] += 1

#------------------------------------------------------------------------------------------

sumbission[["DocumentId", "QueryId"]].to_csv(
    "./data/sumbission7.csv", index=False)

#------------------------------------------------------------------------------------------

sumbission.groupby("QueryId").rel.count().min()
