import numpy as np
import pandas as pd
import math
import random


def readRatingsDatas():
    path = 'data/ml-latest-small/ratings.csv'
    ratingsDF = pd.read_csv(path, usecols=[0, 1])
    data = dict()
    for v in ratingsDF.values:
        if v[0] not in data:
            data[v[0]] = set()
        data[v[0]].add(v[1])
    return data


def readMoviesDatas():
    path = 'data/ml-latest-small/movies.csv'
    moviesDF = pd.read_csv(path)


def getTrainsetAndTestset(data):
    train, test = dict(), dict()
    for u in data.keys():
        test[u] = set(random.sample(data[u], math.ceil(0.2 * len(data[u]))))
        train[u] = data[u] - test[u]
    return train, test


def cosSim(set1, set2):
    return len(set1 & set2) / math.sqrt(len(set1) * len(set2) * 1.0)


# def userSimilarity(train):
#     W = dict()
#     for u1 in train:
#         u1Dict = dict()
#         for u2 in train:
#             if u1 == u2:
#                 continue
#             u1Dict[u2] = cosSim(train[u1], train[u2])
#         W[u1] = u1Dict
#     return W

# 使用倒查表的方式，可以降低时间消耗
def userSimilarity(train):
    itemUser = dict()
    for u in train:
        for item in train[u]:
            if item not in itemUser:
                itemUser[item] = set()
            itemUser[item].add(u)
    C = dict()
    N = dict()
    for item in itemUser:
        for u1 in itemUser[item]:
            C[u1] = dict()
            if u1 not in N:
                N[u1] = 0
            N[u1] += 1
            for u2 in itemUser[item]:
                if u1 == u2:
                    continue
                if u2 not in C[u1]:
                    C[u1][u2] = 0
                C[u1][u2] += 1 / math.log(1 + len(itemUser[item]))
    W = dict()
    for u1 in C:
        W[u1] = dict()
        for u2 in C[u1]:
            W[u1][u2] = C[u1][u2] / math.sqrt(N[u1] * N[u2])
    return W



# 根据相似度最高的K个用户计算用户-物品的兴趣
def userItem(train, W, K):
    rank = dict()
    for u1 in train:
        rank[u1] = dict()
        for u2, u2Sim in sorted(W[u1].items(), key=lambda x: x[1], reverse=True)[:K]:
            for item in train[u2]:
                if item in train[u1]:
                    continue
                if item not in rank[u1]:
                    rank[u1][item] = 0.0
                rank[u1][item] += u2Sim
    return rank


# 根据相似度最高的K个用户推荐N个商品
def recommendByRank(rank, N):
    recommend = dict()
    for u in rank:
        recommend[u] = dict(sorted(rank[u].items(), key=lambda x: x[1], reverse=True)[:N])
    return recommend

def recall(pre, test):
    hit = 0.0
    n_re = 0.0
    for u in test:
        hit += len(set(pre[u].keys()) & test[u])
        n_re += len(test[u])
    return hit / n_re

def precision(pre, test):
    hit = 0.0
    n_pr = 0.0
    for u in test:
        hit += len(set(pre[u].keys()) & test[u])
        n_pr += len(pre[u])
    return hit / n_pr

def precisionWithN(pre, test, N):
    hit = 0.0
    n_pr = 0.0
    for u in test:
        hit += len(set(pre[u].keys()) & test[u])
        # 可能推荐电影的个数不足N
        n_pr += min(N, len(pre[u]))
    return hit / n_pr

def coverage(pre, train):
    hit = 0.0
    n_item = 0.0
    for u in train:
        hit += len(pre[u])
        n_item += len(train[u])
    return hit / n_item

def popularity(pre, train):
    item_popularity = dict()
    for u in train:
        for item in train[u]:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1

    ret = 0.0
    n = 0.0
    for u in pre:
        for item in pre[u]:
            ret += math.log(1 + item_popularity[item])
            n += 1
            # print(ret, n)
    return ret / n


def evaluate(train, test, K, N):
    result = dict()
    for k in K:
        rank = userItem(train, W, k)
        result[k] = dict()
        for n in N:
            recommend =recommendByRank(rank, k)
            re = recall(recommend, test)
            pr = precisionWithN(recommend, test, n)
            cov = coverage(recommend, train)
            pop = popularity(recommend, train)
            result[k][n] = {'precision':pr, 'recall':re, 'coverage': cov, 'popularity':pop}
    return result

def evaluateDF(result, K):
    for k in K:
        my_df = pd.DataFrame.from_dict(result[k], orient='index')
        print("K is", k)
        print(my_df)
        print()


if __name__ == '__main__':
    train = {
        'A': {'a', 'b', 'd'},
        'B': {'a', 'c'},
        'C': {'b', 'e'},
        'D': {'c', 'd', 'e'}
    }

    ratingsData = readRatingsDatas()
    train, test = getTrainsetAndTestset(ratingsData)

    W = userSimilarity(train)
    K = [10, 20, 40, 80]
    N = [5, 10, 20, 40, 80]
    result = evaluate(train, test, K, N)
    evaluateDF(result, K)

