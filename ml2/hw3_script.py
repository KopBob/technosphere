#!/usr/bin/env python
# -*- encoding: utf-8

import pandas as pd
import commands
import random

from joblib import Parallel, delayed

from src.constants import NUM_CORES

SERVER = "local"

DATA = None

CHUNK_SIZE = 300


def oracle(x):
    query = "java -cp ./hw3_files/OracleRegression.jar Oracle"
    for xi in x:
        query += " " + str(xi)
    return commands.getoutput(query)


def get_samples(i, size):
    samples = random.sample(DATA.index[i * size:i * size + size], CHUNK_SIZE)

    with open('./hw3_files/train/%s_%d.csv' % (SERVER, i), 'a') as f:
        for i in samples:
            x = DATA.ix[i].as_matrix()
            y = oracle(x)
            f.write("%d,%s\n" % (i, y))

    print samples


if __name__ == '__main__':
    print "Num of cores - ", NUM_CORES

    print "Reading X_public.."
    global DATA
    df = pd.read_csv("./hw3_files/X_public.csv", index_col=0)

    n_samples = len(df.index)
    size = n_samples / 2 / NUM_CORES

    if SERVER == "local":
        DATA = df[n_samples / 2:]
    else:
        DATA = df[:n_samples / 2]

    print "Fetching data..."
    Parallel(n_jobs=NUM_CORES)(delayed(get_samples)(i, size) for i in range(NUM_CORES))

    print "Finished!"
