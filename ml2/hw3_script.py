#!/usr/bin/env python
# -*- encoding: utf-8

import argparse

import pandas as pd
import commands
import random

from joblib import Parallel, delayed

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

SERVER = None

DATA = None

CHUNK_SIZE = 400

PATH_TO_HW3_FILES = "./hw3_files"


def oracle(x):
    query = "java -cp %s/OracleRegression.jar Oracle" % PATH_TO_HW3_FILES
    for xi in x:
        query += " " + str(xi)
    return commands.getoutput(query)


def get_samples(i, size):
    # samples = random.sample(DATA.index[i * size:i * size + size], CHUNK_SIZE)
    samples = random.sample(DATA.index, CHUNK_SIZE)

    with open('%s/private/%s_%d.csv' % (PATH_TO_HW3_FILES, SERVER, i), 'a') as f:
        for i in samples:
            x = DATA.loc[i].as_matrix()
            y = oracle(x)
            f.write("%d,%s\n" % (i, y))

    print samples


def parse_args():
    parser = argparse.ArgumentParser(description='Active Learning Oracle')
    parser.add_argument("-s", "--server", action="store", type=str, help="Server name")
    return parser.parse_args()


if __name__ == '__main__':
    global SERVER

    args = parse_args()
    SERVER = args.server
    print "Num of cores - ", NUM_CORES
    print SERVER

    print "Reading X_private.."
    global DATA
    df = pd.read_csv("%s/X_private.csv" % PATH_TO_HW3_FILES, index_col=0)

    n_samples = len(df.index)
    size = n_samples / 2 / NUM_CORES

    DATA = df
    # if SERVER == "right":
    #     DATA = df[n_samples / 2:]
    # else:
    #     DATA = df[:n_samples / 2]

    print "Fetching data..."
    Parallel(n_jobs=NUM_CORES)(delayed(get_samples)(i, size) for i in range(NUM_CORES))

    print "Finished!"
