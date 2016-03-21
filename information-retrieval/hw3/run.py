#!/usr/bin/env python

import sys

from dupli import all_in_one

if __name__ == '__main__':
    # files = sys.argv[1:]
    # all_in_one(files)

    from os import listdir
    from os.path import isfile, join

    PATH_TO_DATASET = "./dataset/"

    tarballs = ["".join((PATH_TO_DATASET, f)) for f in listdir(PATH_TO_DATASET) if isfile(join(PATH_TO_DATASET, f))]

    all_in_one(tarballs[:2])
