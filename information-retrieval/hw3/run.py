#!/usr/bin/env python

import sys

from dupli import all_in_one

if __name__ == '__main__':
    files = sys.argv[1:]
    all_in_one(files)
