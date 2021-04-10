#!/usr/bin/python3
import sys
import argparse
import os

from os.path import basename, splitext, dirname

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    
    csv = args.csv    

    data = pd.read_csv(csv, index_col=False)
    data.columns = data.columns.str.replace(' ', '')
    print (data)
    print ("--------------")
    data = data[data.bsize.eq(256)]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.plot(data["faults"] / data["batches"] - data["4k_dups"] / data["batches"], data["perf"], "b*", label="perf")
    #ax2.plot(data["size"], data["batches"], "r.-", label="batches")
    #plt.plot(data["size"], data["avg_vablock"], label="avg_vablock")

    figname = splitext(basename(csv))[0] + "-stats.png"
    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)


if __name__== "__main__":
  main()
