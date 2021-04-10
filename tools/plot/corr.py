#!/usr/bin/python3
import sys
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    parser.add_argument('-o', type=str, help='app name')
    args = parser.parse_args()

    data = pd.read_csv(args.csv, index_col=False)
    corr = data.corr()
    print (args.o)
    print (corr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns) + 1,1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)

    figname = args.o + "_corr.png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)


if __name__== "__main__":
  main()
