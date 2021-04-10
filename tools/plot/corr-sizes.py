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
    parser.add_argument('csv', nargs="+", type=str, help='Full path to CSV file')
    args = parser.parse_args()
    
    for csv in args.csv:
        data = pd.read_csv(csv, index_col=False)
        print (data)
        corr = data.corr()
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

        figname = splitext(basename(csv))[0] + "-corr.png"
        plt.tight_layout()
        print('saving figure:', figname)
        fig.savefig(figname, dpi=500)
        plt.close(fig)


if __name__== "__main__":
  main()
