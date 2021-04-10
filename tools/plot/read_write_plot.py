#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
from os.path import basename, splitext, dirname

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import Fault
from Fault import Experiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    c = args.csv
    m = "*"

    if not ".txt" in args.csv:
        print("Suspicious input file with no/wrong extension:", args.csv)
        exit(1)
    
    e = Experiment(c)
    
    ranges, faults, batches, num_batches = e.get_raw_data()
    e.print_info()

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()
    #fig = plt.gcf()
    ax = fig.add_subplot(1,1,1)
    
    read=[]
    rx = []
    write=[]
    wx = []
    for i,f in enumerate(faults):
        if f.access_type == "r":
            read.append(f.fault_address)
            rx.append(i)
        elif f.access_type == "w":
            write.append(f.fault_address)
            wx.append(i)
        else:
            print("unaccounted for access type:", faults.access_type)


    
    #print(ranges)
    ax.hlines(ranges, 0, len(faults))
    ax.plot(rx, read, 'b' + m, markersize=1, label="read")
    ax.plot(wx, write, 'r' + m, markersize=1, label="write")

    #fig.set_size_inches(16, 10)

    plt.legend()
    ax.set_xlabel('Fault Occurence')
    ax.set_ylabel('Fault Fault Index')
    psize = dirname(args.csv).split("_")[1]
    figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-faults.png"
    
    ylabels = map(lambda t: '0x%013X' % int(t), ax.get_yticks())
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

if __name__== "__main__":
  main()

