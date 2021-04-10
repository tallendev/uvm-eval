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

import numpy as np

import itertools

import Batch
from Batch import Experiment

marker="."
markersize=1

apps = ["cublas", "linear", "random", "stream"]
sizes = {"cublas" : [16384], "linear" : [4194304], "random" : [4194304], "stream": [500002816]}
bsizes = [256]

def do_all_more_3(e, csv, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5):

    figsize = plt.rcParams.get('figure.figsize')
    print("figsize:", figsize)
    figsize[1] = figsize[1] * 1.5

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams.update({"text.usetex": True, 
                        "font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 16})


    fig, axs = plt.subplots(3,1)
    plt.xlabel("Time")
    axs[0].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Total Batch Faults")

    axs[1].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Batch Faults No Duplicates")

    axs[2].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Batch Faults Migration-Only")

    figend = "batchsizes6"
    psize = None
    psize = basename(dirname(csv)).replace("_", "-")
    #if "pf" in csv:
    #    psize = dirname(csv).split("_")[2]
    #else:
    #    psize = dirname(csv).split("_")[1]


    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + "-rev.png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    #fig.savefig(figname, dpi=500, figsize=figsize)
    plt.close(fig)


def do_all_more(e, csv, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5):

    figsize = plt.rcParams.get('figure.figsize')
    print("figsize:", figsize)
    figsize[1] = figsize[1] * 1.5

    plt.rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(3,2)
    plt.xlabel("Time")
    axs[0, 0].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Total Batch Size")

    axs[0, 1].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Batch Size No Dups")

    axs[1, 0].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Batch Size w/o Already Resident")

    axs[1, 1].plot(x3, y3, "b"+marker, markersize=markersize) 
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("# VABlocks w/ 1+ Transfer in Batch")

    axs[2, 0].plot(x4, y4, "b"+marker, markersize=markersize) 
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("# Transfers in Batch")

    axs[2, 1].plot(x5, y5, "b"+marker, markersize=markersize) 
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].set_ylabel("Batch Transfer Size")
    

    figend = "batchsizes6"
    psize = None
    psize = basename(dirname(csv)).replace("_", "-")
    #if "pf" in csv:
    #    psize = dirname(csv).split("_")[2]
    #else:
    #    psize = dirname(csv).split("_")[1]


    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    #fig.savefig(figname, dpi=500, figsize=figsize)
    plt.close(fig)


def do_all(e, csv, x0, x1, x2, x3, y0, y1, y2, y3):
    fig, axs = plt.subplots(2,2)
    plt.xlabel("Time")
    axs[0, 0].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Batch Size")

    axs[0, 1].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Batch Size No Dups")

    axs[1, 0].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Batch Size w/o Already Resident")

    axs[1, 1].plot(x3, y3, "b"+marker, markersize=markersize) 
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("# VABlocks w/ 1+ Transfer in Batch")
    

    figend = "batchsizessquare"
    #psize = None
    #if "pf" in csv:
    #    psize = dirname(csv).split("_")[2]
    #else:
    #    psize = dirname(csv).split("_")[1]
    psize = basename(dirname(csv)).replace("_", "-")


    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

def do_batch_len_time_duplicates(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_duplicates()

    plot(x_ts, y_sizes, "Time", "Duplicates in Batch", "batchsizes-dups", csv)
    return (x_ts, y_sizes)


def do_batch_len_time_transfer_size(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_transfer_size()

    plot(x_ts, y_sizes, "Time", "Batch Transfer Size", "batchsizes-tsize", csv)
    return (x_ts, y_sizes)

def do_batch_len_time_transfer(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_transfers()

    plot(x_ts, y_sizes, "Time", "# Transfers in Batch", "batchsizes-transfers", csv)
    return (x_ts, y_sizes)


def do_batch_len_time_trimmed(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_trimmed_faults()

    plot(x_ts, y_sizes, "Time", "Batch Size w/o Already Resident", "batchsizes-trimmed", csv)
    return (x_ts, y_sizes)

def do_batch_len_time_nodup(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_no_duplicates()

    plot(x_ts, y_sizes, "Time", "VAblocks w/ Transfer in Batch", "batchsizes-vablock", csv)
    return (x_ts, y_sizes)

def do_batch_len_time_vablock(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_num_vablocks()

    plot(x_ts, y_sizes, "Time", "VAblocks w/ Transfer in Batch", "batchsizes-vablock", csv)
    return (x_ts, y_sizes)
    
def do_batch_len_time(e, csv):
    x_ts = e.get_batch_relative_start_times() 
    y_sizes = e.get_total_faults()
    plot(x_ts, y_sizes, "Time", "Batch Size (All Faults)", "batchsizes", csv)
    return (x_ts, y_sizes)

def plot(x, y, xlab, ylab, figend, csv):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    #if "pf" in csv:
    #    psize = dirname(csv).split("_")[2]
    #else:
    #    psize = dirname(csv).split("_")[1]
    psize = basename(dirname(csv)).replace("_", "-")
    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"

    ax.plot(x, y, "b" + marker, markersize=markersize)

    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

def title_lookup(s):
    if "cublas" in s:
        return "SGEMM"
    elif "stream" in s:
        return "Stream"

def duo_2(fig, axs, e, csv, x0, x1,  y0, y1, col):

    axs[0, col].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0,col].set_title(title_lookup(splitext(basename(csv))[0].split("_")[0]))
    if col == 0:
        axs[0, col].set_ylabel("Total Batch Faults")

    axs[1, col].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[1, col].set_xlabel("Start Time (ns)")
    #axs[1, col].set_ylabel("Batch Faults No Duplicates")
    if col == 0:
        axs[1, col].set_ylabel("No Duplicates")


def duo_3(fig, axs, e, csv, x0, x1, x2, y0, y1, y2, col):

    axs[0, col].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0, col].set_xlabel("Time")
    axs[0, col].set_ylabel("Total Batch Faults")
    
    axs[0,col].set_title(title_lookup(splitext(basename(csv))[0].split("_")[0]))

    axs[1, col].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[1, col].set_xlabel("Time")
    #axs[1, col].set_ylabel("Batch Faults No Duplicates")
    axs[1, col].set_ylabel("No Duplicates")

    axs[2, col].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[2, col].set_xlabel("Time")
    axs[2, col].set_ylabel("Migration-Only")
    #axs[2, col].set_ylabel("Batch Faults Migration-Only")


def duo2(c1, c2):
    e = Experiment(c1)
    e2 = Experiment(c2)

    matplotlib.rcParams['agg.path.chunksize'] = 10000

    x0, y0 = do_batch_len_time(e, c1)
    x1, y1 = do_batch_len_time_nodup(e, c1)

    x02, y02 = do_batch_len_time(e2, c2)
    x12, y12 = do_batch_len_time_nodup(e2, c2)


    #figsize = plt.rcParams.get('figure.figsize')
    #print("figsize:", figsize)
    #figsize[1] = figsize[1] * 1.5
    #plt.rcParams['figure.figsize'] = figsize

    plt.rcParams.update({"text.usetex": True, 
                        "font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 14})


    fig, axs = plt.subplots(2,2)
    plt.xlabel("Time")


    duo_2(fig, axs, e, c1, x0, x1, y0, y1, 0)
    duo_2(fig, axs, e2, c2, x02, x12, y02, y12, 1)

    figend = "duo2"

    figname = splitext(basename(c1))[0].split("_")[0] + "-" + splitext(basename(c2))[0].split("_")[0]  + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    #fig.savefig(figname, dpi=500, figsize=figsize)
    plt.close(fig)

def duo(c1, c2):
 
    e = Experiment(c1)
    e2 = Experiment(c2)

    matplotlib.rcParams['agg.path.chunksize'] = 10000

    x0, y0 = do_batch_len_time(e, c1)
    x1, y1 = do_batch_len_time_nodup(e, c1)
    x2, y2 = do_batch_len_time_trimmed(e, c1)

    x02, y02 = do_batch_len_time(e2, c2)
    x12, y12 = do_batch_len_time_nodup(e2, c2)
    x22, y22 = do_batch_len_time_trimmed(e2, c2)


    figsize = plt.rcParams.get('figure.figsize')
    print("figsize:", figsize)
    figsize[1] = figsize[1] * 1.5

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams.update({"text.usetex": True, 
                        "font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 14})


    fig, axs = plt.subplots(3,2)
    plt.xlabel("Time")


    duo_3(fig, axs, e, c1, x0, x1, x2, y0, y1, y2, 0)
    duo_3(fig, axs, e2, c2, x02, x12, x22, y02, y12, y22, 1)

    figend = "duo3"

    figname = splitext(basename(c1))[0].split("_")[0] + "-" + splitext(basename(c2))[0].split("_")[0]  + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    #fig.savefig(figname, dpi=500, figsize=figsize)
    plt.close(fig)

def per_app(c):
 
    e = Experiment(c)
    print(c)

    matplotlib.rcParams['agg.path.chunksize'] = 10000

    print("Cached Faults:", sum(e.get_total_faults()))    
    print("Transfers:", sum(e.get_transfers()))    
    x0, y0 = do_batch_len_time(e, c)
    x1, y1 = do_batch_len_time_nodup(e, c)
    x2, y2 = do_batch_len_time_trimmed(e, c)
    x3, y3 = do_batch_len_time_vablock(e, c)
    x4, y4 = do_batch_len_time_transfer(e, c)
    x5, y5 = do_batch_len_time_transfer_size(e, c)
    do_all(e, c, x0, x1, x2, x3, y0, y1, y2, y3)
    do_all_more(e, c, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5)
    do_all_more_3(e, c, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5)

def main():
    if len(sys.argv) == 2:
        parser = argparse.ArgumentParser()
        parser.add_argument('csv', type=str, help='Full path to CSV file')
        args = parser.parse_args()
        c = args.csv

        if not ".txt" in args.csv:
            print("Suspicious input file with no/wrong extension:", args.csv)
            exit(1)
        per_app(c)
    elif len(sys.argv) == 3:
        parser = argparse.ArgumentParser()
        parser.add_argument('csv1', type=str, help='Full path to CSV file')
        parser.add_argument('csv2', type=str, help='Full path to CSV file')
        args = parser.parse_args()
        c1 = args.csv1
        c2 = args.csv2

        duo2(c1, c2)
        duo(c1, c2)

    elif len(sys.argv) == 1:
        print ("Processing defaults")
        for app in apps:
            for size in sizes[app]:
                for bsize in bsizes:
                    c = f"../../benchmarks/{app}/log_{size}_bsize_{bsize}/{app}_0.txt"
                    print(c)
                    per_app(c)

if __name__== "__main__":
  main()

