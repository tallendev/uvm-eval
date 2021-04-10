#!/usr/bin/python3
import sys
import argparse
import os

from os.path import basename, splitext, dirname
import re

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

import numpy as np

import itertools

from natsort import humansorted

from Fault import Experiment

from multiprocessing import Pool

#NUM_THREADS=12


appname = None

class ExperimentStats:
    # TODO should size, bsize be standardized into Fault.py?
    def __init__(self, e, size, bsize, app):
        self.app = app
        self.avg_batches = np.mean([len(b) for b in e.batches])
        self.num_batches = len(e.batches)
        self.num_faults = len(e.faults)
        self.num_4k_dups = sum(e.get_duplicate_faults_4k())
        self.num_64k_dups = sum(e.get_duplicate_faults_64k())
        self.avg_vablock = e.count_avg_vablocks_per_batch()
        #print(self.avg_vablock)
        self.size = size.strip().replace(" ", "")
        self.bsize = bsize
        self.pf = e.pf


class ExperimentPerf:
    def __init__(self, perf, alloced, size, bsize, pf, app):
        self.app = app
        self.perf = perf
        self.alloced = int(alloced)
        self.size = size.strip().replace(" ", "")
        self.bsize = bsize
        self.pf = pf

def parse_bsize(t):
    bsize, lis = t
    es = []
    for csv in humansorted(lis):
        e = Experiment(csv)
        size = None
        if "pf" in csv:
            size = basename(dirname(csv)).split("_")[2]
        else:
            size = basename(dirname(csv)).split("_")[1]
        es.append(ExperimentStats(e, size, bsize))
    return (es, bsize)

def parse_bsize_sub(csv):
    e = Experiment(csv)
    size = None
    if "pf" in csv:
        size = basename(dirname(csv)).split("_")[2]
    else:
        size = basename(dirname(csv)).split("_")[1]
    return (e, size)

def parse_bsize_sub2(t):
    bsize, csv = t
    e = Experiment(csv)
    size = None
    if "pf" in csv:
        size = basename(dirname(csv)).split("_")[2]
    else:
        size = basename(dirname(csv)).split("_")[1]
    app = splitext(basename(csv))[0].split("_")[0]
    return ExperimentStats(e, size, bsize, app)


def parse_bsize_para(t):
    bsize, lis = t
    es = []
    with Pool() as pool:
        es = pool.map(parse_bsize_sub, humansorted(lis, reverse=True))
        es = [ExperimentStats(e, s, bsize) for e, s in es]
    return (es, bsize)


def main():
    global appname
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', nargs="+", type=str, help='Full path to CSV file')
    parser.add_argument('-f', nargs="+")
    parser.add_argument('--app', type=str, help="app name for output")
    args = parser.parse_args()
    m = "*"
    
    

    expstats = []
    expstats_pf = []
    bsize_dicts = []
    
    perfs = []
    for csv in humansorted(args.p):
        bsize = int(splitext(basename(csv))[0].split("-")[-1])
        appname = basename(dirname(csv))
        pf = "pf" in csv
        with open(csv, "r") as c:
            while True:
                size = c.readline().split(",")[1]
                alloced = c.readline().split(",")[1]
                perf = float(c.readline().split(",")[1])
                perfs.append(ExperimentPerf(perf, alloced, size, bsize, pf, appname))
                if c.tell() == os.fstat(c.fileno()).st_size:
                    break


    bsize_dict = dict()
    for csv in humansorted(args.f):
        bsize = int(basename(dirname(csv)).split("_")[-1])
        if bsize not in bsize_dict:
            bsize_dict[bsize] = []

        bsize_dict[bsize].append(csv)

    if "tealeaf" in args.app:
        with Pool(processes=2) as pool:
            #x, y, bsize = pool.map(parse_bsize, bsize_dict.items())
            ret = pool.map(parse_bsize, bsize_dict.items())
            for e, bsize in ret:
                if e.pf:
                    expstats_pf += e
                else:
                    expstats += e
    else:
        all_pairs = [(bsize, i) for bsize, lis in bsize_dict.items() for i in lis]
        with Pool() as pool:
            #x, y, bsize = pool.map(parse_bsize, bsize_dict.items())
        #ret = pool.map(parse_bsize, bsize_dict.items())
        #ret = map(parse_bsize_para, bsize_dict.items())
            #ret = pool.map(parse_bsize_para, bsize_dict.items())
            ret = pool.map(parse_bsize_sub2, sorted(all_pairs, key=lambda f: os.stat(f[1]).st_size, reverse=True))
        for e in ret:
            if e.pf:
                expstats_pf.append(e)
            else:
                expstats.append(e)

    # normalize the app performance
    # FIXME we actually are not gonna use this script for multiple apps but this still works so might as well keep it i guess
    apps = {perf.app for perf in perfs}
    print ("apps:", apps)
    for app in apps:
        perf_set = [perf for perf in perfs if perf.app == app]
        #perf_set = [perf for perf in perfs if perf.app == app and perf.pf]
        m = max(perf_set, key=lambda p: p.perf).perf
        for perf in perf_set:
            perf.perf = perf.perf / m

#    for app in apps:
#        perf_set = [perf for perf in perfs if perf.app == app and not perf.pf]
#        m = max(perf_set, key=lambda p: p.perf).perf
#        for perf in perf_set:
#            perf.perf = perf.perf / m

    sizes = {perf.size for perf in perfs}

    for size in sizes:
        csv_name = "csvs-size/" + args.app + f"-{size}.csv"
        print ("saving to", csv_name)
        with open(csv_name, "w+") as csv:
            #csv.write("bsize, pf, size, faults, batches, avg_batch, 4k_dups, 64k_dups, perf\n")
            csv.write("perf, pf, bsize, size, faults, batches, avg_batch, 4k_dups, 64k_dups, alloced, avg_vablock\n")
            for exp in expstats:
                ep = None
                for e in perfs:
                    #print (f"{e.bsize} == {exp.bsize}, {e.pf} == {exp.pf}, {e.size} == {exp.size}")
                    if exp.bsize == e.bsize and exp.pf == e.pf and exp.size == e.size and size == exp.size:
                        ep = e
                        break
                if ep is None:
                    continue
                    print (f"no perf match: {exp.bsize}, {exp.pf}, {exp.size}")
                    print (f"no perf match: {type(exp.bsize)}, {type(exp.pf)}, {type(exp.size)}")
                    print (f"no perf match: {size}")
                csv.write(f"{ep.perf}, {1 if e.pf else 0}, {exp.bsize}, {exp.size}, {exp.num_faults}, {exp.num_batches}, {exp.avg_batches}, {exp.num_4k_dups}, {exp.num_64k_dups}, {ep.alloced}, {exp.avg_vablock}\n")

        csv_name = "csvs-size/" + args.app + f"-pf-{size}.csv"
        print ("saving to", csv_name)
        with open(csv_name, "w+") as csv:
            #csv.write("bsize, pf, size, faults, batches, avg_batch, 4k_dups, 64k_dups, perf\n")
            csv.write("perf, pf, bsize, size, faults, batches, avg_batch, 4k_dups, 64k_dups, alloced, avg_vablock\n")
            for exp in expstats_pf:
                ep = None
                for e in perfs:
                    #print (f"{e.bsize} == {exp.bsize}, {e.pf} == {exp.pf}, {e.size} == {exp.size}")
                    if exp.bsize == e.bsize and exp.pf == e.pf and exp.size == e.size and size == e.size:
                        ep = e
                        break
                if ep is None:
                    continue
                    print (f"no perf match: {exp.bsize}, {exp.pf}, {exp.size}")
                    print (f"no perf match: {type(exp.bsize)}, {type(exp.pf)}, {type(exp.size)}")
                csv.write(f"{ep.perf}, {1 if e.pf else 0}, {exp.bsize}, {exp.size}, {exp.num_faults}, {exp.num_batches}, {exp.avg_batches}, {exp.num_4k_dups}, {exp.num_64k_dups}, {ep.alloced}, {exp.avg_vablock}\n")

        csv_name = "csvs-size/" + args.app + f"-all-{size}.csv"
        print ("saving to", csv_name)
        with open(csv_name, "w+") as csv:
            #csv.write("bsize, pf, size, faults, batches, avg_batch, 4k_dups, 64k_dups, perf\n")
            csv.write("perf, pf, bsize, size, faults, batches, avg_batch, 4k_dups, 64k_dups, alloced, avg_vablock\n")
            for exp in expstats_pf + expstats:
                ep = None
                for e in perfs:
                    #print (f"{e.bsize} == {exp.bsize}, {e.pf} == {exp.pf}, {e.size} == {exp.size}")
                    if exp.bsize == e.bsize and exp.pf == e.pf and exp.size == e.size and size == e.size:
                        ep = e
                        break
                if ep is None:
                    continue
                    print (f"no perf match: {exp.bsize}, {exp.pf}, {exp.size}")
                    print (f"no perf match: {type(exp.bsize)}, {type(exp.pf)}, {type(exp.size)}")
                csv.write(f"{ep.perf}, {1 if e.pf else 0}, {exp.bsize}, {exp.size}, {exp.num_faults}, {exp.num_batches}, {exp.avg_batches}, {exp.num_4k_dups}, {exp.num_64k_dups}, {ep.alloced}, {exp.avg_vablock}\n")


    print("done")


if __name__== "__main__":
  main()

