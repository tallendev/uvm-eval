
#!/usr/bin/python3

import sys
import argparse
import sys
import os
import argparse
import statistics
from colorsys import hsv_to_rgb
from os.path import basename, splitext, dirname
from collections import Counter, defaultdict

import itertools, matplotlib

import scipy.stats
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import itertools
from multiprocessing import Pool

import Fault
from Fault import Experiment

from enum import Enum

class DupTypes(Enum):
    INTRA = "intra"
    INTER = "inter"
    ALL = "all"

def sort_lists(list1, list2):
    sorted_lists = sorted(zip(list1, list2), key=lambda x: x[0])
    sorted_list1 = [x[0] for x in sorted_lists]
    sorted_list2 = [x[1] for x in sorted_lists]
    return sorted_list1, sorted_list2

def count_occurrences(numbers):
    unique_numbers = list(set(numbers))
    counts = [numbers.count(num) for num in unique_numbers]
    return unique_numbers, counts

def count_occurrences_all(numbers):
    # using a defaultdict structure, use every number in numbers as a key, and store the count of each number where
    # the count is provided by number.num_instances
    d = defaultdict(int)
    for number in numbers:
        d[number.fault_address] += number.num_instances
    keys = d.keys()
    return sort_lists(keys, [d[val] for val in keys])

def count_occurrences_intra(numbers):
    # using a defaultdict structure, use every number in numbers as a key, and store the count of each number where
    # the count is provided by number.num_instances
    d = defaultdict(int)
    for number in numbers:
        d[number.fault_address] += number.num_instances - 1
    keys = d.keys()
    return sort_lists(keys, [d[val] for val in keys])


def main():
    parser = argparse.ArgumentParser(description='Process multiple CSV files.')
    parser.add_argument('csv_files', metavar='CSV', type=str, nargs='+',
                        help='CSV files to process')
    parser.add_argument('-o', type=str, default="", help='output filename')

    args = parser.parse_args()
    for v in DupTypes:
        plt.clf()

        matplotlib.rcParams['agg.path.chunksize'] = 10000

        plt.rcParams["figure.figsize"] = (18.5, 10.5)
        for c in args.csv_files:
            e = Experiment(c)
            ranges, faults, batches, num_batches = e.get_raw_data()
            e.print_info()
            print("ranges:", ranges)

            ranges, faults, batches, num_batches = e.get_raw_data()
            e.print_info()

            psize = basename(dirname(c))  # .split("_")[-1]
            prefix = splitext(basename(c))[0].split('_')[0] + "-" + psize.split("_")[1]
            ranges.append(sys.maxsize)
            bfaults = []
            bdups = []
            #intra
            for b in batches:
                faults = [f.fault_address for f in b]
                if v == DupTypes.INTRA:
                    # intra does special processing that eliminates the need to -1 counts
                    unique, counts = count_occurrences_intra(b)
                elif v == DupTypes.INTER:
                    unique, counts = count_occurrences(faults)
                    counts = [c - 1 for c in counts]
                elif v == DupTypes.ALL:
                    unique, counts = count_occurrences_all(b)
                    counts = [c - 1 for c in counts]
                else:
                    print("Type doesn't match???")
                    sys.exit(1)
                bdups.append(sum(counts))

                # bfaults.append(len(b))
            bdups.sort()
            compute_cdf(bdups, prefix)

        plt.xlabel("# of Duplicates")
        plt.ylabel("Probability")

        prefix = "multi"
        plt.title(f"{prefix}")

        if v == DupTypes.INTRA:
            outdir = "dupfaultsintra"
        elif v == DupTypes.INTER:
            outdir = "dupfaultsinter"
        elif v == DupTypes.ALL:
            outdir = "dupfaultsall"
        else:
            print("Type doesn't match????????")
            sys.exit(1)

        # require the X axis labels to be listed vertically
        plt.xticks(rotation=90, fontsize='small')
        figname = None
        if args.o == "":
            full_outdir = f"{outdir}/{prefix}/"
            if not os.path.exists(full_outdir):
                os.makedirs(full_outdir)
            figname = full_outdir + f"multidupfaultscdf.png".replace("_", "-")
        else:
            figname = args.o
            if ".png" not in figname:
                figname += ".png"
        plt.legend()
        plt.tight_layout()
        print('saving figure:', figname)
        plt.savefig(figname)


def compute_cdf(bdups, label):
    bdups_data = Counter(bdups)
    bdups_keys = np.sort(list(bdups_data.keys()))
    bdups_count = np.array([bdups_data[key] for key in bdups_keys])
    cdf = np.cumsum(bdups_count) / len(bdups)
    print("bdups:", *bdups)
    print("cdf:", *cdf)
    print(f"plotting {label}")
    plt.plot(bdups_keys, cdf, label=label)


if __name__== "__main__":
    main()

