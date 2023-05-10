#!/usr/bin/python3
import multiprocessing
import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, splitext, dirname
from collections import defaultdict

import matplotlib
from scipy import stats

matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

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
    #for c in args.csv_files:
    with ThreadPoolExecutor() as executor:
        executor.map(process_csv, args.csv_files)


    #with multiprocessing.Pool() as pool:
    #    pool.map(process_csv, args.csv_files)

def process_csv(c):
    e = Experiment(c)
    ranges, faults, batches, num_batches = e.get_raw_data()
    e.print_info()
    ranges.append(sys.maxsize)
    for v in DupTypes:
        if v == DupTypes.INTRA:
            outdir = "fitdupfaultsintra"
            label = f"Intra-SM Duplicates"
        elif v == DupTypes.INTER:
            outdir = "fitdupfaultsinter"
            label = f"Inter-SM Duplicates"
        elif v == DupTypes.ALL:
            outdir = "fitdupfaultsall"
            label = f"All Duplicates"
        else:
            print("Type doesn't match????????")
            sys.exit(1)


        bdups = []
        faultlist = []
        for b in batches:
            faults = [f.fault_address for f in b]
            faultlist += faults
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
        fmin = min(faultlist)
        plot_hist(bdups, c, label, outdir)

        for i in range(len(ranges) - 1):
            rbatches = []
            rbfaults = []
            rbdups = []
            for b in batches:
                for j in range(len(b)):
                    batch = []
                    for fault in b:
                        if ranges[i] <= fault.fault_address < ranges[i + 1]:
                            batch.append(fault)
                    if batch:
                        rbatches.append(b)
            if (rbatches):
                for b in rbatches:
                    _, counts = count_occurrences_all(b)
                    rbfaults.append(sum(counts))
                    unique, counts = count_occurrences_intra(b)
                    counts = [c for c in counts]
                    rbdups.append(sum(counts))
            r = ranges[i] - fmin
            plot_hist(bdups, c, label, outdir, ext=f"-{hex(r)}")

def fit_and_plot_distribution(args):
    x, csv, xlabel, outdir, ext, distribution = args

    # Get the fitted parameters and calculate the sum of squared errors
    try:
        params = distribution.fit(x)
        sse = np.sum((x - distribution(*params).rvs(len(x))) ** 2)
    except Exception as e:
        print(f"Error fitting {distribution.name}: {e}")
        return None

    plt.clf()

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams["figure.figsize"] = (18.5, 10.5)

    plt.hist(x, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    pdf = distribution(*params).pdf(x)
    plt.plot(x, pdf, label=f"{distribution.name}")
    plt.xlabel(xlabel)
    plt.legend(loc='upper right')

    psize = basename(dirname(csv))
    prefix = splitext(basename(csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}-{distribution.name}")

    plt.xticks(rotation=90, fontsize='small')

    full_outdir = f"{outdir}/{prefix}/"
    if not os.path.exists(full_outdir):
        os.makedirs(full_outdir)
    figname = full_outdir + (splitext(basename(csv))[0] + "-" + psize + f"-dupshist{ext}-{distribution.name}.png").replace("_", "-")

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)

    return (sse, distribution, params)

def plot_hist(x, csv, xlabel, outdir, ext=""):
    distributions = [
        stats.alpha, stats.beta, stats.gamma, stats.norm, stats.lognorm, stats.uniform,
        stats.expon, stats.logistic, stats.cauchy, stats.weibull_min, stats.weibull_max
    ]

    # Prepare the arguments for the fit_and_plot_distribution function
    args = [(x, csv, xlabel, outdir, ext, distribution) for distribution in distributions]

    # Use a multiprocessing Pool to parallelize the fitting and plotting of distributions
    with multiprocessing.Pool() as pool:
        results = pool.map(fit_and_plot_distribution, args)

    # Filter out None results (due to errors) and find the best distribution and parameters
    results = [r for r in results if r is not None]
    best_sse, best_distribution, best_params = min(results, key=lambda x: x[0])

    psize = basename(dirname(csv))
    prefix = splitext(basename(csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    full_outdir = f"{outdir}/{prefix}/"
    open(full_outdir + "/best.txt", 'w').write(f"Best: {best_distribution.name}, {best_params}\n")

"""
def plot_hist(x, csv, xlabel, outdir, ext=""):
    # x dups
    # y faults

    distributions = [
        stats.alpha, stats.beta, stats.gamma, stats.norm, stats.lognorm, stats.uniform,
        stats.expon, stats.logistic, stats.cauchy, stats.weibull_min, stats.weibull_max
    ]

    # Best fitting distribution and parameters
    best_distribution = None
    best_params = None
    best_sse = np.inf

    psize = basename(dirname(csv))  # .split("_")[-1]
    prefix = splitext(basename(csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    full_outdir = f"{outdir}/{prefix}/"
    # Fit each distribution to the data
    for distribution in distributions:
        try:
            # Get the fitted parameters
            params = distribution.fit(x)

            # Calculate the sum of squared errors
            sse = np.sum((x - distribution(*params).rvs(len(x))) ** 2)

        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")
            continue
        # Update the best distribution if the current one has a lower SSE
        if sse < best_sse:
            best_distribution = distribution
            best_params = params
            best_sse = sse

        plt.clf()

        matplotlib.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams["figure.figsize"] = (18.5, 10.5)

        plt.hist(x, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
        pdf = distribution(*params).pdf(x)
        plt.plot(x, pdf, label=f"{distribution.name}")
        plt.xlabel(xlabel)
        plt.legend(loc='upper right')

        plt.title(f"{prefix}-{distribution.name}")

        plt.xticks(rotation=90, fontsize='small')

        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(csv))[0] + "-" + psize + f"-dupshist{ext}-{distribution.name}.png").replace(
            "_", "-")

        plt.tight_layout()
        print('saving figure:', figname)
        plt.savefig(figname)
    open(full_outdir + "/best.txt", 'w').write(f"Best: {best_distribution.name}, {best_params}\n")
"""

if __name__== "__main__":
    main()

