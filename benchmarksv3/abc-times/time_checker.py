#!/usr/bin/python3

import sys
import argparse
import sys
import os
import argparse
from os.path import basename, splitext, dirname

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    c = args.csv

    csv_lines = None
    with open(c, "r") as csv:
        csv_lines = csv.readlines()

    csv_lines = [ [int(k) for k in line.split(',')] for line in csv_lines]
    sorted(csv_lines, key=lambda x: x[0])

    for line in csv_lines:
        tid = line[0]
        a_time = line[2] - line[1]
        b_time = line[3] - line[2]
        c_time = line[4] - line[3]
        a_time2 = line[5] - line[4]
        b_time2 = line[6] - line[5]
        c_time2 = line[7] - line[6]
        a_time3 = line[8] - line[7]
        b_time3 = line[9] - line[8]
        c_time3 = line[10] - line[9]

        print("tid, a, b, c, a, b, c, a, b, c:", tid, a_time, b_time, c_time, a_time2, b_time2, c_time2, a_time3, b_time3, c_time3)

    base = min(csv_lines, key=lambda x: x[1])[1]
    for line in csv_lines:
        print("tid, mina, minb:", line[0], line[1] - base, line[2]-base)

    for line in csv_lines:
        print("tid, adiff, adiff2", line[0], line[4] - line[1], line[7]-line[4], line[10] - line[7])
    


if __name__== "__main__":
  main()
