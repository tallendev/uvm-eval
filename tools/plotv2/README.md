# Overview:
This directory contains scripts and modules for generating plots from the paper. Scripts with
run examples are used to generate the plots; the others are accessory modules.

## Batch Data Files:
Toools for plotting fault patterns from the batchd- versions of the driver.
### Batch.py
reads input files and organizes them into a class hierarchy; has some utility functions
### genfigs.py:
Generate batch\_dist\_plot.py figures for all inputs defined in heading\
python3 ./genfigs.py
### genfigs.sh:
Run genfigs.py without thread explosion (PREFERRED)\
./genfigs.sh
### batch\_dist\_plot.py
a feature-crept file that does most figures from uvm-system-arch paper; refer to genfigs.sh
### dup.sh
generate fault duplicate plots - uncomment desired benchmarks at the top of the file\
./dup.sh
### perf.sh
generate batch size performance plots - uncomment desired benchmarks at top of file\
./perf.sh
### dup\_plot.py
plots related to fault duplicates
### perf\_bs\_plot.py
Performance over batch sizes; probably not used for this project


## Fault Data Files:
Tools for plotting fault patterns from the faults- versions of the driver.
### Fault.py
fault-style for reading CSVs and organizing them into a class hierarchy
### read\_write\_batch\_plot.py
Plots and organizes faults by r/w\
python3 ./read\_write\_batch\_plot.py [output.txt generated using faults driver]\
Note: tools/plot has some legacy tools for other kinds of access pattern plotting
