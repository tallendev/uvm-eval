import yaml
import sys
import re
from os.path import basename
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (12, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         #'legend.fontsize': 'xx-large'
         }
pylab.rcParams.update(params)


copy = glob("../BabelStream-copy/out/stream*.log")
copy = sorted(copy, key=lambda x: int(basename(x).split('m')[1].split('.')[0]))


uvm = glob("out/stream*.log")
uvm = sorted(uvm, key=lambda x: int(basename(x).split('m')[1].split('.')[0]))

bw =   re.compile('^Bandwidth \(GB/s\): (.*)')
size = re.compile('^Total size: .* \(=(.*) MB\)')
#ptriad = re.compile('^Triad:(.*)')

uvm_copy = []
copy_sizes = []

uvm_triad = []
triad_sizes = []
pfaults = []
print copy

for f in copy:
    with open (f, "r") as log:
        for line in log:
            col = 0
            results = bw.findall(line)
	    if results != []:
		results = results[0].strip().split()
	        uvm_copy.append(results[col])
            results = size.findall(line)
	    if results != []:
		results = results[0].strip().split()
#MB to gb
	        copy_sizes.append( float(results[col])/1000)

for f in uvm:
    with open (f, "r") as log:
        for line in log:
            col = 0
            results = bw.findall(line)
	    if results != []:
		results = results[0].strip().split()
	        uvm_triad.append(results[col])
            results = size.findall(line)
	    if results != []:
		results = results[0].strip().split()
#MB to gb
	        triad_sizes.append( float(results[col])/1000)

print uvm_copy
print copy_sizes
print uvm_triad
print triad_sizes
assert len(uvm_triad) == len(triad_sizes), "len(uvm_triad) != len(triad_sizes), %d != %d" % (len(uvm_triad), len(triad_sizes))
assert len(uvm_copy) == len(copy_sizes), "len(uvm_copy) != len(copy_sizes), %d != %d" % (len(uvm_copy), len(copy_sizes))

ticks = range(1, len(triad_sizes) + 1)


plt.plot(ticks, uvm_copy, color='xkcd:blue violet', marker='s')
plt.plot(ticks, uvm_triad, color='xkcd:seafoam', marker='^')


plt.title("Stream Cold Access Problem Size/Time on V100 (16GB)")
plt.ylabel("GB/s")
#plt.ylabel("Stream Time (s)")
plt.xlabel("Total Memory Size (GB)")
plt.xticks(ticks, [ '{0:.1f}'.format(i) for i in triad_sizes], ha="center")
leg = ["Copy", "Triad"] #, "Copy", "Scale", "Add", "Triad"]

colors = ['xkcd:blue violet', 'xkcd:seafoam'] #, 'xkcd:orangeish', 'xkcd:blue violet']
markers = ['s', '^'] #, '*', '+', '^']
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

handles = [f("s", colors[i]) for i in range(len(colors))]
handles += [f(markers[i], "k") for i in range(len(markers))]

plt.legend(leg, loc='upper right')
plt.tight_layout()

plt.grid(True, axis="y")
plt.savefig("triad-v100.png", dpi=300)


