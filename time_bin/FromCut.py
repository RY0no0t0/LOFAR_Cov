#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import os
import sys
import time
from datetime import timedelta

#start the time counter
start = time.time()

def format_time:

def draw_FullTrace(trace):
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12,12), sharey=True)

    for i in range(6):
        axes[i].plot(traces[10000*i:10000*(i+1)])
    axes[6].plot(np.concatenate((traces[60000:], np.zeros(70000-len(traces))), axis=None))

    rect = Rectangle((len(traces)-60000, -0.001), (70000-len(traces)),0.002, fc="lightgray")
    axes[6].add_patch(rect)

    plt.show()


#User arguments
if len(sys.argv) not in (3, 4):
    print("User argument must include \n 1. Name of the file of the trace \n" \
    "2. Name of output folder \n (3. Number of units trimmed at the beginning and at the end of trace: Default 6000)")
    sys.exit(1)

trace_name = sys.argv[1]
foldername = sys.argv[2]
dirname = "results/"+foldername

if len(sys.argv) == 4:
    trim = sys.argv[3]
else:
    trim = 6000

print("Output Directory %s" % dirname, flush=True)
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

#Read trace
traces = pd.read_csv(trace_name, delimiter="\t")
