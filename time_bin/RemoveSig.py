#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import os
import sys
import time
from datetime import timedelta

#start the time counter
start = time.time()

def remove_signal(trace, sig_window):
    sig = np.argmax(trace)
    return np.delete(trace, np.arange(sig-sig_window,sig+sig_window+1))

def get_reals(trace, dur):
    to_consider = trace[:-(len(trace)%dur)]
    return np.split(to_consider, len(to_consider)/dur)

def make_cov(trace, duration, sig_window):
    new_trace = remove_signal(trace, sig_window)
    reals = get_reals(new_trace, duration)
    N = len(reals)
    print(f"Number of realizations: {N}")

    if N==0:
        return np.zeros((duration, duration)), N
    
    mus = np.average(reals, axis=0)
    diffs = reals-mus

    cov = np.empty((duration, duration))
    for i in range(duration):
        for j in range(duration-i):
            cov[i, i+j] = np.sum(diffs[:,i]*diffs[:,j])/N
            cov[i+j,j] = cov[i,i+j]

    return cov, N

def draw_FullTrace(trace, dirname):
    plt.figure()

    rows = len(trace)//10000+1
    fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(12,2*rows), sharey=True)

    for i in range(rows-1):
        axes[i].plot(traces[10000*i:10000*(i+1)])
    axes[-1].plot(np.concatenate((traces[(rows-1)*10000:], np.zeros(rows*10000-len(traces))), axis=None))

    rect = Rectangle((len(traces)-(rows-1)*10000, -0.001), (rows*10000-len(traces)), 0.002, fc="lightgray")
    axes[-1].add_patch(rect)

    fn = os.path.join(dirname, "full_trace.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

def draw_cov(trace, trim, durs, sig_window, dirname):
    tc = trace[trim:-trim].reset_index(drop=True).values.flatten()

    dursN = len(durs)

    dircovname = os.path.join(dirname, "Covs")
    if os.path.exists(dircovname) == False:
        os.mkdir(dircovname)

    fig, axes = plt.subplots(nrows=1, ncols=dursN, figsize=(5*dursN, 4))

    for j in range(dursN):
        cov, N = make_cov(tc, durs[j], sig_window)
        sns.heatmap(cov, xticklabels=False, yticklabels=False, ax=axes[j], square=True, cmap="bwr", norm=colors.CenteredNorm())
        axes[j].set_title(f"N={N}")
        axes[j].set_xlabel(f"duration={durs[j]*5} ns")

        cov_name = f"Cov_{durs[j]}.npy"
        fn_cov = os.path.join(dircovname, cov_name)
        np.save(fn_cov, cov)

    fn = os.path.join(dirname, "cov.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

current = os.getcwd()
bigfolder = os.path.join(current, "results/remove")
if os.path.exists(bigfolder) == False:
    os.mkdir(bigfolder)

#User arguments
if len(sys.argv) != 6:
    print("User argument must include \n" \
    "1. Name of the file of the trace \n" \
    "2. Name of output folder \n" \
    "3. Number of units trimmed at the beginning and at the end of trace: Default 6000 \n" \
    "4. List of number of bins as noise window (without [] separated by commas and no space) \n" \
    "5. Signal window in units of bins")
    sys.exit(1)

trace_name = sys.argv[1]
foldername = sys.argv[2]
dirname = os.path.join(bigfolder, foldername)
trim = int(sys.argv[3])
durs = np.array([int(x) for x in sys.argv[4].split(",")])
sig_window = int(sys.argv[5])

print("Output Directory %s" % dirname, flush=True)
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

#Read trace
traces = pd.read_csv(trace_name, delimiter="\t")

# Plot the full trace
draw_FullTrace(traces, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Making Covariance Matrix: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'w') as f:
    f.write(f"Making Covariance Matrix: {elapsed}\n")

#Make Covariance Matrix and Plot them
draw_cov(traces, trim, durs, sig_window, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Finished Covariance Matrix: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'a') as f:
    f.write(f"Finished Covariance Matrix: {elapsed}\n")