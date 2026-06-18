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

def format_time(sec):
    td = timedelta(seconds=int(round(sec)))
    td_str = str(td)
    days = 0

    if "day" in td_str:
        day_part, time_part = td_str.split(", ")
        days = int(day_part.split()[0])
        h, m, s = time_part.split(":")
    else:
        h, m, s = td_str.split(":")

    parts = []
    if days:
        parts.append(f"{days} d")
    if int(h) or days:
        parts.append(f"{int(h)} h")
    if int(m) or int(h) or days:
        parts.append(f"{int(m)} min")
    parts.append(f"{int(float(s))} sec")

    return " ".join(parts)

def draw_FullTrace(trace, dirname):
    plt.figure()

    rows = len(trace)//10000+1
    fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(12,2*rows), sharey=True)

    for i in range(rows-1):
        axes[i].plot(traces[10000*i:10000*(i+1)])
    axes[-1].plot(np.concatenate((traces[(rows-1)*10000], np.zeros(rows*10000-len(traces))), axis=None))

    rect = Rectangle((len(traces)-(rows-1)*10000, -0.001), (rows*10000-len(traces)), 0.002, fc="lightgray")
    axes[-1].add_patch(rect)

    fn = os.path(dirname, "full_trace.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

def draw_cov():
    tc = traces[6000:-6000].reset_index(drop=True).values.flatten()
    maxamp = np.abs(tc).max()
    ratios = np.array([2.3, 2.7, 3.1, 3.5])
    cuts = maxamp/ratios
    durs = np.array([200, 500, 1000, 1500])

    cutsN = len(cuts)
    dursN = len(durs)

    fig, axes = plt.subplots(nrows=cutsN, ncols=dursN, figsize=(5*dursN, 4*cutsN))

    for i in range(cutsN):
        for j in range(dursN):
            cov, N = make_cov(tc, cuts[i], durs[j])
            sns.heatmap(cov, xticklabels=False, yticklabels=False, ax=axes[i,j], square=True, cmap="bwr", norm=colors.CenteredNorm())
            # if N != 0:
            #     sns.heatmap(cov, xticklabels=False, yticklabels=False, ax=axes[i,j], square=True, cmap="bwr", norm=colors.CenteredNorm())
            # else:
            #     sns.heatmap(cov, xticklabels=False, yticklabels=False, ax=axes[i,j], square=True, cmap=["white"], linecolor="lightgray", linewidths=1, cbar=False)
            axes[i,j].set_title(f"N={N}")

            if i==cutsN-1:
                axes[cutsN-1, j].set_xlabel(f"duration={durs[j]} units")

        axes[i,0].set_ylabel(f"cut=maxamp/{ratios[i]}")

    plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=cutsN, ncols=1, figsize=(12, 2*cutsN))

    for i in range(cutsN):
        axes[i].plot(tc)
        axes[i].axhline(y=cuts[i], color="tab:red")
        axes[i].axhline(y=-cuts[i], color="tab:red")
        axes[i].set_ylabel(f"cut=maxamp/{ratios[i]}")

    plt.show()
    plt.close()



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

# Plot the full trace
draw_FullTrace(traces, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Making Covariance Matrix: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'w') as f:
    f.write(f"Making Covariance Matrix: {elapsed}\n")

