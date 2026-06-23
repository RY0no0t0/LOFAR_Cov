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

# def format_time(sec):
#     td = timedelta(seconds=int(round(sec)))
#     td_str = str(td)
#     days = 0

#     if "day" in td_str:
#         day_part, time_part = td_str.split(", ")
#         days = int(day_part.split()[0])
#         h, m, s = time_part.split(":")
#     else:
#         h, m, s = td_str.split(":")

#     parts = []
#     if days:
#         parts.append(f"{days} d")
#     if int(h) or days:
#         parts.append(f"{int(h)} h")
#     if int(m) or int(h) or days:
#         parts.append(f"{int(m)} min")
#     parts.append(f"{int(float(s))} sec")

#     return " ".join(parts)

def get_real(trace, cut, duration, start):
    s = start
    while True:
        if s+duration >= len(trace):
            raise IndexError(f"Index exceeded: {s+duration}")
        real = trace[s:s+duration]
        if np.abs(real).max() < cut:
            break
        else:
            s = np.where(np.abs(real) >= cut)[0][-1]+1+s

    return real, s+duration

def get_reals(trace, cut, duration):
    s = 0
    reals = []
    while True:
        try:
            real, end = get_real(trace, cut, duration, s)
        except IndexError:
            break
        else:
            reals.append(real)
            s = end
    
    to_return = np.array(reals)
    print(f"Number of realizations: {len(to_return)}")
    return to_return

def make_cov(trace, cut, duration):
    reals = get_reals(trace, cut, duration)
    N = len(reals)

    if N==0:
        return np.zeros((duration, duration)), N
    
    return np.cov(reals.T), N

def draw_FullTrace(trace, dirname):
    plt.figure()

    rows = len(trace)//10000+1
    fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(12,2*rows), sharey=True)

    axes = np.atleast_1d(axes)

    for i in range(rows-1):
        axes[i].plot(traces[10000*i:10000*(i+1)])
    axes[-1].plot(np.concatenate((traces[(rows-1)*10000:], np.zeros(rows*10000-len(traces))), axis=None))

    rect = Rectangle((len(traces)-(rows-1)*10000, -0.001), (rows*10000-len(traces)), 0.002, fc="lightgray")
    axes[-1].add_patch(rect)

    fn = os.path.join(dirname, "full_trace.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

def draw_cov(trace, trim, ratios, durs, dirname):
    tc = trace[trim:-trim].reset_index(drop=True).values.flatten()
    maxamp = np.abs(tc).max()
    cuts = maxamp/ratios
    ratios_string = np.char.replace(ratios.astype(str), '.', '-')

    cutsN = len(cuts)
    dursN = len(durs)

    dircovname = os.path.join(dirname, "Covs")
    if os.path.exists(dircovname) == False:
        os.mkdir(dircovname)

    fig, axes = plt.subplots(nrows=cutsN, ncols=dursN, figsize=(5*dursN, 4*cutsN))

    axes = np.atleast_2d(axes)

    for i in range(cutsN):
        for j in range(dursN):
            cov, N = make_cov(tc, cuts[i], durs[j])
            sns.heatmap(cov, xticklabels=False, yticklabels=False, ax=axes[i,j], square=True, cmap="bwr", norm=colors.CenteredNorm())
            axes[i,j].set_title(f"N={N}")

            if i==cutsN-1:
                axes[cutsN-1, j].set_xlabel(f"duration={durs[j]} bins")

            cov_name = f"Cov_"+ratios_string[i]+f"_{durs[j]}.npy"
            fn_cov = os.path.join(dircovname, cov_name)
            np.save(fn_cov, cov)

        axes[i,0].set_ylabel(f"cut=maxamp/{ratios[i]}")

    fn = os.path.join(dirname, "cov.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    fig, axes = plt.subplots(nrows=cutsN, ncols=1, figsize=(12, 2*cutsN))

    for i in range(cutsN):
        axes[i].plot(tc)
        axes[i].axhline(y=cuts[i], color="tab:red")
        axes[i].axhline(y=-cuts[i], color="tab:red")
        axes[i].set_ylabel(f"cut=maxamp/{ratios[i]}")

    fn = os.path.join(dirname, "cut.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

current = os.getcwd()
bigfolder = os.path.join(current, "results/cuts")
if os.path.exists(bigfolder) == False:
    os.mkdir(bigfolder)

#User arguments
if len(sys.argv) != 6:
    print("User argument must include \n" \
    "1. Name of the file of the trace \n" \
    "2. Name of output folder \n" \
    "3. Number of units trimmed at the beginning and at the end of trace: Default 6000 \n" \
    "4. List of ratios relative to max amplitude for cuts (without [] separated by commas and no space) \n" \
    "5. List of number of bins as noise window (formatted the same)")
    sys.exit(1)

trace_name = sys.argv[1]
foldername = sys.argv[2]
dirname = os.path.join(bigfolder, foldername)
trim = int(sys.argv[3])
ratios = np.array([float(x) for x in sys.argv[4].split(",")])
durs = np.array([int(x) for x in sys.argv[5].split(",")])

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
draw_cov(traces, trim, ratios, durs, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Finished Covariance Matrix: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'a') as f:
    f.write(f"Finished Covariance Matrix: {elapsed}\n")