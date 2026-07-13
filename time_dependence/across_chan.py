#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import os
import sys
import random
import time

#start the time counter
start = time.time()

def get_random_channel(size):
    #87, 87, 85, 87, 87, 85
    stns = []
    chns = []
    for i in range(size):
        stn = int(6*random.random())
        stns.append(stn)
        if stn in (2,5):
            chns.append(int(86*random.random()))
        else:
            chns.append(int(88*random.random()))

    return stns, chns

def modify_data(data, trim):
    modified = pd.concat([data.columns.to_frame().T, data]).reset_index(drop=True).iloc[:,0].str.split(expand=True).astype(float)
    return modified.values[trim:-trim].T

def read_files(size, foldername, trim):
    stns, chns = get_random_channel(size)

    for i in range(size):
        fn = f"traces_station{stns[i]}_channel{chns[i]}_pol{chns[i]%2}.dat"
        data = pd.read_csv(foldername+"/"+fn, delimiter="\t")
        modified = modify_data(data, trim)

        if i == 0:
            traces = np.empty((size, *modified.shape))
        
        traces[i] = modified
    
    return traces, stns, chns

def remove_signal(trace, sig_window):
    sig = np.argmax(trace)
    return np.delete(trace, np.arange(sig-sig_window,sig+sig_window+1))

def get_reals(trace, dur):
    to_consider = trace[:-(len(trace)%dur)]
    return np.split(to_consider, len(to_consider)/dur)

def make_cov(trace, duration, sig_window):
    new_trace = remove_signal(trace, sig_window)
    reals = np.array(get_reals(new_trace, duration))
    N = len(reals)

    if N==0:
        return np.zeros((duration, duration)), N

    return np.cov(reals.T), N

def get_covs(traces, size, dur, sig_window, dirname):
    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Making Covariance Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Making Covariance Matrices: {elapsed}\n")

    covs = [None]*size

    # dircovname = os.path.join(dirname, "Covs")
    # if os.path.exists(dircovname) == False:
    #     os.mkdir(dircovname)

    for j in range(size):
        cov, N = make_cov(traces[j,1], dur, sig_window)
        covs[j] = cov

        # cov_name = f"Cov_{dur}.npy"
        # fn_cov = os.path.join(dircovname, cov_name)
        # np.save(fn_cov, cov)

    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Finished making matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished making matrices: {elapsed}\n")

    return covs, N
    
def draw_cov(covs, size, N, dur, stns, chns, dirname):
    #Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing Covariance Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing Covariance Matrices: {elapsed}\n")
    
    # Make plots
    rownum = size//3+1
    fig, axes = plt.subplots(nrows=rownum, ncols=3, figsize=(12, 3*rownum))

    axes = np.atleast_2d(axes)

    for j in range(rownum*3):
        row = j//3
        col = j%3

        if j < size:
            sns.heatmap(covs[j], xticklabels=False, yticklabels=False, ax=axes[row, col], square=True, cmap="bwr", norm=colors.CenteredNorm())
            axes[row, col].set_title(f"{stns[j]}-{chns[j]}")
        else:
            axes[row, col].set_visible(False)
    
    plt.suptitle(f"Size: {dur} ({dur*5} ns), {N} realizations")
    plt.tight_layout()

    fn = os.path.join(dirname, "covs.png")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()

    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing Matrices: {elapsed}\n")
    

def draw_1D(covs, size, x_time, dur, stns, chns, dirname):
    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing 1D plots: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing 1D plots: {elapsed}\n")
    
    # Plot!
    rownum = size//3+1
    fig, axes = plt.subplots(nrows=rownum, ncols=3, figsize=(12, 3*rownum), sharex=True, sharey=True)

    axes = np.atleast_2d(axes)

    for j in range(rownum*3):
        row = j//3
        col = j%3

        if row == rownum-1:
            axes[row, col].set_xlabel(r"$\Delta t_{i,j}$ [ns]")
        if col == 0:
            axes[row, col].set_ylabel(r"Cov($\Delta t_{i,j}$)")

        if j < size:
            axes[row, col].plot(x_time[:50], covs[j][0,:50])
            axes[row, col].scatter(x_time[:50], covs[j][0,:50], marker=".")
            axes[row, col].set_title(f"{stns[j]}-{chns[j]}")
        else:
            axes[row, col].set_visible(False)
            axes[row-1, col].set_xlabel(r"$\Delta t_{i,j}$ [ns]")
            axes[row-1, col].tick_params(labelbottom=True)
    
    plt.suptitle(f"1D function of the first row for first 50 bins. Size: {dur} ({dur*5} ns), {N} realizations")
    plt.tight_layout()

    fn = os.path.join(dirname, "1d.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    # Print Time
    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing 1D plot: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing 1D plot: {elapsed}\n")

def overplot_1D(covs, size, x_time, dur, stns, chns, dirname):
    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing 1D plots: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing 1D plots: {elapsed}\n")

    # Plot!
    plt.figure(figsize=(12, 8))
    for i in range(50):
        plt.axvline(x=x_time[i], color="lightgray", linestyle="--")
    plt.axhline(y=0, color="lightgray")
    for j in range(size):
        plt.plot(x_time[:50], covs[j][0,:50], label=f"{stns[j]}-{chns[j]}")
        plt.scatter(x_time[:50], covs[j][0,:50], marker=".")
    plt.legend()
    plt.title(f"Overplotting 1D functions for first 50 bins (size: {dur}, {size} antennas)")

    fn = os.path.join(dirname, "overplot.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    # Print Time
    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing overplot: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing overplot: {elapsed}\n")


current = os.getcwd()
bigfolder = os.path.join(current, "results/random")
if os.path.exists(bigfolder) == False:
    os.mkdir(bigfolder)

#User arguments
if len(sys.argv) != 7:
    print("User argument must include \n" \
    "1. Name of the folder of the traces \n" \
    "2. Name of output folder \n" \
    "3. Number of bins trimmed at the beginning and at the end of trace \n" \
    "4. Number of bins as noise window \n" \
    "5. Signal window in units of bins \n" \
    "6. Number of channels to compare")
    sys.exit(1)

trace_folder = sys.argv[1]
foldername = sys.argv[2]
dirname = os.path.join(bigfolder, foldername)
trim = int(sys.argv[3])
dur = int(sys.argv[4])
sig_window = int(sys.argv[5])
size = int(sys.argv[6])

print("Output Directory %s" % dirname, flush=True)
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

#Read trace
traces, stns, chns = read_files(size, trace_folder, trim)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Read data: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'w') as f:
    f.write(f"Read data: {elapsed}\n")

# Make covariance matrix
covs, N = get_covs(traces, size, dur, sig_window, dirname)

# Plots
draw_cov(covs, size, N, dur, stns, chns, dirname)
x_time = traces[0,0] # For x-axis (Maybe I should check that all traces[i,0] are the same)
draw_1D(covs, size, x_time, dur, stns, chns, dirname)
overplot_1D(covs, size, x_time, dur, stns, chns, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Finished Everything: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'a') as f:
    f.write(f"Finished Everything: {elapsed}\n")