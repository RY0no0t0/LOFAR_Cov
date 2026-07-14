#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import h5py
import os
import sys
import random
import time
from itertools import groupby

#start the time counter
start = time.time()

def add_timestamp(start, fn_time, string):
    middle = time.time()
    elapsed = middle-start
    print(string+f": {elapsed}", flush=True)
    with open(fn_time, 'a') as f:
        f.write(string+f": {elapsed}\n")


def format_consecutive_ranges(numbers):
    sorted_nums = sorted(set(numbers)) 
    ranges = []
    
    # Group numbers where (number - index) is identical
    for _, g in groupby(enumerate(sorted_nums), lambda pair: pair[1] - pair[0]):
        group = [t[1] for t in g]
        
        if len(group) > 1:
            ranges.append(f"{group[0]}-{group[-1]}")
        else:
            ranges.append(str(group[0]))
            
    return ", ".join(ranges)

def check_empty_channels(traces, fn_time):

    for i in range(traces.shape[0]):
        all0 = []
        one0 = []
        for j in range(traces.shape[1]):
            if (traces[i,j,j%2]==0).all() and (traces[i,j,(j+1)%2]==0).all():
                all0.append(j)
            elif (traces[i,j,j%2]==0).all() or (traces[i,j,(j+1)%2]==0).all():
                one0.append(j)
        print(f"stn {i} | empty channels:{format_consecutive_ranges(all0)} | one polarization empty: {format_consecutive_ranges(one0)}", flush=True)
        with open(fn_time, 'a') as f:
            f.write(f"stn {i} | empty channels:{format_consecutive_ranges(all0)} | one polarization empty: {format_consecutive_ranges(one0)} \n")

def modify_data(raw_traces, trim):
    new = raw_traces.reshape(-1, 2, raw_traces.shape[-1]) #stack all stations together
    deleted = new[~np.any(np.all(new==0,axis=2), axis=1)] #delete all channels if at least one of the polarization has all 0 data
    reduced = deleted.reshape(-1, deleted.shape[-1]) #stack all polarizations together, making it 2D array of (channels, time_bins)
    return reduced[:, trim:-trim]

def read_file(fn):
    try:
        with h5py.File(fn, 'r') as f:
            traces = f["traces"][:9]
            if traces.shape[2] != 2 or np.argmax(traces.shape)!= 3:
                print(f"{fn}: The shape seems to mismatch, got {traces.shape}. It should be (stn, chn, 2(pol), time_bin (biggest))")
                sys.exit(1)
            return traces
    except Exception as e:
        print(f"{fn}: SKIPPED (File is corrupted or invalid HDF5)")
        sys.exit(1)

def calc_rms(trace, length):
    return np.sqrt(np.mean(np.square(trace[:length])))

def remove_signal(trace, sig_window, rms, threshold):
    if np.max(trace)/rms > threshold:
        sig = np.argmax(trace)
        return np.delete(trace, np.arange(sig-sig_window,sig+sig_window+1)), True
    return trace, False

def get_reals(traces, dur, sig_window, rms, threshold, fn_time):
    reals = []
    n_removed = 0
    for trace in traces:
        new, rem = remove_signal(trace, sig_window, rms, threshold)
        to_consider = new[:-(len(new)%dur)]
        reals.append(np.split(to_consider, len(to_consider)/dur))
        if rem:
            n_removed += 1
        
    print(f"{n_removed}/{len(traces)} had above SNR {threshold} signal", flush=True)
    with open(fn_time, 'a') as f:
        f.write(f"{n_removed}/{len(traces)} had above SNR {threshold} signal \n")

    return np.vstack(reals)

def make_cov(traces, duration, sig_window, rms_window, threshold, fn_time):
    rms = calc_rms(traces[0], rms_window)
    reals = np.array(get_reals(traces, duration, sig_window, rms, threshold, fn_time))
    N = len(reals)

    if N==0:
        return np.zeros((duration, duration)), N

    return np.cov(reals.T), N
    
def draw_cov(cov, N, dur, dirname, evt_name):
    # Make plots
    plt.figure()
    sns.heatmap(cov, xticklabels=False, yticklabels=False, square=True, cmap="bwr", norm=colors.CenteredNorm())
    plt.suptitle(evt_name)
    plt.title(f"Size: {dur} ({dur*5} ns), {N} realizations")
    plt.tight_layout()

    fn = os.path.join(dirname, "cov.png")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()
    

def draw_1D(cov, x_time, dur, dirname, evt_name):
    # Plot!
    plt.figure(figsize=(12,6))
    plt.axhline(y=0, c="gray", linestyle="--")
    plt.plot(x_time[:50], cov[0, :50])
    plt.scatter(x_time[:50], cov[0, :50], marker=".")

    plt.xlabel(r"$\Delta t_{i,j}$ [ns]")
    plt.ylabel(r"Cov($\Delta t_{i,j}$)")
    
    plt.suptitle(evt_name)
    plt.title(f"1D function of the first row for first 50 bins. Size: {dur} ({dur*5} ns), {N} realizations")
    plt.tight_layout()

    fn = os.path.join(dirname, "1d.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

current = os.getcwd()
bigfolder = os.path.join(current, "results/Covs")
if os.path.exists(bigfolder) == False:
    os.mkdir(bigfolder)

#User arguments
if len(sys.argv) != 7:
    print("User argument must include \n" \
    "1. Name of the folder of the traces \n" \
    "2. Number of bins trimmed at the beginning and at the end of trace \n" \
    "3. Number of bins as noise window \n" \
    "4. Signal window in units of bins \n" \
    "5. Number of bins to calculate RMS from \n" \
    "6. SNR threshold at which we define signal")
    sys.exit(1)

fn = sys.argv[1]
trim = int(sys.argv[2])
dur = int(sys.argv[3])
sig_window = int(sys.argv[4])
rms_window = int(sys.argv[5])
threshold = float(sys.argv[6])

evt_name = fn.rsplit('/', 1)[-1].split('.', 1)[0]
foldername = ""+evt_name+f"_{dur}"
dirname = os.path.join(bigfolder, foldername)

print("Output Directory %s" % dirname, flush=True)
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

#Read trace
raw_trs = read_file(fn)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Read data: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'w') as f:
    f.write(f"Read data: {elapsed}\n")

# check empty channels
check_empty_channels(raw_trs, fn_time)

# Modify data
trs = modify_data(raw_trs, trim)

# Make covariance matrix
add_timestamp(start, fn_time, "Making Covariance Matrix")
cov, N = make_cov(trs, dur, sig_window, rms_window, threshold, fn_time)
cov_name = f"Cov.npy"
fn_cov = os.path.join(dirname, cov_name)
np.save(fn_cov, cov)
add_timestamp(start, fn_time, "Finished making matrix")

print(f"Number of Realization: {N}", flush=True)
with open(fn_time, 'a') as f:
    f.write(f"Number of Realization: {N} \n")

# Plots
# Heatmap
add_timestamp(start, fn_time, "Drawing Covariance Matrix")
draw_cov(cov, N, dur, dirname, evt_name)
add_timestamp(start, fn_time, "Finished Drawing Matrix")

add_timestamp(start, fn_time, "Drawing 1D plot")
with h5py.File(fn, 'r') as f:
    x_time = f["times"][0,1,0] # For x-axis (Maybe I should check that all traces[i,0] are the same)
draw_1D(cov, x_time, dur, dirname, evt_name)
add_timestamp(start, fn_time, "Finished Drawing 1D plot")

# Print time
add_timestamp(start, fn_time, "Finished Everything")