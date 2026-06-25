#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import os
import sys
import time

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
    reals = np.array(get_reals(new_trace, duration))
    N = len(reals)
    print(f"Number of realizations: {N}")

    if N==0:
        return np.zeros((duration, duration)), N

    return np.cov(reals.T), N

def get_cov(trace, trim, durs, dursN, sig_window, dirname):
    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Making Covariance Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Making Covariance Matrices: {elapsed}\n")

    tc = trace[trim:-trim].reset_index(drop=True).values.flatten()
    covs = [None]*dursN
    Ns = [None]*dursN

    dircovname = os.path.join(dirname, "Covs")
    if os.path.exists(dircovname) == False:
        os.mkdir(dircovname)

    for j in range(dursN):
        cov, N = make_cov(tc, durs[j], sig_window)
        covs[j] = cov
        Ns[j] = N

        cov_name = f"Cov_{durs[j]}.npy"
        fn_cov = os.path.join(dircovname, cov_name)
        np.save(fn_cov, cov)

    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Finished making matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished making matrices: {elapsed}\n")

    return covs, Ns

def draw_cov(covs, Ns, durs, dursN, dirname):
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing Covariance Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing Covariance Matrices: {elapsed}\n")

    fig, axes = plt.subplots(nrows=1, ncols=dursN, figsize=(5*dursN, 4))

    axes = np.atleast_1d(axes)

    for j in range(dursN):
        sns.heatmap(covs[j], xticklabels=False, yticklabels=False, ax=axes[j], square=True, cmap="bwr", norm=colors.CenteredNorm())
        axes[j].set_title(f"Size: {durs[j]} x {durs[j]} ({durs[j]*5} ns)")
        axes[j].set_xlabel(f"(Number of Realizations = {Ns[j]})")

    fn = os.path.join(dirname, "cov.png")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    fn = os.path.join(dirname, "cov.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing Matrices: {elapsed}\n")

def draw_zoom(covs, Ns, durs, dursN, dirname):
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing Zoomed Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing Zoomed Matrices: {elapsed}\n")
    
    fig, axes = plt.subplots(nrows=1, ncols=dursN, figsize=(5*dursN, 4))

    axes = np.atleast_1d(axes)

    for j in range(dursN):
        sns.heatmap(covs[j][:50, :50], xticklabels=False, yticklabels=False, ax=axes[j], square=True, cmap="bwr", norm=colors.CenteredNorm())
        axes[j].set_title(f"Size: {durs[j]} x {durs[j]} ({durs[j]*5} ns)")
        axes[j].set_xlabel(f"(Number of Realizations = {Ns[j]})")
    
    plt.suptitle("Zoomed into first 50 bx 50")
    fn = os.path.join(dirname, "zoom.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing Zoomed Matrices: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing Zoomed Matrices: {elapsed}\n")

def draw_1D(covs, durs, dursN, dirname):
    middle = time.time()
    elapsed = middle-start
    print(f"Drawing 1D plot: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Drawing 1D plot: {elapsed}\n")
    
    fig, axes = plt.subplots(nrows=2, ncols=dursN, figsize=(5*dursN, 4), sharey=True)

    axes = np.atleast_2d(axes)

    for j in range(dursN):
        axes[0,j].plot(5*np.arange(50), covs[j][0,:50])
        axes[0,j].scatter(5*np.arange(50),covs[j][0,:50], marker='.')
        axes[0,j].set_title(f"Size: {durs[j]} x {durs[j]} ({durs[j]*5} ns)")
        axes[0,j].set_xlabel(r"$\Delta t_{i,j}$ [ns]")

        if durs[j]>=30:
            samples = np.linspace(0, durs[j]-20, 10).astype(int)
            for i in samples:
                axes[1,j].plot(5*np.arange(20), covs[j][i,i:20+i], label=f"Row {i}")
                axes[1,j].scatter(5*np.arange(20),covs[j][i,i:20+i], marker='.')
            axes[1,j].set_xlabel(r"$\Delta t_{i,j}$ [ns]")
            # axes[1,j].legend()
    
    axes[0,0].set_ylabel(r"Cov($\Delta t_{i,j}$)")
    axes[1,0].set_ylabel(r"Cov($\Delta t_{i,j}$)")

    plt.suptitle("Top: 1D function of the first row for first 50 intervals \n"
                   +"Bottom: 1D functiono of equally spaced 10 rows for first 20 intervals")
    fn = os.path.join(dirname, "1d.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()

    middle = time.time()
    elapsed = middle-start
    print(f"Finished Drawing 1D plot: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, "time.txt")
    with open(fn_time, 'a') as f:
        f.write(f"Finished Drawing 1D plot: {elapsed}\n")

def make_plots(trace, trim, durs, sig_window, dirname):
    dursN = len(durs)
    covs, Ns = get_cov(trace, trim, durs, dursN, sig_window, dirname)

    draw_cov(covs, Ns, durs, dursN, dirname)
    draw_zoom(covs, Ns, durs, dursN, dirname)
    draw_1D(covs, durs, dursN, dirname)



current = os.getcwd()
bigfolder = os.path.join(current, "results/remove")
if os.path.exists(bigfolder) == False:
    os.mkdir(bigfolder)

#User arguments
if len(sys.argv) != 6:
    print("User argument must include \n" \
    "1. Name of the file of the trace \n" \
    "2. Name of output folder \n" \
    "3. Number of units trimmed at the beginning and at the end of trace \n" \
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

# Print time
middle = time.time()
elapsed = middle-start
print(f"Read data and finished drawing trace: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'w') as f:
    f.write(f"Read data and finished drawing trace: {elapsed}\n")

#Make Covariance Matrix and Plot them
make_plots(traces, trim, durs, sig_window, dirname)

# Print time
middle = time.time()
elapsed = middle-start
print(f"Finished Everything: {elapsed}", flush=True)
fn_time = os.path.join(dirname, "time.txt")
with open(fn_time, 'a') as f:
    f.write(f"Finished Everything: {elapsed}\n")