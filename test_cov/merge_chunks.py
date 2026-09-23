#!/usr/bin/env python
import numpy as np
import glob
import os
import sys

# Put the chunks of channels written by all_chan.py back into a single file each

names = ["means", "stds", "rmses", "norm_rmses"]

def get_chunks(dirname, name):
    fns = glob.glob(os.path.join(dirname, f"{name}_*_*.npy"))
    # Sort them by the first channel they hold
    fns.sort(key=lambda fn: int(os.path.basename(fn).rsplit("_", 2)[-2]))
    return fns

def check_chunks(fns):
    # The chunks have to follow each other without a hole or an overlap
    last = None
    for fn in fns:
        first, stop = [int(x) for x in os.path.basename(fn)[:-4].rsplit("_", 2)[-2:]]
        if last is not None and first != last:
            print(f"{fn}: The chunks are not continuous, channel {last} is missing or repeated", flush=True)
            return False
        last = stop
    return True


current = os.getcwd()
bigfolder = os.path.join(current, "results")

#User arguments
if len(sys.argv) != 2:
    print("User argument must include \n" \
    "1. Name of the folder of the results (inside results)")
    sys.exit(1)

foldername = sys.argv[1]
dirname = os.path.join(bigfolder, foldername)

if os.path.exists(dirname) == False:
    print(f"{dirname}: The folder does not exist")
    sys.exit(1)

print("Merging %s" % dirname, flush=True)

for name in names:
    fns = get_chunks(dirname, name)

    if len(fns) == 0:
        print(f"{name}: No chunk to merge", flush=True)
        continue
    if check_chunks(fns) == False:
        sys.exit(1)

    merged = np.concatenate([np.load(fn) for fn in fns])
    np.save(os.path.join(dirname, f"{name}.npy"), merged)

    print(f"{name}: {len(fns)} chunks merged into {merged.shape}", flush=True)
