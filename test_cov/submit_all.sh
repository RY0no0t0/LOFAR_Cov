#!/bin/bash
# Submit the three reconstructions of all channels.
# SLURM opens the output files before the job starts, so the folders are made here:
cd "$(dirname "$0")"

for out in allchan_nondiag allchan_diag allchan_ska
do
    if [[ ! -d "results/$out" ]]; then mkdir -p "results/$out"; fi
done

sbatch all_chan_nondiag.sb
sbatch all_chan_diag.sb
sbatch all_chan_ska.sb
