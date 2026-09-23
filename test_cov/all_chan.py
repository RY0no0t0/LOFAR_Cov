#!/usr/bin/env python
import nifty.re as jft
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import os
import sys
import time

#start the time counter
start = time.time()

# enable float64 precision
jax.config.update("jax_enable_x64", True)

# initialize JAX random key
seed = 42
key = random.PRNGKey(seed)

# Parameters of the noise covariance (fit of the measured LOFAR data)
A = 2.89331354e-07
omega0 = 5.85403204e-01
gamma = 2.70651972e-02
p = 9.21301273e-01
C = -7.39389317e-03

# Priors of the correlated field
cf_kwargs = {
    "offset_mean": 0,
    "offset_std": (1.71e-4, 1e-5),
    "fluctuations": (5e-4, 8e-4),
    "loglogavgslope": (7., 2.5),
    "flexibility": (1.0, 1.0),
    "asperity": (1.0e-8, 1.0e-8),
    "prefix": "",
}

# Priors of the gaussian window (mean, std)
window_pos = (3.1e2, 50)
window_width = (30., 30)

# Settings of the minimization
n_vi_iterations = 6
n_samples = 4
sample_mode = "linear_resample"

# norm_rmse above which a channel is counted as suspicious
threshold = 0.1


def add_timestamp(start, fn_time, string):
    middle = time.time()
    elapsed = middle-start
    print(string+f": {elapsed}", flush=True)
    if fn_time is not None:
        with open(fn_time, 'a') as f:
            f.write(string+f": {elapsed}\n")

def format_time(elapsed):
    return f"{int(elapsed // 60)}m {elapsed % 60:.1f}s"

def show_progress(start, i, tot):
    now = time.time()
    eta = (now-start)*(tot-i)
    eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
    print(f"--- Iteration {i} | Estimated Time Remaining: {eta_str} ---", flush=True)
    return now

def cov(Ns, A, omega0, gamma, p, C):
    omega = np.fft.rfftfreq(1024, d=5e-9)/1e+8
    fourier = np.abs(A*(1/(1+(np.abs(omega-omega0)/gamma)**(2*p))+C))
    cov0 = np.fft.irfft(fourier)[:Ns]
    return jax.scipy.linalg.toeplitz(cov0)

def read_cov(fn, Ns):
    try:
        Cov = np.load(fn)
    except Exception:
        print(f"{fn}: Could not read the covariance matrix", flush=True)
        sys.exit(1)

    if Cov.ndim != 2 or Cov.shape[0] != Cov.shape[1]:
        print(f"{fn}: The covariance matrix must be square, got {Cov.shape}", flush=True)
        sys.exit(1)
    if Cov.shape[0] != Ns:
        print(f"{fn}: The size of the matrix ({Cov.shape[0]}) does not match the length of the traces ({Ns})", flush=True)
        sys.exit(1)

    return jnp.asarray(Cov)

def make_diagonal(Cov):
    # Keep the variance of each time bin but drop every correlation
    return jnp.diag(jnp.diag(Cov))

def invert(Cov, fn_time=None):
    # Inverse and inverse Cholesky factor of the covariance matrix
    L = jax.scipy.linalg.cholesky(Cov)

    if not np.all(np.isfinite(np.asarray(L))):
        jitter = 1e-10*np.trace(np.asarray(Cov))/Cov.shape[0]
        add_timestamp(start, fn_time, f"Cholesky failed, adding a jitter of {jitter} to the diagonal")
        Cov = Cov + jitter*jnp.eye(Cov.shape[0])
        L = jax.scipy.linalg.cholesky(Cov)

        if not np.all(np.isfinite(np.asarray(L))):
            print("The covariance matrix is not positive definite", flush=True)
            sys.exit(1)

    return jax.scipy.linalg.inv(Cov), jax.scipy.linalg.inv(L)

def get_cov(cov_type, cov_file, Ns, fn_time=None):
    if cov_type == "file":
        Cov = read_cov(cov_file, Ns)
    else:
        Cov = cov(Ns, A, omega0, gamma, p, C)

    if cov_type == "diag":
        Cov = make_diagonal(Cov)

    Inv_Cov, Inv_Std = invert(Cov, fn_time)

    noise_cov_inv = lambda x: jnp.matmul(Inv_Cov, x)
    noise_std_inv = lambda x: jnp.matmul(Inv_Std, x)

    return noise_cov_inv, noise_std_inv

def fieldmaker(shape, distances, prefix, **args):
    cfm = jft.CorrelatedFieldMaker(prefix=f"{prefix}")
    cfm.set_amplitude_total_offset(
        offset_mean=args["offset_mean"], offset_std=args["offset_std"]
    )
    args.pop("offset_mean")
    args.pop("offset_std")
    cfm.add_fluctuations(
        shape=shape,
        distances=distances,
        **args,
    )
    cf_model = cfm.finalize()

    return cf_model, cfm.power_spectrum

class SignalModel(jft.Model):
    def __init__(self, cf_model, window_mean_pos_prior, window_std_prior, x_grid):
        self.cf_model = cf_model
        self.window_mean_pos_prior = window_mean_pos_prior
        self.window_std_prior = window_std_prior
        self.x_grid = x_grid

        super().__init__(init=
                         self.cf_model.init | self.window_mean_pos_prior.init | self.window_std_prior.init
                         )

    def gaussian_window(self, x_grid, x0, sigma):
        return jnp.exp(-(x_grid-x0)**2/(2*sigma**2))

    def __call__(self, x):
        return self.cf_model(x) * self.gaussian_window(self.x_grid, self.window_mean_pos_prior(x), self.window_std_prior(x))

def calc_RMSE(y_mean, y_true):
    # calculate statistical measurements
    diff = y_mean - y_true
    return np.sqrt(np.mean(diff**2)) #Root mean square

def reconstruct(d, d_true, x_grid, noise_cov_inv, noise_std_inv, fn_time=None, key=key):
    tot = len(d)

    # Sampling of the traces, read from the time bins (which are in ns)
    distances = (x_grid[1]-x_grid[0])*1e-9

    n = 0
    means = []
    stds = []
    rmses = []
    norm_rmses = []
    now = time.time()

    for i in range(tot):

        now = show_progress(now, i, tot)

        jax.clear_caches()

        x0 = jft.NormalPrior(mean=window_pos[0], std=window_pos[1], name='mean_pos_window', shape=(1,))
        sigma = jft.NormalPrior(mean=window_width[0], std=window_width[1], name='std_window', shape=(1,))

        shape = len(x_grid)

        cf_model, ps = fieldmaker(shape=shape, distances=distances, **cf_kwargs)
        my_model = SignalModel(cf_model, x0, sigma, x_grid)

        lh = jft.Gaussian(data=d[i], noise_cov_inv=noise_cov_inv, noise_std_inv=noise_std_inv).amend(my_model)

        key, subkey = random.split(key, 2)
        init_pos = jft.Vector(lh.init(subkey))

        key, sampling_key = random.split(key, 2)

        draw_linear_kwargs = dict(
        cg_name=None,
        cg_kwargs=dict(absdelta=1e-5 * jft.size(lh.domain), maxiter=100),
        )

        kl_kwargs = dict(minimize_kwargs=dict(name=None, xtol=1e-4, maxiter=35))

        optimize_kl_args = dict(
            likelihood=lh,
            position_or_samples=init_pos,
            n_total_iterations=n_vi_iterations,
            n_samples=n_samples,
            key=sampling_key,
            draw_linear_kwargs=draw_linear_kwargs,
            kl_kwargs=kl_kwargs,
            sample_mode=sample_mode,
            )

        s, state = jft.optimize_kl(**optimize_kl_args)

        y_samples = tuple(my_model(s) for s in s)
        y_mean, y_std = jft.mean_and_std(y_samples)
        means.append(y_mean)
        stds.append(y_std)

        rmse = calc_RMSE(y_mean, d_true[i])
        norm_rmse = rmse/np.max(np.abs(d_true[i]))
        rmses.append(rmse)
        norm_rmses.append(norm_rmse)

        if norm_rmse > threshold:
            n += 1

    print("=== All Channels complete ===", flush=True)
    print(f"Total Suspicious Indices: {n}/{tot}\n", flush=True)
    if fn_time is not None:
        with open(fn_time, 'a') as f:
            f.write(f"Total Suspicious Indices: {n}/{tot} (norm_rmse>{threshold}) \n")

    return np.array(means), np.array(stds), np.array(rmses), np.array(norm_rmses), n, key

def write_details(dirname, suffix, cov_type, cov_file, data_name, true_name, time_name, first, last, tot, distances, n, elapsed):
    fn_details = os.path.join(dirname, f"details{suffix}.txt")

    with open(fn_details, 'w') as f:
        f.write("Reconstruction of all channels with correlated field model.\n\n")

        f.write("Covariance:\n")
        f.write(f"    Type: {cov_type}\n")
        if cov_type == "file":
            f.write(f"    File: {cov_file}\n")
        else:
            f.write("    Parameters:\n")
            f.write(f"    A={A}\n    omega0={omega0}\n    gamma={gamma}\n    p={p}\n    C={C}\n")
        if cov_type == "diag":
            f.write("    Only the diagonal (variance of each bin) of the matrix is kept\n")

        f.write("\nData:\n")
        f.write(f"    Noised traces: {data_name}\n")
        f.write(f"    True traces: {true_name}\n")
        f.write(f"    Time bins: {time_name}\n")
        f.write(f"    Channels: {first}-{last-1} out of {tot}\n")

        f.write("\nPriors:\n")
        f.write(f"    cf_kwargs = {cf_kwargs}\n")
        f.write(f"    x0 = jft.NormalPrior(mean={window_pos[0]}, std={window_pos[1]}, name='mean_pos_window', shape=(1,))\n")
        f.write(f"    sigma = jft.NormalPrior(mean={window_width[0]}, std={window_width[1]}, name='std_window', shape=(1,))\n")

        f.write("\nMinimization:\n")
        f.write(f"    distances={distances}\n")
        f.write(f"    n_vi_iterations={n_vi_iterations}\n    n_samples={n_samples}\n    sample_mode={sample_mode}\n")

        f.write("\nResults:\n")
        f.write(f"Total Suspicious Indices: {n}/{last-first} (norm_rmse>{threshold})\n")
        f.write(f"Runtime: {format_time(elapsed)}\n")


if __name__ == "__main__":

    current = os.getcwd()
    bigfolder = os.path.join(current, "results")
    os.makedirs(bigfolder, exist_ok=True)

    #User arguments
    if len(sys.argv) != 8:
        print("User argument must include \n" \
        "1. Type of the covariance matrix (nondiag, diag or file) \n" \
        "2. Name of the file of the covariance matrix (only read for file, put model otherwise) \n" \
        "3. Name of the file of the noised traces \n" \
        "4. Name of the file of the true traces \n" \
        "5. Name of the file of the time bins \n" \
        "6. Name of output folder \n" \
        "7. First channel and number of channels, separated by a comma and no space (0,0 runs all of them)")
        sys.exit(1)

    cov_type = sys.argv[1]
    cov_file = sys.argv[2]
    data_name = sys.argv[3]
    true_name = sys.argv[4]
    time_name = sys.argv[5]
    foldername = sys.argv[6]
    dirname = os.path.join(bigfolder, foldername)
    first, n_chan = [int(x) for x in sys.argv[7].split(",")]

    if cov_type not in ("nondiag", "diag", "file"):
        print(f"The type of the covariance matrix must be nondiag, diag or file, got {cov_type}")
        sys.exit(1)
    if cov_type == "file" and cov_file == "model":
        print("The type file needs the name of the file of the covariance matrix")
        sys.exit(1)

    print("Output Directory %s" % dirname, flush=True)
    os.makedirs(dirname, exist_ok=True)

    #Read data
    x_grid = np.load(time_name)
    d = np.load(data_name)
    d_true = np.load(true_name)

    if d.shape != d_true.shape:
        print(f"The noised traces {d.shape} and the true traces {d_true.shape} must have the same shape")
        sys.exit(1)
    if d.shape[-1] != len(x_grid):
        print(f"The traces ({d.shape[-1]} bins) and the time bins ({len(x_grid)}) must have the same length")
        sys.exit(1)

    # Select the channels to run
    tot = len(d)
    last = tot if n_chan == 0 else min(first+n_chan, tot)
    if first >= tot:
        print(f"The first channel ({first}) is out of the {tot} channels")
        sys.exit(1)
    d = d[first:last]
    d_true = d_true[first:last]
    suffix = "" if (first == 0 and last == tot) else f"_{first}_{last}"

    # Print time
    middle = time.time()
    elapsed = middle-start
    print(f"Read data: {elapsed}", flush=True)
    fn_time = os.path.join(dirname, f"time{suffix}.txt")
    with open(fn_time, 'w') as f:
        f.write(f"Read data: {elapsed}\n")
    print(f"Running channels {first}-{last-1} out of {tot} with the {cov_type} covariance matrix", flush=True)

    # Make the inverse of the covariance matrix
    add_timestamp(start, fn_time, "Inverting Covariance Matrix")
    noise_cov_inv, noise_std_inv = get_cov(cov_type, cov_file, len(x_grid), fn_time)
    add_timestamp(start, fn_time, "Finished inverting matrix")

    # Reconstruct every channel
    add_timestamp(start, fn_time, "Reconstructing Channels")
    means, stds, rmses, norm_rmses, n, key = reconstruct(d, d_true, x_grid, noise_cov_inv, noise_std_inv, fn_time, key)
    add_timestamp(start, fn_time, "Finished reconstructing channels")

    np.save(os.path.join(dirname, f"means{suffix}.npy"), means)
    np.save(os.path.join(dirname, f"stds{suffix}.npy"), stds)
    np.save(os.path.join(dirname, f"rmses{suffix}.npy"), rmses)
    np.save(os.path.join(dirname, f"norm_rmses{suffix}.npy"), norm_rmses)

    # Write down the settings and the results
    write_details(dirname, suffix, cov_type, cov_file, data_name, true_name, time_name, first, last, tot,
                  (x_grid[1]-x_grid[0])*1e-9, n, time.time()-start)

    # Print time
    add_timestamp(start, fn_time, "Finished Everything")
