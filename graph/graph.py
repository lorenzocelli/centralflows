import os
import h5py
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot graphs")
parser.add_argument(
    "-e", "--exp", nargs="+", required=True, help="List of experiment directories"
)
parser.add_argument(
    "--max-iter", type=int, default=-1, help="Maximum number of iterations to plot"
)
parser.add_argument(
    "--smoothing",
    type=int,
    default=1,
    help="Smoothing window size for Hessian eigenvalues",
)
parser.add_argument("--raw", action="store_true", help="Plot raw Hessian eigenvalues")
parser.add_argument("--eig-max", type=float, default=-1, help="Maximum y-axis for eigenvalues")

args = parser.parse_args()

plt.style.use("ggplot")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["CMU Sans Serif"]
plt.rcParams["font.weight"] = "medium"
plt.rcParams["axes.unicode_minus"] = False

COLUMN_WIDTH = 3.287

if args.raw:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(4 * 2.5, COLUMN_WIDTH))
else:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3 * 2.5, COLUMN_WIDTH))

lines = []

for exp_dir in args.exp:
    data = os.path.join(exp_dir, "data.hdf5")
    exp = os.path.basename(os.path.normpath(exp_dir))

    with h5py.File(data, "r", libver="latest", swmr=True) as df:
        discrete = df["discrete"]
        loss_train = discrete["train_loss"]
        # loss_test = discrete["test_loss"]
        grad_sq = discrete["grad_norm_sq"]
        step = df["step"][:]

        i_min = 0
        i_max = args.max_iter if args.max_iter > 0 else len(step)

        ax1.plot(step[i_min:i_max], loss_train[i_min:i_max], label=f"Train {exp}")
        # ax1.plot(step[i_min:i_max], loss_test[i_min:i_max], label=f"Test {exp}")
        ax2.plot(step[i_min:i_max], grad_sq[i_min:i_max], label=exp)

        if "effective_hessian_eigs" in discrete:
            heigs = discrete["effective_hessian_eigs"]
            raw_heigs = discrete["hessian_eigs"]

            for i in range(heigs.shape[1]):
                weights = np.ones(args.smoothing) / args.smoothing
                heigs_avg = np.convolve(heigs[:, i], weights, mode="same")[i_min:i_max]
                (line,) = ax3.plot(step[i_min:i_max], heigs[i_min:i_max, i], alpha=0.25)
                lines.append(
                    (
                        step[i_min:i_max],
                        heigs_avg,
                        line.get_color(),
                        f"Eig {i + 1} {exp}",
                    )
                )

            if args.raw:
                ax4.plot(
                    step[i_min:i_max],
                    raw_heigs[i_min:i_max, i],
                    label=f"Raw Eig {i + 1} {exp}",
                )

for i, (x, y, color, label) in enumerate(lines):
    # Ensure smoothed lines are plotted on top of the semitransparent ones
    ax3.plot(x, y, color=color, label=label)

if args.eig_max > 0:
    ax3.set_ylim(bottom=0, top=args.eig_max)

ax3.axhline(2, color="black", linewidth=0.8, linestyle="--")

ax3.axvline(600, color="#188487", linewidth=0.8, linestyle="--")
ax3.axvline(1000, color="#188487", linewidth=0.8, linestyle="--")
ax3.annotate('unfreeze', xy=(500, 5), color='#188487', horizontalalignment='right', verticalalignment='center', rotation=90)
ax3.annotate('freeze', xy=(1000, 5), color='#188487', horizontalalignment='right', verticalalignment='center', rotation=90)

# ax1.legend()
# ax2.legend()
# ax3.legend()

ax1.set_title("Training Loss", fontsize=11, loc="left")
ax2.set_title("Gradient Norm Squared", fontsize=11, loc="left")
ax3.set_title("Effective Hessian Eigenvalues", fontsize=11, loc="left")

if args.raw:
    # ax4.legend()
    ax4.set_title("Raw Hessian Eigenvalues", fontsize=11, loc="left")

# Align title left
fig.suptitle("Muon with Frozen/Unfrozen Preconditioner lr=1e-4", fontsize=11, fontweight="bold", y=0.94, x=0.043, ha="left")

fig.tight_layout()
plt.savefig(f"{'_'.join(args.exp)}.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
