import os
import sys
import h5py
import matplotlib.pyplot as plt

dir_path = sys.argv[1]
file = os.path.join(dir_path, "data.hdf5")

with h5py.File(file, "r", libver="latest", swmr=True) as df:
    discrete = df["discrete"]
    loss = discrete["train_loss"][:]
    step = df["step"][:]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(step, loss, label="Train Loss")

    ax2.plot(step, discrete["grad_norm_sq"][:], label='Gradient Norm')

    if "effective_hessian_eigs" in discrete:
        heigs = discrete["effective_hessian_eigs"]
        for i in range(heigs.shape[1]):
            ax3.plot(step, heigs[:, i], label=f"Eig {i + 1}")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    plt.show()
