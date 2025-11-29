import os
import sys
import h5py
import matplotlib.pyplot as plt

dir_path = sys.argv[1]
file = os.path.join(dir_path, "data.hdf5")

with h5py.File(file, "r", libver="latest", swmr=True) as df:
    heigs = df['discrete']['effective_hessian_eigs']
    loss = df['discrete']['train_loss'][:]
    step = df['step'][:]
    grad_norm = df['discrete']['grad_norm_sq'][:]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(step, loss, label='Train Loss')
    ax2.plot(step, grad_norm, label='Gradient Norm')

    for k in heigs.keys():
        ax3.plot(step, heigs[k][:, 0], label=f'{k} - Eig 1')
        # for i in range(7):
        #     ax3.plot(step, heigs[k][:, i], label=f'{k} - Eig {i+1}')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
