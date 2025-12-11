import argparse
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_combined(experiment_dir, save_path=None):
    data_path = os.path.join(experiment_dir, "data.hdf5")
    config_path = os.path.join(experiment_dir, "config.json")
    
    if not os.path.exists(data_path):
        print(f"Error: No data.hdf5 found in {experiment_dir}")
        return

    # Load config for titles/thresholds
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print(f"Loading data from {data_path}...")
    
    with h5py.File(data_path, "r") as f:
        if 'discrete' not in f:
            print("Error: Could not find 'discrete' process data.")
            return
            
        steps = f['step'][:]
        
        # --- LOAD LOSS DATA ---
        loss_discrete = f['discrete']['train_loss'][:]
        
        loss_central = None
        if 'central' in f and 'predicted_loss' in f['central']:
            loss_central = f['central']['predicted_loss'][:]
            
        loss_stable = None
        if 'stable' in f and 'train_loss' in f['stable']:
            loss_stable = f['stable']['train_loss'][:]

        # --- LOAD EIGENVALUE DATA ---
        discrete_grp = f['discrete']
        if 'effective_hessian_eigs' in discrete_grp:
            eigs = discrete_grp['effective_hessian_eigs'][:]
            metric_name = "Effective Sharpness"
            threshold = 2.0 
        elif 'hessian_eigs' in discrete_grp:
            eigs = discrete_grp['hessian_eigs'][:]
            metric_name = "Hessian Eigenvalues"
            lr = config.get('opt', {}).get('lr', None)
            threshold = (2.0 / lr) if lr else None
        else:
            eigs = None

        # --- PREPROCESSING ---
        # We use the discrete loss to determine the valid range (mask NaNs)
        mask = ~np.isnan(loss_discrete)
        steps = steps[mask]
        loss_discrete = loss_discrete[mask]
        
        if loss_central is not None: loss_central = loss_central[mask]
        if loss_stable is not None: loss_stable = loss_stable[mask]
        if eigs is not None: eigs = eigs[mask]

        if len(steps) == 0:
            print("No valid data found.")
            return

        # --- PLOTTING ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=120, sharex=True)
        
        # SUBPLOT 1: LOSS
        ax1.plot(steps, loss_discrete, lw=1.5, label='Discrete Optimizer', color='#1f77b4')
        if loss_stable is not None:
            ax1.plot(steps, loss_stable, lw=1.5, label='Gradient Flow', color='#d62728', alpha=0.8)
        if loss_central is not None:
            ax1.plot(steps, loss_central, lw=1.5, linestyle='--', color='black', 
                     label='Central Flow Prediction', alpha=0.9)
        
        ax1.set_ylabel('Training Loss', fontsize=11)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.legend(frameon=True, fontsize=9)

        # SUBPLOT 2: SHARPNESS
        if eigs is not None:
            num_eigs_to_plot = min(eigs.shape[1], 5)
            colors = plt.cm.viridis(np.linspace(0, 0.8, num_eigs_to_plot))
            
            for i in range(num_eigs_to_plot):
                ax2.plot(steps, eigs[:, i], lw=1.5, alpha=0.9, label=f'$\\lambda_{i+1}$', color=colors[i])

            if threshold:
                ax2.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, label='Stability Threshold')
                
            ax2.set_ylabel(metric_name, fontsize=11)
            ax2.legend(loc='upper right', frameon=True, fontsize=9, ncol=2)
            
            # Smart Y-limits for sharpness
            y_max = np.nanmax(eigs)
            ax2.set_ylim(0, max(y_max * 1.1, threshold * 1.2 if threshold else y_max))
        else:
            ax2.text(0.5, 0.5, "No Eigenvalue Data Available", 
                     ha='center', va='center', transform=ax2.transAxes)

        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('Step', fontsize=11)

        # Main Title
        arch_name = config.get("arch", {}).get("class", "Model")
        data_name = config.get("data", {}).get("class", "Dataset")
        opt_name = config.get("opt", {}).get("class", "Optimizer")
        fig.suptitle(f'{opt_name} on {arch_name} / {data_name}\nLoss and Sharpness Dynamics', fontsize=14, y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88) # Make room for suptitle

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss and sharpness combined.")
    parser.add_argument("dir", type=str, help="Path to experiment directory")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot image")
    
    args = parser.parse_args()
    plot_combined(args.dir, args.save)