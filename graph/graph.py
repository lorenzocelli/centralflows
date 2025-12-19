import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

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

    for i in range(1):
        ax3.plot(step, heigs[:, i], label=f'Eig {i+1}')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    plt.savefig(os.path.join(dir_path, "training_plots.png"))
    plt.show()
    plt.close()

    # ============================================
    # GRAFICO PULITO (senza outlier)
    # ============================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # --- Train Loss: skippa primi 200 step ---
    skip_steps = 200
    if len(step) > skip_steps:
        step_trimmed = step[skip_steps:]
        loss_trimmed = loss[skip_steps:]
        ax1.plot(step_trimmed, loss_trimmed, label='Train Loss', linewidth=1.5)
        ax1.set_title('Train Loss (from step 200)')
    else:
        ax1.plot(step, loss, label='Train Loss', linewidth=1.5)
        ax1.set_title('Train Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Gradient Norm: clippa al 99° percentile ---
    grad_threshold = np.percentile(grad_norm, 99)
    grad_clipped = np.clip(grad_norm, 0, grad_threshold)
    ax2.plot(step, grad_clipped, label='Gradient Norm', linewidth=1.5, color='orange')
    ax2.set_title(f'Gradient Norm (clipped at {grad_threshold:.2f})')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Grad Norm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Eigenvalue 1: clippa al 99° percentile ---
    eig1 = heigs[:, 0]
    eig_threshold = np.percentile(eig1, 99)
    eig_clipped = np.clip(eig1, 0, eig_threshold)
    ax3.plot(step, eig_clipped, label='Eig 1', linewidth=1.5, color='green')
    ax3.set_title(f'Eigenvalue 1 (clipped at {eig_threshold:.0f})')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Eigenvalue')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "training_plots_clean.png"), dpi=150)
    plt.close()
    
    # ============================================
    # STAMPA STATISTICHE FINALI
    # ============================================
    print("\n" + "="*50)
    print("📊 TRAINING STATISTICS")
    print("="*50)
    print(f"Final Loss (step {step[-1]:.0f}):        {loss[-1]:.6f}")
    print(f"Final Grad Norm:              {grad_norm[-1]:.6f}")
    print(f"Final Eig 1:                  {eig1[-1]:.2f}")
    print(f"\nAverage Loss (last 500 steps): {np.mean(loss[-500:]):.6f}")
    print(f"Min Loss:                      {np.min(loss):.6f}")
    print(f"Loss at step 200:              {loss[200] if len(loss) > 200 else 'N/A':.6f}")
    print(f"\nGrad Norm 99th percentile:     {grad_threshold:.2f}")
    print(f"Eig 1 99th percentile:         {eig_threshold:.0f}")
    print("="*50)
    
    print(f"\n✅ Plots saved:")
    print(f"   - training_plots.png (original)")
    print(f"   - training_plots_clean.png (without outliers)")
