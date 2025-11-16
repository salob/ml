"""
Generate convergence plots from training history CSV files.
Shows train/validation loss and accuracy across epochs for model comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11

# Find all history files
logs_dir = Path("logs")
cnn_files = sorted(glob.glob(str(logs_dir / "imdb_cnn_history_seed*.csv")))
transformer_files = sorted(glob.glob(str(logs_dir / "imdb_transformer_history_seed*.csv")))

print(f"Found {len(cnn_files)} CNN history files")
print(f"Found {len(transformer_files)} Transformer history files")

# Load data
cnn_data = []
for file in cnn_files:
    df = pd.read_csv(file)
    seed = file.split('seed')[1].replace('.csv', '')
    df['seed'] = int(seed)
    df['model'] = 'CNN'
    cnn_data.append(df)

transformer_data = []
for file in transformer_files:
    df = pd.read_csv(file)
    seed = file.split('seed')[1].replace('.csv', '')
    df['seed'] = int(seed)
    df['model'] = 'Transformer'
    transformer_data.append(df)

# Combine data
all_data = cnn_data + transformer_data

if not all_data:
    print("No training history files found!")
    print("Make sure you've run the models with the updated code.")
    exit(1)

all_df = pd.concat(all_data, ignore_index=True)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Training Convergence Analysis', fontsize=16, fontweight='bold')

# Colors for models
colors = {
    'CNN': '#ed64a6',  # Pink
    'Transformer': '#667eea'  # Purple
}

# Plot 1: Training Loss over Epochs
ax1 = axes[0, 0]
for model in ['CNN', 'Transformer']:
    model_data = all_df[all_df['model'] == model]
    if len(model_data) > 0:
        # Group by epoch and calculate mean and std
        grouped = model_data.groupby('epoch')['train_loss'].agg(['mean', 'std'])
        epochs = grouped.index
        
        ax1.plot(epochs, grouped['mean'], label=model, color=colors[model], linewidth=2)
        if len(model_data['seed'].unique()) > 1:
            ax1.fill_between(epochs, 
                            grouped['mean'] - grouped['std'], 
                            grouped['mean'] + grouped['std'], 
                            alpha=0.2, color=colors[model])

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Convergence', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Loss over Epochs
ax2 = axes[0, 1]
for model in ['CNN', 'Transformer']:
    model_data = all_df[all_df['model'] == model]
    if len(model_data) > 0:
        grouped = model_data.groupby('epoch')['val_loss'].agg(['mean', 'std'])
        epochs = grouped.index
        
        ax2.plot(epochs, grouped['mean'], label=model, color=colors[model], linewidth=2)
        if len(model_data['seed'].unique()) > 1:
            ax2.fill_between(epochs, 
                            grouped['mean'] - grouped['std'], 
                            grouped['mean'] + grouped['std'], 
                            alpha=0.2, color=colors[model])

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss Convergence', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Validation Accuracy over Epochs
ax3 = axes[1, 0]
for model in ['CNN', 'Transformer']:
    model_data = all_df[all_df['model'] == model]
    if len(model_data) > 0:
        grouped = model_data.groupby('epoch')['val_accuracy'].agg(['mean', 'std'])
        epochs = grouped.index
        
        ax3.plot(epochs, grouped['mean'], label=model, color=colors[model], linewidth=2)
        if len(model_data['seed'].unique()) > 1:
            ax3.fill_between(epochs, 
                            grouped['mean'] - grouped['std'], 
                            grouped['mean'] + grouped['std'], 
                            alpha=0.2, color=colors[model])

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Accuracy')
ax3.set_title('Validation Accuracy Progression', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Validation F1 Score over Epochs
ax4 = axes[1, 1]
for model in ['CNN', 'Transformer']:
    model_data = all_df[all_df['model'] == model]
    if len(model_data) > 0:
        grouped = model_data.groupby('epoch')['val_f1'].agg(['mean', 'std'])
        epochs = grouped.index
        
        ax4.plot(epochs, grouped['mean'], label=model, color=colors[model], linewidth=2)
        if len(model_data['seed'].unique()) > 1:
            ax4.fill_between(epochs, 
                            grouped['mean'] - grouped['std'], 
                            grouped['mean'] + grouped['std'], 
                            alpha=0.2, color=colors[model])

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Validation F1 Score')
ax4.set_title('Validation F1 Score Progression', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/convergence_analysis.png', dpi=300, bbox_inches='tight')
print("Convergence plot saved to reports/convergence_analysis.png")

# Create a second figure for train vs validation loss comparison (overfitting detection)
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Training vs Validation Loss - Overfitting Detection', fontsize=16, fontweight='bold')

for idx, model in enumerate(['CNN', 'Transformer']):
    ax = axes2[idx]
    model_data = all_df[all_df['model'] == model]
    
    if len(model_data) > 0:
        train_grouped = model_data.groupby('epoch')['train_loss'].agg(['mean', 'std'])
        val_grouped = model_data.groupby('epoch')['val_loss'].agg(['mean', 'std'])
        epochs = train_grouped.index
        
        # Plot training loss
        ax.plot(epochs, train_grouped['mean'], label='Training Loss', 
                color=colors[model], linewidth=2, linestyle='-')
        if len(model_data['seed'].unique()) > 1:
            ax.fill_between(epochs, 
                           train_grouped['mean'] - train_grouped['std'], 
                           train_grouped['mean'] + train_grouped['std'], 
                           alpha=0.15, color=colors[model])
        
        # Plot validation loss
        ax.plot(epochs, val_grouped['mean'], label='Validation Loss', 
                color=colors[model], linewidth=2, linestyle='--', alpha=0.8)
        if len(model_data['seed'].unique()) > 1:
            ax.fill_between(epochs, 
                           val_grouped['mean'] - val_grouped['std'], 
                           val_grouped['mean'] + val_grouped['std'], 
                           alpha=0.15, color=colors[model])
        
        # Mark early stopping point (where validation loss starts increasing)
        val_mean = val_grouped['mean'].values
        if len(val_mean) > 1:
            min_idx = np.argmin(val_mean)
            ax.axvline(x=epochs[min_idx], color='red', linestyle=':', 
                      label=f'Best Val Loss (Epoch {epochs[min_idx]})', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model} - Train vs Validation Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("Overfitting analysis plot saved to reports/overfitting_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("CONVERGENCE SUMMARY")
print("="*60)

for model in ['CNN', 'Transformer']:
    model_data = all_df[all_df['model'] == model]
    if len(model_data) > 0:
        print(f"\n{model}:")
        print(f"  Seeds analyzed: {sorted(model_data['seed'].unique())}")
        print(f"  Average epochs until convergence: {model_data.groupby('seed')['epoch'].max().mean():.1f}")
        
        final_metrics = model_data.groupby('seed').last()
        print(f"  Final validation accuracy: {final_metrics['val_accuracy'].mean():.4f} ± {final_metrics['val_accuracy'].std():.4f}")
        print(f"  Final validation F1: {final_metrics['val_f1'].mean():.4f} ± {final_metrics['val_f1'].std():.4f}")
        print(f"  Final validation loss: {final_metrics['val_loss'].mean():.4f} ± {final_metrics['val_loss'].std():.4f}")

print("\n" + "="*60)
print("Plots generated successfully!")
print("="*60)
