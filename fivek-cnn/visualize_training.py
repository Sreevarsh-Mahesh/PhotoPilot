#!/usr/bin/env python3
"""
Training Visualization Script

Generates graphs and tables for training analysis including:
- Loss curves (training vs validation)
- Accuracy curves by task
- Confusion matrices
- Learning rate schedule
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_training_history(csv_path='checkpoints/training_history.csv'):
    """Load training history from CSV."""
    df = pd.read_csv(csv_path)
    
    # Parse dictionary strings
    df['train_acc_dict'] = df['train_acc'].apply(ast.literal_eval)
    df['val_acc_dict'] = df['val_acc'].apply(ast.literal_eval)
    
    # Extract individual accuracies
    df['train_acc_ap'] = df['train_acc_dict'].apply(lambda x: x['aperture'])
    df['train_acc_iso'] = df['train_acc_dict'].apply(lambda x: x['iso'])
    df['train_acc_shut'] = df['train_acc_dict'].apply(lambda x: x['shutter'])
    
    df['val_acc_ap'] = df['val_acc_dict'].apply(lambda x: x['aperture'])
    df['val_acc_iso'] = df['val_acc_dict'].apply(lambda x: x['iso'])
    df['val_acc_shut'] = df['val_acc_dict'].apply(lambda x: x['shutter'])
    
    return df

def plot_loss_curves(df, save_path='training_loss_curves.png'):
    """Plot training vs validation loss."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    
    # Find best epoch
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_loss = df['val_loss'].min()
    
    # Plot lines
    ax.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    ax.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    
    # Highlight best model
    ax.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label=f'Best Model (Epoch {best_epoch})')
    ax.plot(best_epoch, best_loss, 'g*', markersize=20, label=f'Best Loss: {best_loss:.4f}')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss curves to {save_path}")
    plt.close()

def plot_accuracy_curves(df, save_path='training_accuracy_curves.png'):
    """Plot accuracy curves for each task."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = df['epoch'].values
    
    # Training accuracy
    ax1.plot(epochs, df['train_acc_ap']*100, 'b-o', label='Aperture', linewidth=2, markersize=8)
    ax1.plot(epochs, df['train_acc_iso']*100, 'g-s', label='ISO', linewidth=2, markersize=8)
    ax1.plot(epochs, df['train_acc_shut']*100, 'r-^', label='Shutter', linewidth=2, markersize=8)
    ax1.axhline(y=33.3, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Random (33.3%)')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Accuracy by Task', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Validation accuracy
    ax2.plot(epochs, df['val_acc_ap']*100, 'b-o', label='Aperture', linewidth=2, markersize=8)
    ax2.plot(epochs, df['val_acc_iso']*100, 'g-s', label='ISO', linewidth=2, markersize=8)
    ax2.plot(epochs, df['val_acc_shut']*100, 'r-^', label='Shutter', linewidth=2, markersize=8)
    ax2.axhline(y=33.3, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Random (33.3%)')
    
    # Highlight best epoch
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    ax2.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, alpha=0.5, label=f'Best Epoch ({best_epoch})')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy by Task', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy curves to {save_path}")
    plt.close()

def plot_loss_components(df, save_path='loss_components.png'):
    """Plot individual loss components (approximate)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = df['epoch'].values
    total_loss = df['val_loss'].values
    
    # Approximate individual losses (assuming equal weighting)
    # In reality, these would be logged separately
    aperture_loss = total_loss / 3
    iso_loss = total_loss / 3
    shutter_loss = total_loss / 3
    
    ax.plot(epochs, aperture_loss, 'b-o', label='Aperture Loss', linewidth=2, markersize=8)
    ax.plot(epochs, iso_loss, 'g-s', label='ISO Loss', linewidth=2, markersize=8)
    ax.plot(epochs, shutter_loss, 'r-^', label='Shutter Loss', linewidth=2, markersize=8)
    ax.plot(epochs, total_loss, 'k--', label='Total Loss', linewidth=3, markersize=10)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss Components by Task (Approximate)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss components to {save_path}")
    plt.close()

def create_metrics_table(df, save_path='training_metrics_table.txt'):
    """Create a formatted text table of training metrics."""
    with open(save_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("TRAINING METRICS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Epoch-by-Epoch Metrics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<30} {'Val Acc':<30}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in df.iterrows():
            train_acc_str = f"Ap:{row['train_acc_ap']:.1%} ISO:{row['train_acc_iso']:.1%} Sh:{row['train_acc_shut']:.1%}"
            val_acc_str = f"Ap:{row['val_acc_ap']:.1%} ISO:{row['val_acc_iso']:.1%} Sh:{row['val_acc_shut']:.1%}"
            f.write(f"{int(row['epoch']):<8} {row['train_loss']:<12.4f} {row['val_loss']:<12.4f} "
                   f"{train_acc_str:<30} {val_acc_str:<30}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("BEST MODEL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        best_row = df.loc[df['val_loss'].idxmin()]
        f.write(f"Best Epoch: {int(best_row['epoch'])}\n")
        f.write(f"Best Validation Loss: {best_row['val_loss']:.4f}\n")
        f.write(f"Training Loss at Best: {best_row['train_loss']:.4f}\n\n")
        
        f.write("Validation Accuracies at Best Epoch:\n")
        f.write(f"  Aperture: {best_row['val_acc_ap']:.1%}\n")
        f.write(f"  ISO: {best_row['val_acc_iso']:.1%}\n")
        f.write(f"  Shutter: {best_row['val_acc_shut']:.1%}\n")
        f.write(f"  Average: {(best_row['val_acc_ap'] + best_row['val_acc_iso'] + best_row['val_acc_shut'])/3:.1%}\n")
    
    print(f"Saved metrics table to {save_path}")

def plot_comparison_chart(df, save_path='training_comparison.png'):
    """Create a comprehensive comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = df['epoch'].values
    
    # 1. Loss comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], 'b-o', label='Training', linewidth=2, markersize=8)
    ax1.plot(epochs, df['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=8)
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss: Training vs Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation accuracy by task
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['val_acc_ap']*100, 'b-o', label='Aperture', linewidth=2, markersize=8)
    ax2.plot(epochs, df['val_acc_iso']*100, 'g-s', label='ISO', linewidth=2, markersize=8)
    ax2.plot(epochs, df['val_acc_shut']*100, 'r-^', label='Shutter', linewidth=2, markersize=8)
    ax2.axhline(y=33.3, color='k', linestyle='--', alpha=0.5, label='Random')
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy by Task')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. Training accuracy by task
    ax3 = axes[1, 0]
    ax3.plot(epochs, df['train_acc_ap']*100, 'b-o', label='Aperture', linewidth=2, markersize=8)
    ax3.plot(epochs, df['train_acc_iso']*100, 'g-s', label='ISO', linewidth=2, markersize=8)
    ax3.plot(epochs, df['train_acc_shut']*100, 'r-^', label='Shutter', linewidth=2, markersize=8)
    ax3.axhline(y=33.3, color='k', linestyle='--', alpha=0.5, label='Random')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Training Accuracy by Task')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # 4. Loss improvement
    ax4 = axes[1, 1]
    loss_improvement = [(df['val_loss'].iloc[0] - val) / df['val_loss'].iloc[0] * 100 
                       for val in df['val_loss']]
    ax4.bar(epochs, loss_improvement, color=['blue', 'green', 'orange'], alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Improvement (%)')
    ax4.set_title('Validation Loss Improvement from Epoch 1')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to {save_path}")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    print("Loading training history...")
    df = load_training_history()
    
    print(f"Loaded {len(df)} epochs of training data")
    print("\nGenerating visualizations...")
    
    # Create output directory
    output_dir = Path('training_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    plot_loss_curves(df, output_dir / 'loss_curves.png')
    plot_accuracy_curves(df, output_dir / 'accuracy_curves.png')
    plot_loss_components(df, output_dir / 'loss_components.png')
    plot_comparison_chart(df, output_dir / 'comparison_chart.png')
    create_metrics_table(df, output_dir / 'metrics_table.txt')
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()


