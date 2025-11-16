#!/usr/bin/env python3
"""
DEPRECATED - NICER HTML, JAVASCRIPT HTML REPORT AVAILABLE BY RUNNING generate_emissions_report.py

Model Comparison Report Generator

This script reads metrics CSV files from the three IMDB models (Logistic Regression, CNN, Transformer)
and generates a comprehensive PDF report with visualizations comparing their performance and characteristics.

Usage: python generate_comparison_report.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics_data():
    """Load and combine metrics from all three model CSV files"""
    files = {
        'LogisticRegression': 'logs/imdb_logreg_metrics.csv',
        'CNN': 'logs/imdb_cnn_metrics.csv', 
        'TinyTransformer': 'logs/imdb_transformer_metrics.csv'
    }
    
    all_data = []
    
    for model_name, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if not df.empty:
                df['model_name'] = model_name
                all_data.append(df)
            else:
                print(f"Warning: {filepath} is empty")
        else:
            print(f"Warning: {filepath} not found")
    
    if not all_data:
        raise FileNotFoundError("No metrics files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_summary_stats(df):
    """Calculate mean and std for each model across all runs"""
    numeric_cols = ['val_accuracy', 'val_f1', 'test_accuracy', 'test_f1', 'total_parameters']
    
    summary = df.groupby('model_name')[numeric_cols].agg(['mean', 'std', 'count']).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary.reset_index()

def create_performance_comparison(df, fig, axes):
    """Create performance comparison plots"""
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    accuracy_data = []
    model_names = []
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        accuracy_data.extend(model_data['test_accuracy'].tolist())
        model_names.extend([model] * len(model_data))
    
    accuracy_df = pd.DataFrame({'Model': model_names, 'Test Accuracy': accuracy_data})
    sns.boxplot(data=accuracy_df, x='Model', y='Test Accuracy', ax=ax1)
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: F1 Score Comparison
    ax2 = axes[0, 1]
    f1_data = []
    model_names = []
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        f1_data.extend(model_data['test_f1'].tolist())
        model_names.extend([model] * len(model_data))
    
    f1_df = pd.DataFrame({'Model': model_names, 'Test F1': f1_data})
    sns.boxplot(data=f1_df, x='Model', y='Test F1', ax=ax2)
    ax2.set_title('Test F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Performance vs Complexity
    ax3 = axes[1, 0]
    summary_stats = calculate_summary_stats(df)
    
    # Handle missing parameters (LogReg doesn't have total_parameters)
    param_data = []
    acc_data = []
    model_labels = []
    
    for _, row in summary_stats.iterrows():
        model = row['model_name']
        if 'total_parameters_mean' in row and not pd.isna(row['total_parameters_mean']):
            param_data.append(row['total_parameters_mean'])
        else:
            # Estimate parameters for LogReg (TF-IDF features)
            param_data.append(100000)  # Approximate TF-IDF feature count
        
        acc_data.append(row['test_accuracy_mean'])
        model_labels.append(model)
    
    colors = ['red', 'green', 'blue']
    for i, (model, params, acc) in enumerate(zip(model_labels, param_data, acc_data)):
        ax3.scatter(params, acc, s=100, alpha=0.7, c=colors[i], label=model)
    
    ax3.set_xlabel('Model Parameters (log scale)')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_xscale('log')
    ax3.set_title('Accuracy vs Model Complexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Configuration
    ax4 = axes[1, 1]
    
    # Show key hyperparameters
    config_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model].iloc[0]  # Take first run
        
        if model == 'LogisticRegression':
            config_text = f"C=1.0\nmax_iter=1000\nTF-IDF: (1,2)-grams"
        elif model == 'CNN':
            config_text = f"Epochs: {model_df.get('num_epochs', 'N/A')}\nBatch: {model_df.get('batch_size', 'N/A')}\nEmb: {model_df.get('embedding_dim', 'N/A')}"
        else:  # Transformer
            config_text = f"Epochs: {model_df.get('num_epochs', 'N/A')}\nHeads: {model_df.get('num_heads', 'N/A')}\nLayers: {model_df.get('num_layers', 'N/A')}"
        
        config_data.append((model, config_text))
    
    ax4.axis('off')
    ax4.set_title('Model Configurations')
    
    y_pos = 0.8
    for model, config in config_data:
        ax4.text(0.1, y_pos, f"{model}:", fontweight='bold', fontsize=10)
        ax4.text(0.1, y_pos-0.1, config, fontsize=9)
        y_pos -= 0.3

def create_summary_table(summary_stats):
    """Create a summary statistics table"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key columns for display
    display_cols = ['model_name', 'test_accuracy_mean', 'test_accuracy_std', 
                   'test_f1_mean', 'test_f1_std', 'test_accuracy_count']
    
    display_data = summary_stats[display_cols].copy()
    display_data.columns = ['Model', 'Test Acc (Mean)', 'Test Acc (Std)', 
                           'Test F1 (Mean)', 'Test F1 (Std)', 'Runs']
    
    # Format numbers
    for col in ['Test Acc (Mean)', 'Test Acc (Std)', 'Test F1 (Mean)', 'Test F1 (Std)']:
        display_data[col] = display_data[col].round(4)
    
    table = ax.table(cellText=display_data.values,
                    colLabels=display_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(display_data) + 1):
        for j in range(len(display_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f2f2f2')
    
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    return fig

def generate_pdf_report():
    """Generate the complete PDF report"""
    
    # Load data
    try:
        df = load_metrics_data()
        summary_stats = calculate_summary_stats(df)
        
        print(f"Found data for {len(df)} total runs across {df['model_name'].nunique()} models")
        print(f"Models: {', '.join(df['model_name'].unique())}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"logs/model_comparison_report_{timestamp}.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        
        # Page 1: Title and Summary Table
        fig = create_summary_table(summary_stats)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # Page 2: Performance Comparisons
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('IMDB Model Performance Comparison', fontsize=16, fontweight='bold')
        
        create_performance_comparison(df, fig, axes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # Page 3: Individual Model Analysis (if multiple runs exist)
        models_with_multiple_runs = df['model_name'].value_counts()
        multi_run_models = models_with_multiple_runs[models_with_multiple_runs > 1].index
        
        if len(multi_run_models) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Stability Analysis', fontsize=16, fontweight='bold')
            
            plot_idx = 0
            for i, model in enumerate(multi_run_models):
                if plot_idx < 4:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    model_data = df[df['model_name'] == model]
                    
                    # Plot accuracy across runs
                    ax.plot(range(len(model_data)), model_data['test_accuracy'], 
                           'o-', alpha=0.7, label='Accuracy')
                    ax.plot(range(len(model_data)), model_data['test_f1'], 
                           's-', alpha=0.7, label='F1')
                    
                    ax.set_title(f'{model} - Run-to-Run Variation')
                    ax.set_xlabel('Run Number')
                    ax.set_ylabel('Score')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, 4):
                axes[i // 2, i % 2].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
        
        # Add metadata to PDF
        d = pdf.infodict()
        d['Title'] = 'IMDB Model Comparison Report'
        d['Author'] = 'ML Energy Efficiency Study'
        d['Subject'] = 'Performance comparison of Logistic Regression, CNN, and Transformer models'
        d['Keywords'] = 'Machine Learning, Energy Efficiency, Model Comparison, IMDB'
        d['Creator'] = 'Python matplotlib/seaborn'
        d['Producer'] = 'Model Comparison Script'
    
    print(f"Report saved to: {pdf_filename}")
    print(f"Report contains {len(df)} experimental runs")
    
    # Print summary to console
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    for _, row in summary_stats.iterrows():
        model = row['model_name']
        acc_mean = row['test_accuracy_mean']
        acc_std = row['test_accuracy_std'] 
        f1_mean = row['test_f1_mean']
        f1_std = row['test_f1_std']
        runs = int(row['test_accuracy_count'])
        
        print(f"\n{model}:")
        print(f"  Runs: {runs}")
        print(f"  Test Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Test F1:       {f1_mean:.4f} ± {f1_std:.4f}")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    generate_pdf_report()