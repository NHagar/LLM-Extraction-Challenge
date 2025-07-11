import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('summary_all_json.csv')

# Filter to only include files ending in _prompt2.json
df = df[df['JSON Filename'].str.endswith('_prompt2.json')]

print(f"Filtered to {len(df)} models ending in _prompt2.json")

# Extract model names from JSON filenames (remove _prompt2.json and date info)
df['Model'] = df['JSON Filename'].str.replace('_prompt2.json', '').str.replace(r'_\d{4}', '', regex=True).str.replace('_november', '').str.replace('_december', '')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison Across All Metrics', fontsize=16, fontweight='bold')

# 1. Accuracy Distribution (Top Left)
axes[0,0].hist(df['Accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(df['Accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {df["Accuracy"].mean():.3f}')
axes[0,0].set_title('Distribution of Accuracy Scores')
axes[0,0].set_xlabel('Accuracy')
axes[0,0].set_ylabel('Number of Models')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Correlation Matrix (Top Right)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
corr_matrix = df[metrics].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[0,1])
axes[0,1].set_title('Correlation Between Metrics')

# 3. Box Plot of All Metrics (Bottom Left)
df_melted = df.melt(id_vars=['Model'], value_vars=metrics, 
                    var_name='Metric', value_name='Score')
sns.boxplot(data=df_melted, x='Metric', y='Score', ax=axes[1,0])
axes[1,0].set_title('Distribution of All Performance Metrics')
axes[1,0].set_ylabel('Score')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Scatter Plot: Precision vs Recall (Bottom Right)
scatter = axes[1,1].scatter(df['Precision'], df['Recall'], 
                           c=df['F1 Score'], cmap='viridis', 
                           s=60, alpha=0.7)
axes[1,1].set_xlabel('Precision')
axes[1,1].set_ylabel('Recall')
axes[1,1].set_title('Precision vs Recall (colored by F1 Score)')
plt.colorbar(scatter, ax=axes[1,1], label='F1 Score')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional visualization: Top 10 Models by F1 Score
plt.figure(figsize=(14, 8))
top_models = df.nlargest(10, 'F1 Score')

x = np.arange(len(top_models))
width = 0.2

plt.bar(x - width*1.5, top_models['Accuracy'], width, label='Accuracy', alpha=0.8)
plt.bar(x - width/2, top_models['Precision'], width, label='Precision', alpha=0.8)
plt.bar(x + width/2, top_models['Recall'], width, label='Recall', alpha=0.8)
plt.bar(x + width*1.5, top_models['F1 Score'], width, label='F1 Score', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Top 10 Models by F1 Score - All Metrics Comparison')
plt.xticks(x, top_models['Model'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('top_10_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Individual metric distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Individual Metric Distributions', fontsize=16, fontweight='bold')

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    ax.hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(df[metric].mean(), color='red', linestyle='--', 
               label=f'Mean: {df[metric].mean():.3f}')
    ax.set_title(f'{metric} Distribution')
    ax.set_xlabel(metric)
    ax.set_ylabel('Number of Models')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('individual_metric_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Precision vs Recall scatter plot (standalone)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Precision'], df['Recall'], 
                     c=df['F1 Score'], cmap='viridis', 
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
plt.xlabel('Precision', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('Precision vs Recall (colored by F1 Score)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='F1 Score')
plt.grid(True, alpha=0.3)

# Add diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Balance')
plt.legend()
plt.tight_layout()
plt.savefig('precision_recall_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Model ranking by F1 Score
plt.figure(figsize=(14, 10))
df_sorted = df.sort_values('F1 Score', ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))

bars = plt.barh(range(len(df_sorted)), df_sorted['F1 Score'], color=colors, alpha=0.8)
plt.yticks(range(len(df_sorted)), df_sorted['Model'], fontsize=8)
plt.xlabel('F1 Score', fontsize=12)
plt.title('All Models Ranked by F1 Score', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, df_sorted['F1 Score'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', va='center', fontsize=6)

plt.tight_layout()
plt.savefig('all_models_ranking.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("Performance Summary Statistics:")
print(df[metrics].describe())

print("\nTop 5 Models by F1 Score:")
print(df.nlargest(5, 'F1 Score')[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])

print("\nBottom 5 Models by F1 Score:")
print(df.nsmallest(5, 'F1 Score')[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])