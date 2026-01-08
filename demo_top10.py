#!/usr/bin/env python3
"""Quick demo to show top 10 topics without filter"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
file_path = "Untitled_Notebook_2025_12_24_14_34_56.csv"
df = pd.read_csv(file_path)
print(f"✓ Loaded {len(df):,} rows")

# Aggregate by topic
aggregated = df.groupby('topic', as_index=False).agg({
    'raw_conversations': 'sum',
    'raw_messages': 'sum',
    'raw_users': 'sum'
}).sort_values('raw_conversations', ascending=False)

# Get top 10
top_10 = aggregated.head(10)

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(range(len(top_10)), top_10['raw_conversations'], 
               color=sns.color_palette("viridis", len(top_10)))

ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['topic'], fontsize=10)
ax.set_xlabel('Raw Conversations', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Topics by Raw Conversations', fontsize=14, fontweight='bold', pad=20)

# Add value labels
for i, (idx, row) in enumerate(top_10.iterrows()):
    value = row['raw_conversations']
    ax.text(value, i, f' {value:,}', va='center', fontsize=9, fontweight='bold')

ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save plot (use non-interactive backend)
plt.ioff()  # Turn off interactive mode
output_file = "top10_topics.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to {output_file}")

# Print top 10
print("\nTop 10 Topics by Raw Conversations:")
print("-" * 80)
for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. {row['topic']:50s} - {row['raw_conversations']:>8,} conversations")

plt.close()
