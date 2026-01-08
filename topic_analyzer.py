#!/usr/bin/env python3
"""
Topic Analysis Tool
Reads CSV file, allows filtering by level1 and level2, and visualizes top 10 topics by raw_conversations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(file_path):
    """Load the CSV file into a pandas DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded {len(df):,} rows from {file_path}")
        return df
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)


def get_unique_values(df, column):
    """Get unique values for a column"""
    return sorted(df[column].dropna().unique().tolist())


def filter_data(df, level1=None, level2=None):
    """Filter DataFrame by level1 and/or level2"""
    filtered_df = df.copy()
    
    if level1:
        filtered_df = filtered_df[filtered_df['level1'] == level1]
        print(f"✓ Filtered by level1: {level1}")
    
    if level2:
        filtered_df = filtered_df[filtered_df['level2'] == level2]
        print(f"✓ Filtered by level2: {level2}")
    
    return filtered_df


def aggregate_by_topic(df):
    """Aggregate data by topic, summing raw_conversations"""
    aggregated = df.groupby('topic', as_index=False).agg({
        'raw_conversations': 'sum',
        'raw_messages': 'sum',
        'raw_users': 'sum'
    }).sort_values('raw_conversations', ascending=False)
    
    return aggregated


def plot_top_topics(df, top_n=10, title_suffix=""):
    """Create a bar plot of top N topics by raw_conversations"""
    top_topics = df.head(top_n)
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_topics)), top_topics['raw_conversations'], 
                   color=sns.color_palette("viridis", len(top_topics)))
    
    # Customize the plot
    ax.set_yticks(range(len(top_topics)))
    ax.set_yticklabels(top_topics['topic'], fontsize=10)
    ax.set_xlabel('Raw Conversations', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Topics by Raw Conversations{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_topics.iterrows()):
        value = row['raw_conversations']
        ax.text(value, i, f' {value:,}', va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def interactive_menu(df):
    """Interactive menu for filtering and visualization"""
    while True:
        print("\n" + "="*60)
        print("TOPIC ANALYSIS TOOL")
        print("="*60)
        print("\nOptions:")
        print("1. Show all unique level1 categories")
        print("2. Show all unique level2 categories")
        print("3. Filter by level1 and visualize")
        print("4. Filter by level2 and visualize")
        print("5. Filter by both level1 and level2 and visualize")
        print("6. Show top 10 topics (no filter)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            level1_values = get_unique_values(df, 'level1')
            print(f"\nFound {len(level1_values)} unique level1 categories:")
            for i, val in enumerate(level1_values, 1):
                print(f"  {i}. {val}")
        
        elif choice == '2':
            level2_values = get_unique_values(df, 'level2')
            print(f"\nFound {len(level2_values)} unique level2 categories:")
            # Show first 50, then ask if user wants to see more
            for i, val in enumerate(level2_values[:50], 1):
                print(f"  {i}. {val}")
            if len(level2_values) > 50:
                print(f"\n... and {len(level2_values) - 50} more")
                show_more = input("Show all? (y/n): ").strip().lower()
                if show_more == 'y':
                    for i, val in enumerate(level2_values[50:], 51):
                        print(f"  {i}. {val}")
        
        elif choice == '3':
            level1_values = get_unique_values(df, 'level1')
            print("\nAvailable level1 categories:")
            for i, val in enumerate(level1_values, 1):
                print(f"  {i}. {val}")
            
            try:
                idx = int(input("\nEnter the number of level1 category: ")) - 1
                if 0 <= idx < len(level1_values):
                    selected_level1 = level1_values[idx]
                    filtered_df = filter_data(df, level1=selected_level1)
                    aggregated = aggregate_by_topic(filtered_df)
                    
                    if len(aggregated) > 0:
                        print(f"\nFound {len(aggregated)} unique topics")
                        title = f" (Filtered by level1: {selected_level1})"
                        fig = plot_top_topics(aggregated, top_n=10, title_suffix=title)
                        plt.show()
                    else:
                        print("No data found with this filter.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '4':
            level2_values = get_unique_values(df, 'level2')
            print("\nAvailable level2 categories (showing first 50):")
            for i, val in enumerate(level2_values[:50], 1):
                print(f"  {i}. {val}")
            if len(level2_values) > 50:
                print(f"\n... and {len(level2_values) - 50} more")
            
            search_term = input("\nEnter level2 category name (or part of it) to search: ").strip()
            matching = [v for v in level2_values if search_term.lower() in v.lower()]
            
            if matching:
                print(f"\nFound {len(matching)} matching categories:")
                for i, val in enumerate(matching, 1):
                    print(f"  {i}. {val}")
                
                try:
                    idx = int(input("\nEnter the number of level2 category: ")) - 1
                    if 0 <= idx < len(matching):
                        selected_level2 = matching[idx]
                        filtered_df = filter_data(df, level2=selected_level2)
                        aggregated = aggregate_by_topic(filtered_df)
                        
                        if len(aggregated) > 0:
                            print(f"\nFound {len(aggregated)} unique topics")
                            title = f" (Filtered by level2: {selected_level2})"
                            fig = plot_top_topics(aggregated, top_n=10, title_suffix=title)
                            plt.show()
                        else:
                            print("No data found with this filter.")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print("No matching categories found.")
        
        elif choice == '5':
            level1_values = get_unique_values(df, 'level1')
            print("\nAvailable level1 categories:")
            for i, val in enumerate(level1_values, 1):
                print(f"  {i}. {val}")
            
            try:
                idx1 = int(input("\nEnter the number of level1 category: ")) - 1
                if 0 <= idx1 < len(level1_values):
                    selected_level1 = level1_values[idx1]
                    
                    # Get level2 values for this level1
                    level2_for_level1 = sorted(df[df['level1'] == selected_level1]['level2'].unique().tolist())
                    print(f"\nAvailable level2 categories for '{selected_level1}':")
                    for i, val in enumerate(level2_for_level1, 1):
                        print(f"  {i}. {val}")
                    
                    idx2 = int(input("\nEnter the number of level2 category: ")) - 1
                    if 0 <= idx2 < len(level2_for_level1):
                        selected_level2 = level2_for_level1[idx2]
                        filtered_df = filter_data(df, level1=selected_level1, level2=selected_level2)
                        aggregated = aggregate_by_topic(filtered_df)
                        
                        if len(aggregated) > 0:
                            print(f"\nFound {len(aggregated)} unique topics")
                            title = f" (Filtered by level1: {selected_level1}, level2: {selected_level2})"
                            fig = plot_top_topics(aggregated, top_n=10, title_suffix=title)
                            plt.show()
                        else:
                            print("No data found with this filter.")
                    else:
                        print("Invalid level2 selection.")
                else:
                    print("Invalid level1 selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '6':
            aggregated = aggregate_by_topic(df)
            print(f"\nShowing top 10 topics from all {len(aggregated)} unique topics")
            fig = plot_top_topics(aggregated, top_n=10)
            plt.show()
        
        elif choice == '7':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1-7.")


def main():
    """Main function"""
    # Default file path
    default_file = Path(__file__).parent / "Untitled_Notebook_2025_12_24_14_34_56.csv"
    
    # Allow command line argument for file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = str(default_file)
    
    # Load data
    print(f"Loading data from: {file_path}")
    df = load_data(file_path)
    
    # Show basic info
    print(f"\nDataset Info:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique topics: {df['topic'].nunique():,}")
    print(f"  Unique level1 categories: {df['level1'].nunique()}")
    print(f"  Unique level2 categories: {df['level2'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Start interactive menu
    interactive_menu(df)


if __name__ == "__main__":
    main()
