import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Set style
sns.set_style("white")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Create clearer variance decomposition data
def create_variance_decomposition():
    # Create a dataframe with decomposition data (from the VAR model output)
    steps = np.arange(1, 11)
    data = {
        'Step': [f'Step {i}' for i in steps],
        'All Providers Combined': [1.0, 0.916, 0.773, 0.686, 0.639, 0.542, 0.505, 0.476, 0.443, 0.432],
        'Baktelekom': [0.0, 0.003, 0.004, 0.003, 0.007, 0.008, 0.008, 0.014, 0.027, 0.030],
        'Citynet': [0.0, 0.002, 0.012, 0.016, 0.019, 0.024, 0.027, 0.028, 0.028, 0.027],
        'Aztelekom': [0.0, 0.007, 0.010, 0.017, 0.024, 0.039, 0.044, 0.046, 0.048, 0.050],
        'KATV 1': [0.0, 0.001, 0.003, 0.008, 0.011, 0.017, 0.020, 0.022, 0.032, 0.034],
        'Azeronline': [0.0, 0.009, 0.014, 0.031, 0.041, 0.048, 0.051, 0.054, 0.058, 0.060],
        'EngiNet': [0.0, 0.014, 0.030, 0.043, 0.052, 0.058, 0.063, 0.068, 0.070, 0.072],
        'Delta Telecom': [0.0, 0.021, 0.035, 0.039, 0.048, 0.061, 0.067, 0.069, 0.064, 0.063],
        'Caspian Telecom': [0.0, 0.003, 0.024, 0.048, 0.057, 0.061, 0.064, 0.067, 0.071, 0.076],
        'Selnet': [0.0, 0.001, 0.016, 0.016, 0.014, 0.027, 0.037, 0.041, 0.039, 0.038],
        'GSP': [0.0, 0.000, 0.005, 0.014, 0.013, 0.020, 0.023, 0.023, 0.023, 0.022]
    }
    fevd_df = pd.DataFrame(data).set_index('Step')
    return fevd_df

# Create version 1: Improved stacked area chart
def plot_stacked_area():
    fevd_df = create_variance_decomposition()
    
    plt.figure(figsize=(14, 10))
    ax = fevd_df.plot(kind='area', stacked=True, color=sns.color_palette("viridis", n_colors=11), alpha=0.8)
    
    # Format axes
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.title('Variance Decomposition: Sources of Variability in Download Speed', fontsize=18, pad=20)
    plt.ylabel('Proportion of Variance Explained', fontsize=16)
    plt.xlabel('Forecast Horizon (Steps Ahead)', fontsize=16)
    plt.ylim(0, 1.05)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Improve appearance of x-axis ticks
    plt.xticks(range(len(fevd_df)), [f'Step {i+1}' for i in range(len(fevd_df))], rotation=0)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    labels_with_percent = []
    for i, label in enumerate(labels):
        # Get the final contribution percentage
        final_contrib = fevd_df.iloc[-1, i] * 100
        # Format: ISP (final % contribution)
        labels_with_percent.append(f"{label} ({final_contrib:.1f}%)")
    
    # Place legend outside the plot
    plt.legend(handles, labels_with_percent, title='ISP (Final % Contribution)', 
               loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    
    # Add clear annotations
    plt.annotate('At Step 1, 100% of variance is\nexplained by combined providers', 
                 xy=(0, 1.0), xytext=(0.5, 0.95), xycoords='data', textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 horizontalalignment='center', fontsize=13,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#333333", alpha=0.9))
    
    # Add second annotation for final step
    plt.annotate(f'By Step 10, individual ISPs explain\n{(1-fevd_df.iloc[-1,0])*100:.1f}% of variance', 
                 xy=(9, 0.5), xytext=(0.5, 0.3), xycoords='data', textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 horizontalalignment='center', fontsize=13,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#333333", alpha=0.9))
    
    # Add white border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(1.5)
    
    # Add watermark explaining FEVD
    explanation = "Forecast Error Variance Decomposition (FEVD) shows the relative\n"
    explanation += "contribution of each ISP to the variance of forecasting errors over time."
    plt.figtext(0.5, 0.01, explanation, ha="center", fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="gray"))
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.97])
    plt.savefig('improved_variance_decomposition1.png', dpi=300, bbox_inches='tight')
    return 'improved_variance_decomposition1.png'

# Create version 2: Bar chart for better comparison
def plot_bar_chart():
    fevd_df = create_variance_decomposition()
    
    # Select specific steps for clarity
    steps_to_plot = ['Step 1', 'Step 3', 'Step 5', 'Step 7', 'Step 10']
    df_selected = fevd_df.loc[steps_to_plot]

    # Create a figure with multiple bar charts
    fig, axes = plt.subplots(1, len(steps_to_plot), figsize=(16, 8), sharey=True)
    
    # Color palette
    colors = sns.color_palette("viridis", n_colors=11)
    
    # Plot each step
    for i, step in enumerate(steps_to_plot):
        data = df_selected.loc[step]
        # Sort from largest to smallest for last step to make it easier to read
        if step == 'Step 10':
            data = data.sort_values(ascending=False)
        
        # Plot horizontal bars
        bars = axes[i].barh(range(len(data)), data, color=colors, alpha=0.7)
        
        # Add value labels to the bars
        for j, bar in enumerate(bars):
            if data.iloc[j] > 0.03:  # Only label if the value is significant
                axes[i].text(data.iloc[j] + 0.01, j, f"{data.iloc[j]:.2f}", 
                             va='center', fontsize=9)
        
        # Set titles and format
        axes[i].set_title(step, fontsize=14)
        axes[i].grid(axis='x', linestyle='--', alpha=0.4)
        axes[i].set_xlim(0, 1.05)
        
        # Only add y-tick labels on the first chart
        if i == 0:
            axes[i].set_yticks(range(len(data)))
            axes[i].set_yticklabels(data.index)
        else:
            axes[i].set_yticks(range(len(data)))
            axes[i].set_yticklabels([])
        
        # Format x-axis as percentage
        axes[i].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add common xlabel and ylabel
    fig.text(0.5, 0.01, 'Proportion of Variance Explained', ha='center', fontsize=16)
    fig.text(0.01, 0.5, 'Internet Service Provider', va='center', rotation='vertical', fontsize=16)
    
    # Add main title
    fig.suptitle('Variance Decomposition Across Forecast Horizon', fontsize=18, y=0.98)
    
    # Add explanation
    explanation = "This visualization shows how the sources of uncertainty in download speed forecasts evolve over time.\n"
    explanation += "Initially, the combined providers explain all variance, but individual ISPs increasingly contribute as the forecast extends."
    fig.text(0.5, 0.05, explanation, ha="center", fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="gray"))
    
    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.95])
    plt.savefig('improved_variance_decomposition2.png', dpi=300, bbox_inches='tight')
    return 'improved_variance_decomposition2.png'

# Create version 3: Pie charts
def plot_pie_charts():
    fevd_df = create_variance_decomposition()
    
    # Select specific steps for clarity
    steps_to_plot = ['Step 1', 'Step 3', 'Step 5', 'Step 10']
    
    # Create 2x2 grid of pie charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("viridis", n_colors=11)
    
    # Plot each step
    for i, step in enumerate(steps_to_plot):
        data = fevd_df.loc[step]
        
        # Combine small slices for better readability
        threshold = 0.03
        small_isps = data[data < threshold].index.tolist()
        if small_isps:
            other_value = data[small_isps].sum()
            clean_data = data.drop(small_isps)
            clean_data['Other ISPs'] = other_value
        else:
            clean_data = data
            
        # Sort by value for consistent color assignment
        clean_data = clean_data.sort_values(ascending=False)
        
        # Create pie chart
        wedges, texts, autotexts = axes[i].pie(
            clean_data, 
            colors=colors[:len(clean_data)],
            autopct='%1.1f%%', 
            startangle=90,
            pctdistance=0.85,
            explode=[0.05 if x != 'All Providers Combined' and clean_data[x] > 0.05 else 0 for x in clean_data.index]
        )
        
        # Styling
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
            
        # Add title
        axes[i].set_title(f"{step}: Variance Contribution", fontsize=14, pad=20)
        
        # Equal aspect ratio
        axes[i].set_aspect('equal')
    
    # Create unified legend for all charts
    handles = []
    labels = []
    for name, color in zip(fevd_df.columns, colors):
        handles.append(plt.Rectangle((0,0), 1, 1, color=color))
        # Get the final contribution percentage
        final_contrib = fevd_df.loc['Step 10', name] * 100
        # Format: ISP (final % contribution)
        labels.append(f"{name} ({final_contrib:.1f}%)")
        
    fig.legend(handles, labels, title='ISP (Final % Contribution)', 
               loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=4, fontsize=11, frameon=True)
    
    # Add main title
    fig.suptitle('Evolution of Variance Components Over Forecast Horizon', fontsize=18, y=0.98)
    
    # Add explanation
    explanation = "These pie charts show how the contribution to forecast variance evolves over time.\n"
    explanation += "At Step 1, all variance is internal to the combined providers metric,\n"
    explanation += "but individual ISPs increasingly influence the uncertainty as the forecast extends to Step 10."
    
    fig.text(0.5, 0.08, explanation, ha="center", fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="gray"))
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig('improved_variance_decomposition3.png', dpi=300, bbox_inches='tight')
    return 'improved_variance_decomposition3.png'

# Create all three improved visualizations
if __name__ == "__main__":
    plot1 = plot_stacked_area()
    plot2 = plot_bar_chart()
    plot3 = plot_pie_charts()
    
    print(f"✓ Improved stacked area chart saved to: {plot1}")
    print(f"✓ Bar chart visualization saved to: {plot2}")
    print(f"✓ Pie chart visualization saved to: {plot3}")
    print("\nAll three visualizations provide different perspectives on the same variance decomposition data.")
