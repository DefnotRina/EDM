import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations(df):
    print("Step 5: Generating Visualizations...")
    os.makedirs('results/plots', exist_ok=True)
    
    # Set a premium aesthetic
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Plot 1: Distribution of Student Balance Index (SBI)
    plt.figure()
    sns.histplot(df['SBI'], kde=True, color="skyblue", bins=30)
    plt.title("Distribution of Student Balance Index (SBI)", fontsize=16, fontweight='bold')
    plt.xlabel("SBI Score (Balance Level)", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)
    plt.savefig('results/plots/sbi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: SBI vs Dropout Risk (Boxplot)
    plt.figure()
    sns.boxplot(x='dropout_risk', y='SBI', data=df, palette="coolwarm")
    plt.title("SBI vs. Dropout Risk", fontsize=16, fontweight='bold')
    plt.xlabel("Dropout Risk", fontsize=12)
    plt.ylabel("Student Balance Index (SBI)", fontsize=12)
    plt.savefig('results/plots/sbi_vs_dropout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Cluster Profiles (Comparison of Habits)
    cluster_features = ['sleep_hours', 'exercise_frequency', 'diet_num', 
                        'study_hours_per_day', 'screen_time', 'stress_level']
    cluster_means = df.groupby('Cluster')[cluster_features].mean().reset_index()
    
    # Melt the data for easier plotting with seaborn
    melted_clusters = pd.melt(cluster_means, id_vars='Cluster', var_name='Habit', value_name='Z-Score')
    
    plt.figure()
    sns.barplot(x='Habit', y='Z-Score', hue='Cluster', data=melted_clusters, palette="viridis")
    plt.title("Comparison of Lifestyle Habits Across Clusters", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(title='Student Profile (Cluster)')
    plt.savefig('results/plots/cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations generated and saved to 'results/plots/'.\n")
