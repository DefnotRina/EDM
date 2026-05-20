import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLUSTER_FEATURES = [
    'sleep_hours',
    'exercise_frequency',
    'diet_num',
    'study_hours_per_day',
    'screen_time',
    'stress_level'
]

FEATURE_LABELS = {
    'sleep_hours': 'Sleep',
    'exercise_frequency': 'Exercise',
    'diet_num': 'Diet',
    'study_hours_per_day': 'Study Load',
    'screen_time': 'Screen Time',
    'stress_level': 'Stress'
}

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
    sns.boxplot(x='dropout_risk', y='SBI', hue='dropout_risk', data=df, palette="coolwarm", legend=False)
    plt.title("SBI vs. Dropout Risk", fontsize=16, fontweight='bold')
    plt.xlabel("Dropout Risk", fontsize=12)
    plt.ylabel("Student Balance Index (SBI)", fontsize=12)
    plt.savefig('results/plots/sbi_vs_dropout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Cluster Profiles (Comparison of Habits)
    profile_col = 'Profile_Nickname' if 'Profile_Nickname' in df.columns else 'Cluster'
    cluster_means = df.groupby(profile_col)[CLUSTER_FEATURES].mean().reset_index()
    
    # Melt the data for easier plotting with seaborn
    melted_clusters = pd.melt(
        cluster_means,
        id_vars=profile_col,
        var_name='Habit',
        value_name='Z-Score'
    )
    melted_clusters['Habit'] = melted_clusters['Habit'].map(FEATURE_LABELS)
    
    plt.figure()
    sns.barplot(x='Habit', y='Z-Score', hue=profile_col, data=melted_clusters, palette="viridis")
    plt.title("Lifestyle Habits by Student Profile Nickname", fontsize=16, fontweight='bold')
    plt.xlabel("SBI Component", fontsize=12)
    plt.ylabel("Mean Z-Score", fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(title='Student Profile', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig('results/plots/cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Cluster Profile Heatmap (teacher-friendly nicknames)
    heatmap_data = cluster_means.set_index(profile_col)[CLUSTER_FEATURES]
    heatmap_data = heatmap_data.rename(columns=FEATURE_LABELS)

    plt.figure(figsize=(12, 5.5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Mean Z-Score'}
    )
    plt.title("Named Student Lifestyle Profiles from K-Means", fontsize=16, fontweight='bold')
    plt.xlabel("SBI Component", fontsize=12)
    plt.ylabel("Student Profile Nickname", fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.savefig('results/plots/named_cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations generated and saved to 'results/plots/'.\n")
