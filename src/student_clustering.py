import pandas as pd
from sklearn.cluster import KMeans
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

PROFILE_NICKNAMES = {
    'digital_overloaders': 'Digital Overloaders',
    'academic_burnouts': 'Academic Burnouts',
    'high_load_digital_strivers': 'High-Load Digital Strivers',
    'recovery_deficit': 'Recovery Deficit Group',
    'nutrition_risk': 'Nutrition Risk Group',
    'balanced': 'Balanced Low-Strain Students',
    'moderate': 'Moderate Balance Students'
}


def _name_cluster(row):
    """Assign a presentation-friendly nickname from each cluster's mean z-scores."""
    high_strain = row['study_hours_per_day'] > 0.35 and row['stress_level'] > 0.35
    digital_overload = row['screen_time'] > 0.35 and row['sleep_hours'] < -0.25
    high_load = row['study_hours_per_day'] > 0.35 and row['screen_time'] > 0.35
    recovery_deficit = row['sleep_hours'] < -0.25 or row['exercise_frequency'] < -0.25
    poor_diet = row['diet_num'] < -0.35
    low_strain = row['study_hours_per_day'] < -0.35 and row['screen_time'] < -0.35
    strong_recovery = row['diet_num'] > 0.25 or row['sleep_hours'] > 0.25 or row['exercise_frequency'] > 0.25

    if digital_overload:
        return PROFILE_NICKNAMES['digital_overloaders']
    if high_strain:
        return PROFILE_NICKNAMES['academic_burnouts']
    if high_load:
        return PROFILE_NICKNAMES['high_load_digital_strivers']
    if poor_diet:
        return PROFILE_NICKNAMES['nutrition_risk']
    if recovery_deficit:
        return PROFILE_NICKNAMES['recovery_deficit']
    if low_strain and strong_recovery:
        return PROFILE_NICKNAMES['balanced']
    return PROFILE_NICKNAMES['moderate']


def _summarize_cluster(row):
    strongest = row[CLUSTER_FEATURES].abs().sort_values(ascending=False).head(3).index
    descriptions = []

    for feature in strongest:
        direction = "high" if row[feature] > 0 else "low"
        descriptions.append(f"{direction} {FEATURE_LABELS[feature]}")

    return ", ".join(descriptions)


def cluster_students(df):
    print("Step 4: Student Profiling (K-Means Clustering)...")
    
    # Selecting the features for clustering (the components of the SBI)
    X = df[CLUSTER_FEATURES]
    
    # We will use 3 clusters as a baseline for 'Balanced', 'Moderate', and 'At Risk'
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Analyze the clusters
    cluster_summary = df.groupby('Cluster')[CLUSTER_FEATURES].mean()
    cluster_summary['SBI'] = df.groupby('Cluster')['SBI'].mean()
    cluster_summary['Profile_Nickname'] = cluster_summary.apply(_name_cluster, axis=1)
    cluster_summary['Profile_Summary'] = cluster_summary.apply(_summarize_cluster, axis=1)

    nickname_map = cluster_summary['Profile_Nickname'].to_dict()
    df['Profile_Nickname'] = df['Cluster'].map(nickname_map)
    
    # Save the clustered data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/clustered_data.csv', index=False)
    
    # Append to the report
    report_path = 'results/analysis_report.txt'
    with open(report_path, 'a') as f:
        f.write("PHASE 1: STUDENT LIFESTYLE PROFILING (K-MEANS)\n")
        f.write("----------------------------------------------\n")
        f.write("Cluster Mean Values (Standardized Z-Scores):\n\n")
        f.write(cluster_summary.to_string())
        f.write("\n\nStudent Profile Nicknames:\n")
        for cluster_id, row in cluster_summary.iterrows():
            f.write(
                f"- Cluster {cluster_id}: {row['Profile_Nickname']} "
                f"({row['Profile_Summary']}; mean SBI={row['SBI']:.2f})\n"
            )
        f.write("\n\nInterpretation Key:\n")
        f.write("- Positive values (e.g., 0.5): Above Average in that habit.\n")
        f.write("- Negative values (e.g., -0.5): Below Average in that habit.\n\n")
        f.write("Example: A negative score in 'sleep_hours' means that group is sleep-deprived.\n\n")
    
    print(f"Clustering complete. Identified 3 distinct student profiles.")
    print("Cluster summaries added to 'results/analysis_report.txt'.\n")
    
    return df
