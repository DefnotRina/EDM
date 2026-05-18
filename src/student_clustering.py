import pandas as pd
from sklearn.cluster import KMeans
import os

def cluster_students(df):
    print("Step 4: Student Profiling (K-Means Clustering)...")
    
    # Selecting the features for clustering (the components of the SBI)
    cluster_features = ['sleep_hours', 'exercise_frequency', 'diet_num', 
                        'study_hours_per_day', 'screen_time', 'stress_level']
    X = df[cluster_features]
    
    # We will use 3 clusters as a baseline for 'Balanced', 'Moderate', and 'At Risk'
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Analyze the clusters
    cluster_summary = df.groupby('Cluster')[cluster_features].mean()
    
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
        f.write("\n\nInterpretation Key:\n")
        f.write("- Positive values (e.g., 0.5): Above Average in that habit.\n")
        f.write("- Negative values (e.g., -0.5): Below Average in that habit.\n\n")
        f.write("Example: A negative score in 'sleep_hours' means that group is sleep-deprived.\n\n")
    
    print(f"Clustering complete. Identified 3 distinct student profiles.")
    print("Cluster summaries added to 'results/analysis_report.txt'.\n")
    
    return df
