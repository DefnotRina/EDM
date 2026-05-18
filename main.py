import sys
import os

# Ensure the src folder is accessible
sys.path.append(os.path.abspath("src"))

from data_preprocessing import load_and_clean_data
from feature_engineering import engineer_features
from model_training import train_and_evaluate
from student_clustering import cluster_students
from visualizations import generate_visualizations

def main():
    print("===============================================")
    print("  PROJECT BALANCE: EDM Analysis Pipeline")
    print("===============================================\n")
    
    # Initialize the analysis report
    os.makedirs('results', exist_ok=True)
    with open('results/analysis_report.txt', 'w') as f:
        f.write("PROJECT BALANCE: COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=============================================\n\n")
    
    # 1. Path to the raw data
    raw_data_path = 'enhanced_student_habits_performance_dataset/enhanced_student_habits_performance_dataset.csv'
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Dataset not found at {raw_data_path}.")
        return

    # 2. Execute Pipeline
    cleaned_df = load_and_clean_data(raw_data_path)
    
    engineered_df = engineer_features(cleaned_df)
    
    clustered_df = cluster_students(engineered_df)
    
    train_and_evaluate(clustered_df)
    
    generate_visualizations(clustered_df)
    
    print("\n===============================================")
    print("  Pipeline Execution Completed Successfully!")
    print("===============================================")

if __name__ == "__main__":
    main()
