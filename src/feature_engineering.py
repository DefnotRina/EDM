import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def engineer_features(df):
    print("Step 2: Feature Engineering (Calculating the SBI)...")
    
    # Scale features using Z-score standardization as per the proposal formula
    scaler = StandardScaler()
    cols_to_scale = ['sleep_hours', 'exercise_frequency', 'study_hours_per_day', 'screen_time', 'stress_level', 'diet_num']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Calculate Student Balance Index (SBI)
    # 1. Recovery factors (Weighted Z-scores)
    # Recovery = 0.4(Z_sleep) + 0.3(Z_exercise) + 0.3(Z_diet)
    recovery = (0.4 * df['sleep_hours']) + (0.3 * df['exercise_frequency']) + (0.3 * df['diet_num'])
    
    # 2. Strain factors (Weighted Z-scores)
    # Strain = 0.3(Z_study) + 0.3(Z_screen) + 0.4(Z_stress)
    strain = (0.3 * df['study_hours_per_day']) + (0.3 * df['screen_time']) + (0.4 * df['stress_level'])
    
    # 3. Final SBI Formula
    df['SBI'] = recovery - strain
    
    # Categorize SBI for potential classification tasks
    df['SBI_Category'] = pd.qcut(df['SBI'], q=3, labels=['At Risk', 'Moderate', 'Balanced'])
    
    # Save the feature-engineered data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/engineered_data.csv', index=False)
    
    print("Feature engineering complete. Student Balance Index (SBI) calculated.")
    print("Saved to 'data/processed/engineered_data.csv'.\n")
    return df
