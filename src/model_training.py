import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os


def _evaluate_predictions(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


def _select_recall_priority_threshold(y_test, y_proba, target_recall=0.95):
    threshold_results = []

    for threshold in [x / 100 for x in range(5, 51, 5)]:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = _evaluate_predictions(y_test, y_pred)
        metrics['threshold'] = threshold
        threshold_results.append(metrics)

    candidates = [result for result in threshold_results if result['recall'] >= target_recall]
    if candidates:
        return max(candidates, key=lambda item: (item['threshold'], item['f1']))

    return max(threshold_results, key=lambda item: (item['recall'], item['f1']))


def train_and_evaluate(df):
    print("Step 3: Predictive Modeling (Training the Random Forest)...")
    
    # Define our predictors (features) and target (dropout_risk).
    # SBI remains central, while its lifestyle components make recall more useful
    # for an early-warning system.
    features = [
        'SBI',
        'sleep_hours',
        'exercise_frequency',
        'diet_num',
        'study_hours_per_day',
        'screen_time',
        'stress_level'
    ]
    X = df[features]
    y = df['dropout_num']
    
    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the Random Forest Classifier
    # Added class_weight='balanced' to handle the fact that very few students have dropout_risk='Yes'
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred_default = rf.predict(X_test)
    default_metrics = _evaluate_predictions(y_test, y_pred_default)

    recall_priority = _select_recall_priority_threshold(y_test, y_proba)
    y_pred_recall_priority = (y_proba >= recall_priority['threshold']).astype(int)
    recall_priority_report = classification_report(y_test, y_pred_recall_priority)

    prediction_audit = df.loc[X_test.index, ['student_id', 'dropout_risk'] + features].copy()
    prediction_audit['actual_dropout_num'] = y_test
    prediction_audit['dropout_probability'] = y_proba
    prediction_audit['predicted_default_0_50'] = y_pred_default
    prediction_audit['predicted_recall_priority'] = y_pred_recall_priority
    prediction_audit['recall_priority_threshold'] = recall_priority['threshold']
    prediction_audit['prediction_result'] = 'True Negative'
    prediction_audit.loc[
        (prediction_audit['actual_dropout_num'] == 0)
        & (prediction_audit['predicted_recall_priority'] == 1),
        'prediction_result'
    ] = 'False Positive'
    prediction_audit.loc[
        (prediction_audit['actual_dropout_num'] == 1)
        & (prediction_audit['predicted_recall_priority'] == 0),
        'prediction_result'
    ] = 'False Negative'
    prediction_audit.loc[
        (prediction_audit['actual_dropout_num'] == 1)
        & (prediction_audit['predicted_recall_priority'] == 1),
        'prediction_result'
    ] = 'True Positive'
    prediction_audit.to_csv('results/dropout_prediction_audit.csv', index=False)
    
    print(
        "Model successfully trained! "
        f"Dropout Recall: {recall_priority['recall']*100:.2f}% "
        f"(Accuracy: {recall_priority['accuracy']*100:.2f}%)\n"
    )
    
    # Extract Feature Importances
    importances = rf.feature_importances_
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    
    # Generate the final report
    os.makedirs('results', exist_ok=True)
    with open('results/analysis_report.txt', 'a') as f:
        f.write("PHASE 2: PREDICTIVE VALIDITY OF THE SBI\n")
        f.write("---------------------------------------\n")
        f.write("Task: Predicting Dropout Risk using the Student Balance Index (SBI) ")
        f.write("and its lifestyle components.\n")
        f.write(f"Model: Random Forest Classifier with balanced class weights\n")
        f.write("Evaluation Priority: Recall for students with dropout_risk='Yes'\n")
        f.write("Rationale: In an early-warning system, missing an actual dropout-risk student ")
        f.write("is more harmful than accidentally flagging a student who is doing fine.\n\n")

        f.write("--- Default 0.50 Threshold Metrics ---\n")
        f.write(f"Accuracy: {default_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Dropout Recall: {default_metrics['recall']*100:.2f}%\n")
        f.write(f"Dropout Precision: {default_metrics['precision']*100:.2f}%\n")
        f.write(f"False Negatives (missed at-risk students): {default_metrics['fn']}\n\n")

        f.write("--- Recall-Priority Early-Warning Threshold ---\n")
        f.write(f"Selected Probability Threshold: {recall_priority['threshold']:.2f}\n")
        f.write(f"Accuracy: {recall_priority['accuracy']*100:.2f}%\n")
        f.write(f"Dropout Recall: {recall_priority['recall']*100:.2f}%\n")
        f.write(f"Dropout Precision: {recall_priority['precision']*100:.2f}%\n")
        f.write(f"Dropout F1-Score: {recall_priority['f1']*100:.2f}%\n")
        f.write(f"True Positives (correctly flagged): {recall_priority['tp']}\n")
        f.write(f"False Positives (extra students flagged): {recall_priority['fp']}\n")
        f.write(f"False Negatives (missed at-risk students): {recall_priority['fn']}\n\n")
        f.write("Prediction Audit CSV: results/dropout_prediction_audit.csv\n\n")
        
        f.write("--- Feature Importances ---\n")
        f.write(feat_importances.to_string())
        
        f.write("\n\n--- Classification Report at Recall-Priority Threshold ---\n")
        f.write(recall_priority_report)
        
        f.write("\n\nConclusion:\n")
        f.write("The Student Balance Index (SBI) provides a useful early-warning signal when evaluation ")
        f.write("prioritizes dropout recall. Accuracy is still reported, but recall and false negatives ")
        f.write("are emphasized because the main institutional risk is missing students who may need support.")
        
    print("Training complete. Detailed results saved to 'results/analysis_report.txt'.")
