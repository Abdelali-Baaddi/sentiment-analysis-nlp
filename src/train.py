"""
Model Training Module
"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os


def train_model(train_path, val_path, model_output_path, vectorizer_output_path):
    """
    Train sentiment analysis model
    
    Args:
        train_path: Path to training data CSV
        val_path: Path to validation data CSV
        model_output_path: Path to save trained model
        vectorizer_output_path: Path to save fitted vectorizer
    """
    print("="*50)
    print("LOADING DATA")
    print("="*50)
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_val = val_df['text']
    y_val = val_df['label']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Feature extraction
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    print(f"Feature shape: {X_train_vec.shape}")
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    print("✓ Training complete")
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    
    y_val_pred = model.predict(X_val_vec)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Save model and vectorizer
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_output_path}")
    
    with open(vectorizer_output_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved to {vectorizer_output_path}")
    
    return model, vectorizer, {'accuracy': accuracy, 'precision': precision, 
                              'recall': recall, 'f1': f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--train', type=str, default='../data/processed/train.csv',
                       help='Path to training data')
    parser.add_argument('--val', type=str, default='../data/processed/val.csv',
                       help='Path to validation data')
    parser.add_argument('--model-output', type=str, default='../models/model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--vectorizer-output', type=str, default='../models/vectorizer.pkl',
                       help='Path to save vectorizer')
    
    args = parser.parse_args()
    
    # Train model
    model, vectorizer, metrics = train_model(
        args.train,
        args.val,
        args.model_output,
        args.vectorizer_output
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)