"""
Prediction Module for Sentiment Analysis
"""
import pickle
import numpy as np
from src.preprocessing import TextPreprocessor


class SentimentPredictor:
    """Sentiment prediction class"""
    
    def __init__(self, model_path, vectorizer_path, preprocessor_path=None):
        """
        Initialize predictor with trained model and vectorizer
        
        Args:
            model_path: Path to trained model pickle file
            vectorizer_path: Path to fitted vectorizer pickle file
            preprocessor_path: Path to preprocessor pickle file (optional)
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load or create preprocessor
        if preprocessor_path:
            self.preprocessor = TextPreprocessor.load(preprocessor_path)
        else:
            self.preprocessor = TextPreprocessor()
        
        # Sentiment mapping
        self.sentiment_map = {0: 'negative', 1: 'positive'}
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text string
            
        Returns:
            dict: Prediction results with sentiment and confidence
        """
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(text_vector)[0]
            confidence = float(np.max(proba))
        else:
            confidence = None
        
        return {
            'text': text,
            'sentiment': self.sentiment_map[prediction],
            'label': int(prediction),
            'confidence': confidence
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input text strings
            
        Returns:
            list: List of prediction dictionaries
        """
        # Preprocess all texts
        cleaned_texts = self.preprocessor.preprocess_batch(texts)
        
        # Vectorize
        text_vectors = self.vectorizer.transform(cleaned_texts)
        
        # Predict
        predictions = self.model.predict(text_vectors)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(text_vectors)
            confidences = np.max(probas, axis=1)
        else:
            confidences = [None] * len(predictions)
        
        # Build results
        results = []
        for text, pred, conf in zip(texts, predictions, confidences):
            results.append({
                'text': text,
                'sentiment': self.sentiment_map[pred],
                'label': int(pred),
                'confidence': float(conf) if conf is not None else None
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = SentimentPredictor(
        model_path='../models/best_model.pkl',
        vectorizer_path='../models/tfidf_vectorizer.pkl'
    )
    
    # Test predictions
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointed.",
        "It was okay, nothing special but not bad either.",
        "Best movie I've seen this year! Highly recommended!",
        "Boring and predictable. Would not watch again."
    ]
    
    print("="*70)
    print("SENTIMENT PREDICTIONS")
    print("="*70)
    
    # Single predictions
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.2%}")
        print("-"*70)
    
    # Batch prediction
    print("\n" + "="*70)
    print("BATCH PREDICTIONS")
    print("="*70)
    
    batch_results = predictor.predict_batch(test_texts)
    for i, result in enumerate(batch_results, 1):
        print(f"\n{i}. {result['sentiment'].upper()}", end="")
        if result['confidence']:
            print(f" ({result['confidence']:.2%})")
        else:
            print()
