"""
Text Preprocessing Module for NLP Pipeline
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Text preprocessing class for NLP tasks"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        return re.sub(r'http\S+|www.\S+', '', text)
    
    def remove_special_chars(self, text):
        """Remove special characters and digits"""
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def convert_to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = text.split()
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        if not isinstance(text, str):
            text = str(text)
        
        # Apply all preprocessing steps
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.convert_to_lowercase(text)
        text = self.remove_special_chars(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]
    
    def save(self, filepath):
        """Save preprocessor to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test text
    sample_text = """
    <p>This is a GREAT movie!!! I absolutely loved it. 
    Visit http://example.com for more reviews. 
    The acting was superb and the storyline kept me engaged throughout.</p>
    """
    
    print("Original Text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Preprocess
    cleaned_text = preprocessor.preprocess(sample_text)
    
    print("Preprocessed Text:")
    print(cleaned_text)