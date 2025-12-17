# End-to-End NLP Sentiment Analysis Pipeline

A complete machine learning pipeline for sentiment analysis with data preprocessing, model training, evaluation, and deployment options (API and web app).

## 📁 Project Structure

```
├── api/
│   └── main.py                 # FastAPI REST API
├── data/
│   ├── processed/              # Processed data and results
│   └── raw/                    # Raw dataset
├── models/                     # Saved models and vectorizers
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data Preprocessing
│   └── 03_modeling.ipynb      # Model Training & Evaluation
├── src/
│   ├── preprocessing.py       # Preprocessing module
│   ├── predict.py             # Prediction module
│   └── train.py               # Training module
├── streamlit_app/
│   └── app.py                 # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository** (or ensure you have the project structure)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Dataset Setup

Place your sentiment dataset in `data/raw/` with the following structure:
- Column 1: `review` - text data
- Column 2: `sentiment` - labels (positive/negative)

Example CSV:
```csv
review,sentiment
"This movie was great!",positive
"Terrible film.",negative
```

## 📊 Usage

### 1. Exploratory Data Analysis (EDA)

Run the EDA notebook to understand your data:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This will:
- Analyze data distribution
- Generate visualizations
- Create word clouds
- Identify patterns

### 2. Data Preprocessing

Run the preprocessing notebook:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

Or use the script:
```bash
cd src
python preprocessing.py
```

Preprocessing steps:
- Remove HTML tags
- Remove URLs
- Convert to lowercase
- Remove special characters
- Remove stopwords
- Lemmatization
- Train/validation/test split

### 3. Model Training

Run the modeling notebook:

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Or train via command line:
```bash
cd src
python train.py --train ../data/processed/train.csv --val ../data/processed/val.csv
```

This trains multiple models:
- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

### 4. Make Predictions

Use the prediction module:

```python
from src.predict import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor(
    model_path='models/best_model.pkl',
    vectorizer_path='models/tfidf_vectorizer.pkl'
)

# Single prediction
result = predictor.predict("This movie was amazing!")
print(result)
# Output: {'text': '...', 'sentiment': 'positive', 'label': 1, 'confidence': 0.95}

# Batch prediction
texts = ["Great movie!", "Terrible film."]
results = predictor.predict_batch(texts)
```

## 🌐 Deployment Options

### Option 1: Streamlit Web App

Launch the interactive web application:

```bash
cd streamlit_app
streamlit run app.py
```

Features:
- Single text analysis
- Batch processing
- CSV upload/download
- Visual analytics
- Confidence scores

Access at: `http://localhost:8501`

### Option 2: FastAPI REST API

Launch the REST API:

```bash
cd api
uvicorn main:app --reload
```

API Endpoints:

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was great!"}'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Bad film."]}'
```

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📈 Model Performance

The pipeline trains multiple models and selects the best performer:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~0.85 | ~0.84 | ~0.87 | ~0.85 |
| Logistic Regression | ~0.89 | ~0.88 | ~0.90 | ~0.89 |
| Linear SVM | ~0.88 | ~0.87 | ~0.89 | ~0.88 |
| Random Forest | ~0.85 | ~0.84 | ~0.86 | ~0.85 |

*Note: Actual performance depends on your dataset*

## 🛠️ Customization

### Adding New Models

Edit `notebooks/03_modeling.ipynb` or `src/train.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}
```

### Modifying Preprocessing

Edit `src/preprocessing.py` to add/remove steps:

```python
def preprocess(self, text):
    text = self.remove_html_tags(text)
    text = self.custom_step(text)  # Add your step
    # ... other steps
    return text
```

### Changing Features

Modify vectorizer in `src/train.py`:

```python
# Use different n-grams
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3)  # trigrams
)

# Or use Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
```

## 📝 File Descriptions

### Notebooks
- **01_eda.ipynb**: Data exploration and visualization
- **02_preprocessing.ipynb**: Text cleaning and preparation
- **03_modeling.ipynb**: Model training and evaluation

### Source Code
- **preprocessing.py**: TextPreprocessor class for text cleaning
- **predict.py**: SentimentPredictor class for inference
- **train.py**: Training script with evaluation

### Applications
- **streamlit_app/app.py**: Interactive web interface
- **api/main.py**: REST API with FastAPI

## 🤝 Contributing

Feel free to:
- Add new models
- Improve preprocessing
- Enhance visualizations
- Optimize performance

## 📄 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

**NLTK Data Error:**
```bash
python -c "import nltk; nltk.download('all')"
```

**Model Not Found:**
Ensure you've run the training notebook/script first to generate model files.

**Import Errors:**
Check that all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Analyzing! 🎉**