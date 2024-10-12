import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses text by removing non-alphabetical characters, tokenizing, removing stopwords, and lemmatizing."""
    text = re.sub(r'\W', ' ', text.lower())  # Remove non-alphabet characters and lowercase
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load data from CSV
#def load_data(file_path):
 #   """Loads the dataset, extracts review text and labels, mapping labels 'CG' to 1 and 'OR' to 0."""
  #  data = pd.read_csv(file_path)
   # reviews = data['text'].astype(str)  # Convert to string if not already
    #labels = data['label'].map({'CG': 1, 'OR': 0})  # Map 'CG' to 1 (fake) and 'OR' to 0 (real)
    #return reviews, labels

# Specify your file path
file_path = "fakeReviews.csv"  # Replace with your file path

# Load and preprocess data
reviews, labels = load_data(file_path)
processed_reviews = [preprocess_text(review) for review in reviews]

# Transform reviews to TF-IDF representation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(processed_reviews)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
#Best parameters found:  {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Initialize the SVM model
svm_model = SVC()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set
predictions = best_svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save the best model
dump(best_svm_model, 'review_model.joblib')
