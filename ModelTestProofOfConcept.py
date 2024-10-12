import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import load

# Download necessary NLTK data files (if not already done)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and preprocesses text by removing non-alphabetical characters, tokenizing, and lemmatizing."""
    text = re.sub(r'\W', ' ', text.lower())  # Remove non-alphabet characters and lowercase
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def predict_review_legitimacy(review, model_path='review_model.joblib', vectorizer_path='vectorizer.joblib'):
    """Predicts whether a review is fake or not using a trained SVM model.
    
    Args:
        review (str): The review text to be classified.
        model_path (str): The file path of the trained SVM model.
        vectorizer_path (str): The file path of the TF-IDF vectorizer.

    Returns:
        str: 'Fake' if the review is classified as fake (CG), 'Real' if classified as real (OR).
    """
    # Load the trained SVM model and vectorizer
    svm_model = load(model_path)
    vectorizer = load(vectorizer_path)

    # Preprocess the input review
    processed_review = preprocess_text(review)

    # Transform the review to TF-IDF representation
    review_vector = vectorizer.transform([processed_review])

    # Make a prediction
    prediction = svm_model.predict(review_vector)

    # Return the result
    return 'Fake' if prediction[0] == 1 else 'Real'

# Example usage
if __name__ == "__main__":
    review_to_test = "absolutely beautiful and perfect. I would recommend this to everyone I know. so happy with this product does exactly what I need it to sleek design. highly recommended!"  # Replace with the review you want to test
    result = predict_review_legitimacy(review_to_test)
    print(f"The review is classified as: {result}")
