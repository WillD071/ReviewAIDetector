import requests
from bs4 import BeautifulSoup
import webbrowser
import os
import pandas as pd
import json
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import load

with open("Cookies.json", "r") as f:
    cookies = json.load(f)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Accept-Language': 'en-US, en;q=0.5'
}

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

def predict_review_legitimacy(review, model_path='Training/review_model.joblib', vectorizer_path='Training/vectorizer.joblib'):
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


# Scrape the data
def getdata(url):
    try:
        r = requests.get(url, headers=HEADERS, cookies=cookies)  # Add cookies here
        r.raise_for_status()  # Raise an error for bad responses
        return r.text
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def html_code(url):
    htmldata = getdata(url)
    if htmldata is None:  # Check if data was fetched successfully
        return None
    soup = BeautifulSoup(htmldata, 'html.parser')
    return soup

# Extract reviews, dates, and details
def cus_rev(soup):
    reviews = []
    review_items = soup.find_all("div", {"data-hook": "review"})
    
    if not review_items:
        print("No reviews found. Saving soup content to an HTML file for inspection...")
        save_soup_to_html(soup)  # Save soup if no reviews found
        exit()
        return reviews  # Return empty list if no reviews found

    for item in review_items:
        review_body = item.find("span", {"data-hook": "review-body"})
        review_date = item.find("span", class_="a-size-base a-color-secondary review-date")
        
        if review_body and review_date:
            reviews.append({
                'date': review_date.get_text().strip(),
                'body': review_body.get_text().strip()
            })
    
    return reviews

# Save the soup to an HTML file and open it in a browser
def save_soup_to_html(soup):
    file_path = "Output/soup_output.html"
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(soup.prettify())  # Prettify makes the HTML more readable
    
    # Open the file in the default browser
    webbrowser.open(f"file://{os.path.realpath(file_path)}")

# Main scraping function
def scrape_reviews(url, num_pages=1):
    all_rev_data = []

    page = 1
    while True:
        page_url = f"{url}&pageNumber={page}"
        soup = html_code(page_url)

        if soup is None:  # Check if the soup was created successfully
            continue
        
        rev_data = cus_rev(soup)

        # Extracting only review bodies and dates
        rev_result = [(r['date'], r['body']) for r in rev_data]

        all_rev_data.extend(rev_result)

        if soup.find("li", class_="a-disabled a-last"):
            break #breaks if last page

        page += 1
        
    
    return all_rev_data

def iterateCSV():
    df = pd.read_csv('Output/amazon_reviews.csv')  # Replace with your actual CSV filename
    # Analyze reviews
    total_reviews = len(df)
    ai_generated_count = 0



    with open("Output/fake_reviews.csv", mode="w", newline="") as fake_reviews_file:
        csv_writer = csv.writer(fake_reviews_file)
        
        # Write the header (column names)
        csv_writer.writerow(df.columns)  # Write the same column headers as in the original DataFrame
        
        # Loop through each review in the DataFrame
        for index, row in df.iterrows():  # iterrows() returns an index and a row for each entry
            review_body = row['Review']  # Adjust the column name if necessary
            if predict_review_legitimacy(review_body) == "Fake":
                ai_generated_count += 1
                # Write the entire row of the fake review to the CSV
                csv_writer.writerow(row)
    
    # Calculate the percentage of AI-generated reviews
    if total_reviews > 0:
        ai_percentage = (ai_generated_count / total_reviews) * 100
        print(f"Percentage of fake reviews: {ai_percentage:.2f}%")

def save_reviews_to_csv(review_data):
        # Check if there are reviews to save
        if not review_data:
            print("No review data to save.")
            return

        # Create DataFrame
        df = pd.DataFrame(review_data, columns=["Date", "Review"])
        
        # Save to CSV
        df.to_csv('Output/amazon_reviews.csv', index=False)
        print("Reviews saved to 'Output/amazon_reviews.csv'.")

def scrapeAndSave():
    # Replace the URL with the product review URL
    url = "https://www.amazon.com/AstroAI-Compressor-Portable-Inflator-Digital/product-reviews/B0CJFCRKBD?reviewerType=all_reviews"
    num_pages = 15  # Specify the number of pages you want to scrape

    review_data = scrape_reviews(url, num_pages)

    # Save the scraped reviews to a CSV file
    save_reviews_to_csv(review_data)

    if review_data:
        print("Reviews successfully scraped.")
    else:
        print("No reviews were scraped.")
    

def main():
    scrapeAndSave()
    
    iterateCSV()


if __name__ == "__main__":
    main()
