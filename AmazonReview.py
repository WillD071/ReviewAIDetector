import requests
from bs4 import BeautifulSoup
import webbrowser
import os
import pandas as pd
import 
# Replace these with your actual cookie values
cookies = {
    'session-id': '',
    'session-token': '',
    'ubid-main': '',
    'x-main': '',
    'at-main': '',
    'lc-main': 'en_US',
    'sess-at-main': '',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Accept-Language': 'en-US, en;q=0.5'
}


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
    file_path = "soup_output.html"
    
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


def is_ai_generated(text):
    result = aidetector.  # Adjust based on the actual function in aidetector
    return result['is_ai_generated']  # Adjust based on the actual output of the method

def iterateCSV():
    df = pd.read_csv('amazon_reviews.csv')  # Replace with your actual CSV filename
    # Analyze reviews
    total_reviews = len(df)
    ai_generated_count = 0

    # Loop through each review in the DataFrame
    for index, row in df.iterrows():  # iterrows() returns an index and a row for each entry
        review_body = row['Review Body']  # Adjust the column name if necessary
        if is_ai_generated(review_body):
            ai_generated_count += 1

    # Calculate the percentage of AI-generated reviews
    if total_reviews > 0:
        ai_percentage = (ai_generated_count / total_reviews) * 100
        print(f"Percentage of AI-generated reviews: {ai_percentage:.2f}%")


def main():
    # Function to save reviews to a CSV file
    def save_reviews_to_csv(review_data):
        # Check if there are reviews to save
        if not review_data:
            print("No review data to save.")
            return

        # Create DataFrame
        df = pd.DataFrame(review_data, columns=["Date", "Review"])
        
        # Save to CSV
        df.to_csv('amazon_reviews.csv', index=False)
        print("Reviews saved to 'amazon_reviews.csv'.")

    # Replace the URL with the product review URL
    url = "https://www.amazon.com/MORNYRAY-Waterproof-Snowproof-Protection-Windproof/product-reviews/B08BNHV8Z4?reviewerType=all_reviews"
    num_pages = 2  # Specify the number of pages you want to scrape

    review_data = scrape_reviews(url, num_pages)

    # Save the scraped reviews to a CSV file
    save_reviews_to_csv(review_data)

    if review_data:
        print("Reviews successfully scraped.")
    else:
        print("No reviews were scraped.")
    
    iterateCSV


if __name__ == "__main__":
    main()
