# app.py
from flask import Flask, render_template, request
import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd

app = Flask(__name__)

kaggle_dataset_path = r'C:\Users\aviro\Desktop\web_scraper 2.0\df_file.csv'
kaggle_dataset = pd.read_csv(kaggle_dataset_path)

# Create a label mapping dictionary
label_mapping = {0: "Politics", 1: "Sport", 2: "Technology", 3: "Entertainment", 4: "Business"}

# Step 2: Web Scraping
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_content = soup.get_text()
    return text_content

# Step 3: NLP Analysis - Text Classification using scikit-learn
def classify_topic(text_content):
    # Use the Kaggle dataset for training or inference
    texts = kaggle_dataset['Text'].tolist()
    numeric_labels = kaggle_dataset['Label'].tolist()

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Train a linear SVM classifier
    classifier = LinearSVC()
    classifier.fit(X, numeric_labels)

    # Classify the input text
    input_vector = vectorizer.transform([text_content])
    predicted_category_numeric = classifier.predict(input_vector)[0]

    # Map predicted numeric label to string label
    predicted_category = label_mapping[predicted_category_numeric]

    return predicted_category

# Step 4: Store Data in a Single File
def store_data_in_file(output_folder, topic):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, 'website_topic.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(str(topic))
    return file_path

# Main route with user input form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        output_folder = 'output'

        # Step 2: Web Scraping
        website_content = scrape_website(url)
        print("Web Scraping Done.")

        # Step 3: NLP Analysis - Text Classification using scikit-learn
        predicted_topic = classify_topic(website_content)
        print("Topic Classification Done.")
        
        # Step 4: Store Data in a Single File
        file_path = store_data_in_file(output_folder, predicted_topic)
        print(f"Website topic stored in '{file_path}'.")

        return render_template('index.html', topic=predicted_topic, user_input=url)

    return render_template('form.html')

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
