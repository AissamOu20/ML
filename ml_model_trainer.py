import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset with specified dtypes
data = pd.read_csv(
    r'dataset/cleaned_data.csv',
    dtype={
        'subject': str,
        'body': str,
        'label': str,  # Label as string
        'sender': str,
        'receiver': str,
        'date': str,
        'text_combined': str
    }
)

# Print the shape of the dataset
print(f"Data shape: {data.shape}")

# Check for missing values
missing_values = data.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Fill NaN values with an empty string for text columns and a default value for 'label'
data.fillna({
    'subject': '',
    'body': '',
    'text_combined': '',
    'sender': '',
    'receiver': '',
    'date': '',
}, inplace=True)

# Convert 'label' to numeric (0 or 1)
data['label'] = data['label'].apply(pd.to_numeric, errors='coerce')

# Preprocess the text columns (e.g., subject, body, text_combined)
def preprocess(text):
    """ Nettoyage du texte : suppression des balises HTML, URLs et caractères spéciaux """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    else:
        return ''  # Return an empty string if not a string

# Combine subject and body to create a unified text column if text_combined is missing
if data['text_combined'].isnull().any():
    data['text_combined'] = data['subject'] + ' ' + data['body']

# Apply text preprocessing on the 'text_combined' column
data['cleaned_text'] = data['text_combined'].apply(preprocess)

# Verify the preprocessed text
print("Sample of preprocessed text:")
print(data[['text_combined', 'cleaned_text']].head())

# Feature Extraction using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the cleaned text to get the feature matrix
X = tfidf_vectorizer.fit_transform(data['cleaned_text'])
print("TF-IDF vectorization complete.")

# Get the target labels
y = data['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all CPU cores for faster training

# Train the model
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, r'phishing_detection_model.pkl')
joblib.dump(tfidf_vectorizer, r'tfidf_vectorizer.pkl')
print("Model and vectorizer saved.")
