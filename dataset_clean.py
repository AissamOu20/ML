import pandas as pd
import os
import re

# Define the path to the directory containing your CSV files
data_dir = r'dataset'  # Update with your actual path

# List to hold DataFrames
dataframes = []

# Load each CSV file in the directory
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Append the DataFrame to the list
        dataframes.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Display the first few rows of the combined DataFrame
print("Combined DataFrame:")
print(combined_df.head())
print(combined_df.columns)
print(combined_df.dtypes)

# Data Cleaning
# Remove duplicates
combined_df.drop_duplicates(inplace=True)

# Handle missing values
combined_df.fillna(value={
    'subject': '',           # Remplir les valeurs manquantes dans 'subject' par une chaîne vide
    'body': '',              # Remplir les valeurs manquantes dans 'body' par une chaîne vide
    'label': -1,             # Valeur par défaut pour 'label'
    'sender': '',            # Remplir les valeurs manquantes dans 'sender'
    'receiver': '',          # Remplir les valeurs manquantes dans 'receiver'
    'date': '',              # Remplir les valeurs manquantes dans 'date'
    'urls': 0.0,             # Valeur par défaut pour 'urls'
    'text_combined': ''      # Remplir les valeurs manquantes dans 'text_combined'
}, inplace=True)

# Check if 'text_combined' column is important, if yes, fill it with 'subject' + 'body'
if combined_df['text_combined'].isnull().any():
    combined_df['text_combined'] = combined_df['text_combined'].fillna(combined_df['subject'] + ' ' + combined_df['body'])

# Clean the text columns ('subject', 'body', and 'text_combined') by applying text preprocessing
def preprocess(text):
    """ Nettoyage du texte : suppression des balises HTML, URLs et caractères spéciaux. """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Supprimer les balises HTML
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Supprimer les URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Garder que les lettres et les espaces
        text = re.sub(r'\s+', ' ', text).strip()  # Enlever les espaces multiples
        return text
    else:
        return ''  # Retourner une chaîne vide si le texte n'est pas une chaîne

# Appliquer le nettoyage des textes sur les colonnes 'subject', 'body', et 'text_combined'
combined_df['subject'] = combined_df['subject'].apply(preprocess)
combined_df['body'] = combined_df['body'].apply(preprocess)
combined_df['text_combined'] = combined_df['text_combined'].apply(preprocess)

# Basic Analysis
# Count the number of occurrences of each label
label_counts = combined_df['label'].value_counts()
print("\nLabel Counts:")
print(label_counts)

# Feature Extraction
# Add a new column for the length of the email body
combined_df['body_length'] = combined_df['body'].apply(len)

# Display the first few rows of the updated DataFrame
print("\nUpdated DataFrame with body length:")
print(combined_df.head())

# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = os.path.join(data_dir, 'cleaned_data.csv')
combined_df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")
