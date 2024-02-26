import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Text preprocessing: stemming and removing spaces
stemmer = PorterStemmer()

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    
    # Tokenize and stem each word, exclude short words, and remove stopwords
    tokens = [
        stemmer.stem(word) 
        for word in word_tokenize(text) 
        if word.isalpha() and len(word) > 2 and word not in stop_words
    ]
    
    # Join the stemmed tokens back into a single string
    return tokens