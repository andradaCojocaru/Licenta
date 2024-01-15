from datetime import datetime
from django.shortcuts import render, redirect
import pandas as pd
from .forms import UserSelectionForm, ModelChoiceForm, LdaModelForm, LsaModelForm, NmfModelForm, HdpModelForm

from .models import PertinentWords
from gensim import corpora, models
from gensim.utils import simple_preprocess
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from django.db import connections
from pymongo import MongoClient
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import os
# import random
# import string

# def generate_random_string(length):
#     # Define the characters to choose from (letters and digits)
#     characters = string.ascii_letters + string.digits

#     # Generate a random string of the specified length
#     random_string = ''.join(random.choice(characters) for _ in range(length))

#     return random_string


# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Text preprocessing: stemming and removing spaces
stemmer = PorterStemmer()
corpus_path = "C:\\Licenta\\topic_modeling_project\\data\\corpus"
dictionary_path = "C:\\Licenta\\topic_modeling_project\\data\\dictionary"

def home(request):
    return render(request, 'home.html')

def select_options(request):
    if request.method == 'POST':
        form = UserSelectionForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('result')
    else:
        form = UserSelectionForm()

    return render(request, 'select_options.html', {'form': form})

def bar_graph(request):
    # Dummy data for demonstration
    labels = ['word1', 'word2', 'word3', 'word4']
    data = [10, 20, 15, 25]

    context = {
        'labels': labels,
        'data': data,
    }

    return render(request, 'bar_graph.html', context)

def topic_circles(request):
    topics = [
        {'name': 'Topic 1', 'details': 'Details for Topic 1'},
        {'name': 'Topic 2', 'details': 'Details for Topic 2'},
        {'name': 'Topic 3', 'details': 'Details for Topic 3'},
        # Add more topics as needed
    ]

    context = {
        'topics': topics,
    }

    return render(request, 'topic_circles.html', context)

def get_saved_lsi_model(model_path, model_id):
    # Load the LSI model from the saved file
    lsi_model = models.LdaModel.load(os.path.join(model_path, model_id))
    return lsi_model

def lda_visualization(request):
    #Load a smaller dataset of tweets
    # dset_url = 'https://archive.org/download/misc-dataset/dp-export-tokenized.csv'
    # tweets_df = pd.read_csv(dset_url, nrows=100)  # Load only the first 100 rows
    # tweets = tweets_df['Tweets'].values.tolist()
    # tweets = [t.split(',') for t in tweets]

    # # Example text preprocessing using spaCy for lemmatization
    # nlp = spacy.load("en_core_web_sm")

    # # Lemmatize each document
    # lemmatized_tweets = []
    # for tweet in tweets:
    #     lemmatized_text = []
    #     for doc in nlp.pipe(tweet, disable=["parser", "ner"]):
    #         lemmatized_text.extend([token.lemma_ for token in doc if not token.is_stop])
    #     lemmatized_tweets.append(lemmatized_text)

    # # Create a dictionary and a corpus (bag-of-words representation)
    # dictionary = corpora.Dictionary(lemmatized_tweets)
    # corpus = [dictionary.doc2bow(text) for text in lemmatized_tweets]

    # # Train an LDA model with fewer topics and passes
    # lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)

    # model_path = "models"
    # os.makedirs(model_path, exist_ok=True)
    # lda_model.save(os.path.join(model_path, 'lsi_model'))   

    model_id = request.session.get('model_id', None)
    lda_model = get_saved_lsi_model("models", model_id)

    dictionary = corpora.Dictionary.load(os.path.join(dictionary_path, model_id))
    corpus = corpora.MmCorpus(os.path.join(corpus_path, model_id))

    # Create the pyLDAvis visualization
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    vis_html = pyLDAvis.prepared_data_to_html(vis_data)

    return render(request, 'lda_visualization.html', {'vis_html': vis_html})

def choose_model(request):
    if request.method == 'POST':
        form = ModelChoiceForm(request.POST)
        if form.is_valid():
            selected_model = form.cleaned_data['model_choice']
            return redirect(f'{selected_model}/')
    else:
        form = ModelChoiceForm()

    return render(request, 'select_options.html', {'form': form})

def model_detail(request, selected_model):
    # Implement logic to display details for the selected model
    if request.method == 'POST':
        if selected_model == 'LDA':
            form = LdaModelForm(request.POST)
        elif selected_model == 'NMF':
            form = NmfModelForm(request.POST)
        elif selected_model == 'HDP':
            form = HdpModelForm(request.POST)
        elif selected_model == 'LSA':
            form = LsaModelForm(request.POST)
        else:
            # Handle other models or raise an exception for unsupported models
            raise ValueError("Invalid selected model")

        if form.is_valid():
            # Process the form data and redirect accordingly
            # ...

            return redirect(f'{selected_model}/')
    else:
        if selected_model == 'LDA':
            form = LdaModelForm()
        elif selected_model == 'NMF':
            form = NmfModelForm()
        elif selected_model == 'HDP':
            form = HdpModelForm()
        elif selected_model == 'LSA':
            form = LsaModelForm()
        else:
            # Handle other models or raise an exception for unsupported models
            raise ValueError("Invalid selected model")
    return render(request, 'models_detail.html', {'form': form, 'selected_model': selected_model})

def selected_parameters(request, selected_model):
    # Get the selected parameters from the submitted form data
    request.session['model_name'] = selected_model
    if selected_model == 'LSA':
        selected_parameters = {
            'num_topics': request.POST.get('num_topics'),
            'chunksize': request.POST.get('chunksize'),
            'decay': request.POST.get('decay'),
            'distributed': request.POST.get('distributed'),
            'onepass': request.POST.get('onepass'),
            'power_iters': request.POST.get('power_iters'),
            'extra_samples': request.POST.get('extra_samples'),
            'dtype': request.POST.get('dtype'),
            'random_seed': request.POST.get('random_seed'),
        }
    elif selected_model == 'HDP':
        selected_parameters = {
            'max_chunks': request.POST.get('max_chunks'),
            'max_time': request.POST.get('max_time'),
            'chunksize': request.POST.get('chunksize'),
            'kappa': request.POST.get('kappa'),
            'tau': request.POST.get('tau'),
            'K': request.POST.get('K'),
            'T': request.POST.get('T'),
            'alpha': request.POST.get('alpha'),
            'gamma': request.POST.get('gamma'),
            'eta': request.POST.get('eta'),
            'scale': request.POST.get('scale'),
            'var_converge': request.POST.get('var_converge'),
            'outputdir': request.POST.get('outputdir'),
            'random_state': request.POST.get('random_state'),
        }
    elif selected_model == 'NMF':
        selected_parameters = {
            'num_topics': request.POST.get('num_topics'),
            'chunksize': request.POST.get('chunksize'),
            'passes': request.POST.get('passes'),
            'kappa': request.POST.get('kappa'),
            'minimum_probability': request.POST.get('minimum_probability'),
            'w_max_iter': request.POST.get('w_max_iter'),
            'w_stop_condition': request.POST.get('w_stop_condition'),
            'h_max_iter': request.POST.get('h_max_iter'),
            'h_stop_condition': request.POST.get('h_stop_condition'),
            'eval_every': request.POST.get('eval_every'),
            'normalize': request.POST.get('normalize'),
            'random_state': request.POST.get('random_state'),
        }
    else :
        selected_parameters = {
            'num_topics': request.POST.get('num_topics'),
            'chunksize': request.POST.get('chunksize'),
            'decay': request.POST.get('decay'),
            'gamma_threshold': request.POST.get('gamma_threshold'),
            'eval_every': request.POST.get('eval_every'),
            'iterations': request.POST.get('iterations'),
            'random_state': request.POST.get('random_state'),
            'dtype': request.POST.get('dtype'),
            'alpha': request.POST.get('alpha'),
        }
    
    request.session['selected_parameters'] = {'selected_parameters': selected_parameters}
    #save_to_mongodb(selected_parameters)

    # Render the template with the selected parameters
    return render(request, 'selected_parameters_model.html', {'selected_parameters': selected_parameters})

def save_to_mongodb(selected_parameters, corpus_name, model_id, model_name):
    # Use Djongo's database connection
    client = MongoClient('mongodb+srv://andradacojocaru:andrada@cluster0.rpknlzf.mongodb.net/')  # Replace 'connection_string' with your actual connection string
    db = client['topic_modelling']  # Replace 'db_name' with your actual database name

    # Choose or create a collection in your database
    collection = db['combined_topic_model']  # Replace 'selected_parameters_collection' with your actual collection name
    combined_data = {
        'selected_parameters': selected_parameters,
        'corpus_data': {
            'corpus_name': corpus_name
        },
        'model_id': model_id,
        'model_name': model_name 
    }
    # Insert the selected parameters into the collection
    collection.insert_one(combined_data)

def add_corpus(request):
    
    # Process the form submission and save data to MongoDB or perform other actions

    # Assuming you have retrieved the values from the form
    corpus_name = request.POST.get('corpus_name')
    #preprocessing_option = request.POST.get('preprocessing_option')
    

    # You can pass these values back to the template
    return render(request, 'add_corpus.html')
    
    # If it's a GET request, just render the form
    


def process_corpus(request):
    # Process the form submission and save data to MongoDB or perform other actions
    corpus_name = request.POST.get('corpus_name')
    preprocessing_option = request.POST.get('preprocessing_option')
    selected_parameters = request.session.get('selected_parameters', {}) 
    model_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    request.session['model_id'] = model_id
    model_name = request.session.get('model_name') 
    os.makedirs(dictionary_path, exist_ok=True)
    os.makedirs(corpus_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    save_to_mongodb(selected_parameters, corpus_name, model_id, model_name)

    # Assuming you have other parameters for the model
    # Retrieve them from the database or any other source as needed
    # ...

    # Pass the data to the template for the new page
    return render(request, 'process_corpus.html', {'corpus_name': corpus_name, 'preprocessing_option': preprocessing_option, \
                                                   'selected_parameters' : selected_parameters})

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

def train_lsi_model(request):
    # Fetch the 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Tokenize and preprocess the text
    processed_text = [preprocess_text(text) for text in newsgroups.data]
    model_id = request.session.get('model_id')


    # Create a dictionary from the processed text
    dictionary = corpora.Dictionary(processed_text)
    dictionary.save(os.path.join(dictionary_path, model_id))

    # Create a bag-of-words representation of the corpus
    corpus_bow = [dictionary.doc2bow(text) for text in processed_text]
    corpora.MmCorpus.serialize(os.path.join(corpus_path, model_id), corpus_bow)
    selected_parameters = request.session.get('selected_parameters', {})
    params = {
        key: value 
        for key, value in selected_parameters.get('selected_parameters', {}).items() 
        if value is not None and value.strip()
    }
    #num_topics = selected_parameters.get('selected_parameters', {}).get('num_topics', None)
    #dictionary_as_list = list(dictionary.items())

    model = models.LdaModel(corpus_bow, id2word=dictionary, **params)

    # Save the trained model (optional)
    model_path = "models"
    #os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, model_id))

    return model
def train_lsi_button(request):
    trained = False
    # Check if the button is clicked (POST request)
    if request.method == 'POST' and 'train_button' in request.POST:
        # Train the LSI model
        trained_model = train_lsi_model(request)

        # You can add further logic or pass information to the template if needed
        if trained_model == None:
            trained = False
        else:
            trained = True

        return render(request, 'train_model.html', {'trained': trained})

    return render(request, 'train_model.html', {'trained': trained})