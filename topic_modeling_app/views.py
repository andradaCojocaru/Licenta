from datetime import datetime
from django.shortcuts import render, redirect
import pandas as pd
import multiprocessing
from .tasks import train_model_in_child_process
from .forms import ModelChoiceForm, LdaModelForm, LsaModelForm, NmfModelForm, HdpModelForm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
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

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Text preprocessing: stemming and removing spaces
stemmer = PorterStemmer()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

corpus_path = os.path.join(BASE_DIR, 'data', 'corpus')
dictionary_path = os.path.join(BASE_DIR, 'data', 'dictionary')
models_path = os.path.join(BASE_DIR, 'models')

def home(request):
    return render(request, 'home.html')

def get_saved_lsi_model(model_id):
    # Load the LSI model from the saved file
    lsi_model = models.LdaModel.load(os.path.join(models_path, model_id))
    return lsi_model

def lda_visualization(request): 
    model_id = request.session.get('model_id', None)
    text_id = request.session.get('text_id', None)
    lda_model = get_saved_lsi_model(model_id)

    dictionary = corpora.Dictionary.load(os.path.join(dictionary_path, text_id))
    corpus = corpora.MmCorpus(os.path.join(corpus_path, text_id))

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
    # Dictionary mapping model names to their respective form classes
    form_classes = {
        'LDA': LdaModelForm,
        'NMF': NmfModelForm,
        'HDP': HdpModelForm,
        'LSA': LsaModelForm
    }

    # Check if the request method is POST
    if request.method == 'POST':
        # Get the form class corresponding to the selected model
        form_class = form_classes.get(selected_model)

        # If the selected model is not found in the dictionary, raise an exception
        if form_class is None:
            raise ValueError("Invalid selected model")

        # Instantiate the form with POST data
        form = form_class(request.POST)

        # Check if the form is valid
        if form.is_valid():
            # Process the form data and redirect accordingly
            # ...

            return redirect(f'{selected_model}/')
    else:
        # Get the form class corresponding to the selected model
        form_class = form_classes.get(selected_model)

        # If the selected model is not found in the dictionary, raise an exception
        if form_class is None:
            raise ValueError("Invalid selected model")

        # Instantiate the form without POST data
        form = form_class()

    # Render the template with the form and selected model
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

    existing_data = collection.find_one({
        'corpus_data.corpus_name': corpus_name,
        'selected_parameters': selected_parameters
    })

    if existing_data:
        print("Data with the same corpus name, model name, and parameters already exists.")
        return existing_data['model_id'], True, True
    
    # Check if data with the same corpus name exists
    existing_corpus_data = collection.find_one({
        'corpus_data.corpus_name': corpus_name
    })
    text_id = model_id

    if existing_corpus_data:
        print("Data with the same corpus name already exists, but parameters or model name may differ.")
        text_id = existing_corpus_data['text_id']
    
    combined_data = {
        'selected_parameters': selected_parameters,
        'corpus_data': {
            'corpus_name': corpus_name
        },
        'model_id': model_id,
        'text_id': text_id,
        'model_name': model_name 
    }
    # Insert the selected parameters into the collection
    collection.insert_one(combined_data)
    if existing_corpus_data:
        return text_id, False, True
    return model_id, False, False

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
    text_id = model_id
    model_name = request.session.get('model_name') 
    os.makedirs(dictionary_path, exist_ok=True)
    os.makedirs(corpus_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    id, corpus_exists, data_exists = save_to_mongodb(selected_parameters, corpus_name, model_id, model_name)
    request.session['id'] = id
    request.session['corpus_exists'] = corpus_exists
    request.session['data_exists'] = data_exists

    if data_exists:
        model_id = id
        text_id = id
    if corpus_exists:
        text_id = id
    request.session['model_id'] = model_id
    request.session['text_id'] = text_id
    
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
    text_id = request.session.get('text_id')
    corpus_exists = request.session.get('corpus_exists')
    data_exists = request.session.get('data_exists')

    if data_exists == False:
    # Create a dictionary from the processed text
        if corpus_exists == False:
            dictionary = corpora.Dictionary(processed_text)
            dictionary.save(os.path.join(dictionary_path, text_id))

            # Create a bag-of-words representation of the corpus
            corpus_bow = [dictionary.doc2bow(text) for text in processed_text]
            corpora.MmCorpus.serialize(os.path.join(corpus_path, text_id), corpus_bow)
        else:
            dictionary = corpora.Dictionary.load(os.path.join(dictionary_path, text_id))
            corpus_bow = corpora.MmCorpus(os.path.join(corpus_path, text_id))
        selected_parameters = request.session.get('selected_parameters', {})
        params = {
            key: value 
            for key, value in selected_parameters.get('selected_parameters', {}).items() 
            if value is not None and value.strip()
        }
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit the training task to the executor
            future = executor.submit(
                train_model_in_child_process,
                corpus_bow, dictionary, params, model_id, models_path
            )

            # Wait for the task to complete
            return future.result()
    else:
        model = get_saved_lsi_model(model_id)

    return model

def train_button(request):
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