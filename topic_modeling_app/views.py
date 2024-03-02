from datetime import datetime
from django.shortcuts import render, redirect
import pandas as pd
import multiprocessing
from .preprocessing import preprocess_text
from .db import save_to_mongodb
from .tasks import train_model_in_child_process
from .forms import ModelChoiceForm, LdaModelForm, LsiModelForm, NmfModelForm, HdpModelForm, pLsaModelForm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from .models import PertinentWords
from gensim import corpora, models
from gensim.utils import simple_preprocess
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from django.db import connections
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import os

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
        'LSI': LsiModelForm,
        'PLSA': pLsaModelForm,
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

    # Define the mapping between form fields and model parameters
    parameter_mapping = {
        'pLSA': ['max_iters'],
        'LSI': ['num_topics', 'chunksize', 'decay', 'distributed', 'onepass', 'power_iters', 
                'extra_samples', 'dtype', 'random_seed'],
        'HDP': ['max_chunks', 'max_time', 'chunksize', 'kappa', 'tau', 'K', 'T', 'alpha', 
                'gamma', 'eta', 'scale', 'var_converge', 'outputdir', 'random_state'],
        'NMF': ['num_topics', 'chunksize', 'passes', 'kappa', 'minimum_probability', 
                'w_max_iter', 'w_stop_condition', 'h_max_iter', 'h_stop_condition', 
                'eval_every', 'normalize', 'random_state'],
    }

    selected_parameters = {}
    if selected_model in parameter_mapping:
        for parameter_name in parameter_mapping[selected_model]:
            form_field = parameter_name if parameter_name != 'distributed' else 'distributed_checkbox'
            selected_parameters[parameter_name] = request.POST.get(form_field)

    request.session['selected_parameters'] = {'selected_parameters': selected_parameters}
    # save_to_mongodb(selected_parameters)

    # Render the template with the selected parameters
    return render(request, 'selected_parameters_model.html', {'selected_parameters': selected_parameters})

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