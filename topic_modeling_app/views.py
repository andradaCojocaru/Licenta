from django.shortcuts import render, redirect
import pandas as pd
from .forms import UserSelectionForm, ModelChoiceForm, LdaModelForm, LsaModelForm
from .models import PertinentWords
from gensim import corpora, models
from gensim.utils import simple_preprocess
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

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

def lda_visualization(request):
    # Load a smaller dataset of tweets
    dset_url = 'https://archive.org/download/misc-dataset/dp-export-tokenized.csv'
    tweets_df = pd.read_csv(dset_url, nrows=100)  # Load only the first 100 rows
    tweets = tweets_df['Tweets'].values.tolist()
    tweets = [t.split(',') for t in tweets]

    # Example text preprocessing using spaCy for lemmatization
    nlp = spacy.load("en_core_web_sm")

    # Lemmatize each document
    lemmatized_tweets = []
    for tweet in tweets:
        lemmatized_text = []
        for doc in nlp.pipe(tweet, disable=["parser", "ner"]):
            lemmatized_text.extend([token.lemma_ for token in doc if not token.is_stop])
        lemmatized_tweets.append(lemmatized_text)

    # Create a dictionary and a corpus (bag-of-words representation)
    dictionary = corpora.Dictionary(lemmatized_tweets)
    corpus = [dictionary.doc2bow(text) for text in lemmatized_tweets]

    # Train an LDA model with fewer topics and passes
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=5)

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
        else:
            form = LsaModelForm(request.POST)
        if form.is_valid():
            if selected_model == 'LDA':
                selected_model = form.cleaned_data['lda_model']
            elif selected_model == 'LSA':
                selected_model = form.cleaned_data['lsa_model']
            return redirect(f'{selected_model}/')
    else:
        if selected_model == 'LDA':
            form = LdaModelForm(request.POST)
        else:
            form = LsaModelForm(request.POST)
    return render(request, 'models_detail.html', {'form': form, 'selected_model': selected_model})

def selected_parameters(request, selected_model):
    # Get the selected parameters from the submitted form data
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

    # Render the template with the selected parameters
    return render(request, 'selected_parameters_model.html', {'selected_parameters': selected_parameters})

def add_corpus(request):
    # Your view logic here
    return render(request, 'add_corpus.html')