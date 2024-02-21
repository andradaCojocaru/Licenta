# tasks.py
from gensim import corpora, models
import os

def train_model_in_child_process(corpus_bow, dictionary, params, model_id, models_path):
    model = models.LdaModel(corpus_bow, id2word=dictionary, **params)
    model.save(os.path.join(models_path, model_id))
    return model
