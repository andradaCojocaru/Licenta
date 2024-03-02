# tasks.py
from gensim import corpora, models
import os

def get_model_instance(model_type, corpus_bow, dictionary, params):
    if model_type == "LDA":
        return models.LdaModel(corpus_bow, id2word=dictionary, **params)
    elif model_type == "LSI":
        return models.LsiModel(corpus_bow, id2word=dictionary, **params)
    elif model_type == "PLSA":
        return models.LdaModel(corpus_bow, id2word=dictionary, **params)
    elif model_type == "HDP":
        return models.HdpModel(corpus_bow, id2word=dictionary, **params)
    elif model_type == "NMF":
        return models.NmfModel(corpus_bow, id2word=dictionary, **params)
    else:
        raise ValueError("Invalid model type")
    
def train_model_in_child_process(corpus_bow, dictionary, params, model_id, models_path, model_type):
    model = get_model_instance(model_type, corpus_bow, dictionary, params)
    model.save(os.path.join(models_path, model_id))
    return model
