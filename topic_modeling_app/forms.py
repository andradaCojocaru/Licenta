from django import forms
from .models import UserSelection, LdaModel, LsaModel, HdpModel, NmfModel

class UserSelectionForm(forms.ModelForm):
    class Meta:
        model = UserSelection
        fields = ['preprocessing_option', 'encodation_option', 'model_option']

class ModelChoiceForm(forms.Form):
    MODEL_CHOICES = [
        ('LSA', 'LSA'),
        ('LDA', 'LDA'),
        #('PLSA', 'PLSA'),
        ('NMF', 'NMF'),
        ('HDP', 'HDP'),
    ]

    model_choice = forms.ChoiceField(choices=MODEL_CHOICES)

class LsaModelForm(forms.ModelForm):
    class Meta:
        model = LsaModel
        fields = '__all__'
        widgets = {
            'corpus': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Stream of document vectors or a sparse matrix of shape (num_documents, num_terms)'}),
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of requested factors (latent dimensions)'}),
            'id2word': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'ID to word mapping'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of documents to be used in each training chunk'}),
            'decay': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Weight of existing observations relatively to new ones'}),
            'distributed': forms.CheckboxInput(attrs={'title': 'If True - distributed mode (parallel execution on several machines) will be used'}),
            'onepass': forms.CheckboxInput(attrs={'title': 'Whether the one-pass algorithm should be used for training. Pass False to force a multi-pass stochastic algorithm'}),
            'power_iters': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of power iteration steps to be used. Increasing the number of power iterations improves accuracy, but lowers performance'}),
            'extra_samples': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Extra samples to be used besides the rank k. Can improve accuracy'}),
            'dtype': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Enforces a type for elements of the decomposed matrix'}),
            'random_seed': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Random seed used to initialize the pseudo-random number generator, a local instance of numpy.random.RandomState instance'}),
        }


    def __init__(self, *args, **kwargs):
        super(LsaModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False

class LdaModelForm(forms.ModelForm):
    class Meta:
        model = LdaModel
        fields = '__all__'
        widgets = {
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'The number of requested latent topics to be extracted from the training corpus'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of documents to be used in each training chunk'}),
            'passes': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of passes through the corpus during training'}),
            'update_every': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning'}),
            'alpha': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'A-priori belief on document-topic distribution. Scalar for a symmetric prior over document-topic distribution, 1D array of length equal to num_topics to denote an asymmetric user-defined prior for each topic, or default prior selecting strategies'}),
            'eta': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'A-priori belief on topic-word distribution. Scalar for a symmetric prior over topic-word distribution, 1D array of length equal to num_words to denote an asymmetric user-defined prior for each word, matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination, or default prior selecting strategies'}),
            'decay': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined'}),
            'offset': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Hyper-parameter that controls how much the first steps are slowed down in the first few iterations'}),
            'eval_every': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x'}),
            'iterations': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Maximum number of iterations through the corpus when inferring the topic distribution of a corpus'}),
            'gamma_threshold': forms.NumberInput(attrs={'step': 'any', 'title': 'Minimum change in the value of the gamma parameters to continue iterating'}),
            'minimum_probability': forms.NumberInput(attrs={'step': 'any', 'title': 'Topics with a probability lower than this threshold will be filtered out'}),
            'random_state': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Seed to generate a randomState object for reproducibility'}),
            'dtype': forms.Select(choices=[('float16', 'float16'), ('float32', 'float32'), ('float64', 'float64')], attrs={'title': 'Data type to use during calculations inside the model'}),
        }

    def __init__(self, *args, **kwargs):
        super(LdaModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False

        # Set other fields as not required if needed

class HdpModelForm(forms.ModelForm):
    class Meta:
        model = HdpModel
        fields = '__all__'
        widgets = {
            'corpus': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Corpus in BoW format'}),
            'id2word': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Dictionary for the input corpus'}),
            'max_chunks': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Upper bound on how many chunks to process. It wraps around corpus beginning in another corpus pass, if there are not enough chunks in the corpus'}),
            'max_time': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Upper bound on time (in seconds) for which model will be trained'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of documents in one chunk'}),
            'kappa': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Learning parameter which acts as exponential decay factor to influence extent of learning from each batch'}),
            'tau': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Learning parameter which down-weights early iterations of documents'}),
            'K': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Second level truncation level'}),
            'T': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Top level truncation level'}),
            'alpha': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Second level concentration'}),
            'gamma': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'First level concentration'}),
            'eta': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'The topic Dirichlet'}),
            'scale': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Weights information from the mini-chunk of corpus to calculate rhot'}),
            'var_converge': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Lower bound on the right side of convergence. Used when updating variational parameters for a single document'}),
            'outputdir': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Stores topic and options information in the specified directory'}),
            'random_state': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Adds a little random jitter to randomize results around same alpha when trying to fetch a closest corresponding lda model from suggested_lda_model()'}),
        }


    def __init__(self, *args, **kwargs):
        super(HdpModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False
        # Set other fields as not required if needed
            
class NmfModelForm(forms.ModelForm):
    class Meta:
        model = NmfModel
        fields = '__all__'
        widgets = {
            'corpus': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Training corpus. Can be either iterable of documents, which are lists of (word_id, word_count), or a sparse csc matrix of BOWs for each document. If not specified, the model is left uninitialized'}),
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of topics to extract'}),
            'id2word': forms.TextInput(attrs={'placeholder': 'Optional', 'title': 'Mapping from word IDs to words. Used to determine the vocabulary size, as well as for debugging and topic printing'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of documents to be used in each training chunk'}),
            'passes': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of full passes over the training corpus'}),
            'kappa': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Gradient descent step size. Larger value makes the model train faster, but could lead to non-convergence if set too large'}),
            'minimum_probability': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'If normalize is True, topics with smaller probabilities are filtered out. If normalize is False, topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s'}),
            'w_max_iter': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Maximum number of iterations to train W per each batch'}),
            'w_stop_condition': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'If error difference gets less than that, training of W stops for the current batch'}),
            'h_max_iter': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Maximum number of iterations to train h per each batch'}),
            'h_stop_condition': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'If error difference gets less than that, training of h stops for the current batch'}),
            'eval_every': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if set too low'}),
            'normalize': forms.CheckboxInput(attrs={'title': 'Whether to normalize the result. Allows for estimation of perplexity, coherence, etc.'}),
            'random_state': forms.NumberInput(attrs={'placeholder': 'Optional', 'title': 'Seed for random generator. Needed for reproducibility'}),
        }


    def __init__(self, *args, **kwargs):
        super(NmfModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False
        # Set other fields as not required if needed