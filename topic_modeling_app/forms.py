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
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'decay': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'distributed': forms.CheckboxInput(),
            'onepass': forms.CheckboxInput(),
            'power_iters': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'extra_samples': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'dtype': forms.TextInput(attrs={'placeholder': 'Optional'}),
            'random_seed': forms.NumberInput(attrs={'placeholder': 'Optional'}),
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
            'max_chunks': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'max_time': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'kappa': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'tau': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'K': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'T': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'alpha': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'gamma': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'eta': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'scale': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'var_converge': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'outputdir': forms.TextInput(attrs={'placeholder': 'Optional'}),
            'random_state': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            # Add more widgets for other fields if needed
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
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'passes': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'kappa': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'minimum_probability': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'w_max_iter': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'w_stop_condition': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'h_max_iter': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'h_stop_condition': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'eval_every': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'normalize': forms.CheckboxInput(),
            'random_state': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            # Add more widgets for other fields if needed
        }

    def __init__(self, *args, **kwargs):
        super(NmfModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False
        # Set other fields as not required if needed