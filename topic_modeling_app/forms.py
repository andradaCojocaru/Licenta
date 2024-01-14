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
            'num_topics': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'chunksize': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'decay': forms.NumberInput(attrs={'placeholder': 'Optional'}),
            'gamma_threshold': forms.NumberInput(attrs={'step': 'any'}),
            'dtype': forms.Select(choices=[('float32', 'float32'), ('float64', 'float64')]),  # Adjust choices based on your needs
            # Add more widgets for other fields if needed
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