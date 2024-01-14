from django import forms
from .models import UserSelection, LdaModel, LsaModel

class UserSelectionForm(forms.ModelForm):
    class Meta:
        model = UserSelection
        fields = ['preprocessing_option', 'encodation_option', 'model_option']

class ModelChoiceForm(forms.Form):
    MODEL_CHOICES = [
        ('LSA', 'LSA'),
        ('LDA', 'LDA'),
        ('PLSA', 'PLSA'),
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