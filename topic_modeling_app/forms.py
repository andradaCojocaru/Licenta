from django import forms
from .models import UserSelection, LdaModel

class UserSelectionForm(forms.ModelForm):
    class Meta:
        model = UserSelection
        fields = ['preprocessing_option', 'encodation_option', 'model_option']

class ModelChoiceForm(forms.Form):
    MODEL_CHOICES = [
        ('LSA', 'LSA'),
        ('LDA', 'LDA'),
        ('PLSA', 'PLSA'),
    ]

    model_choice = forms.ChoiceField(choices=MODEL_CHOICES)

class LdaModelForm(forms.ModelForm):
    class Meta:
        model = LdaModel
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
        super(LdaModelForm, self).__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.required = False