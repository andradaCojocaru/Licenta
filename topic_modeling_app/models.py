from django.db import models

class UserSelection(models.Model):
    preprocessing_option = models.CharField(max_length=100)
    encodation_option = models.CharField(max_length=100)
    model_option = models.CharField(max_length=100)

class PertinentWords(models.Model):
    word = models.CharField(max_length=50)
    frequency = models.IntegerField()
