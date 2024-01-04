from django.db import models

class UserSelection(models.Model):
    preprocessing_option = models.CharField(max_length=100)
    encodation_option = models.CharField(max_length=100)
    model_option = models.CharField(max_length=100)

class PertinentWords(models.Model):
    word = models.CharField(max_length=50)
    frequency = models.IntegerField()

class LsaModel(models.Model):
    num_topics = models.IntegerField()
    chunksize = models.IntegerField()
    decay = models.FloatField()
    #distributed = models.BooleanField()
    # onepass = models.BooleanField()
    # power_iters = models.IntegerField()
    # extra_samples = models.IntegerField()
    # dtype = models.CharField(max_length=10)  # Assuming dtype is a string
    # random_seed = models.IntegerField()

    def __str__(self):
        return f"LSA Model - {self.num_topics} topics, {self.chunksize} chunksize \
            , {self.decay} decay"

class LdaModel(models.Model):
    num_topics = models.IntegerField()
    chunksize = models.IntegerField()
    # Other fields as needed

    def __str__(self):
        return f"LDA Model - {self.num_topics} topics, {self.passes} passes"


    
class PlsaModel(models.Model):
    num_topics = models.IntegerField()
    passes = models.IntegerField()
    # Other fields as needed

    def __str__(self):
        return f"PLSA Model - {self.num_topics} topics, {self.passes} passes"
