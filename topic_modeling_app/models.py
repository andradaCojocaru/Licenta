from djongo import models

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
    distributed = models.BooleanField()
    onepass = models.BooleanField()
    power_iters = models.IntegerField()
    extra_samples = models.IntegerField()
    dtype = models.CharField(max_length=10)  # Assuming dtype is a string
    random_seed = models.IntegerField()

    def __str__(self):
        return f"LSA Model - {self.num_topics} topics, {self.chunksize} chunksize \
            , {self.decay} decay"

class LdaModel(models.Model):
    num_topics = models.IntegerField()
    chunksize = models.IntegerField()
    decay = models.FloatField()
    gamma_threshold = models.FloatField(null=True, blank=True)
    distributed = models.BooleanField()
    dtype = models.CharField(max_length=10)  # Assuming dtype is a string
    eval_every = models.IntegerField(null=True, blank=True)
    iterations = models.IntegerField(null=True, blank=True)
    gamma_threshold = models.FloatField(null=True, blank=True)
    minimum_probability = models.FloatField(null=True, blank=True)
    random_state = models.IntegerField(null=True, blank=True)
    minimum_phi_value = models.FloatField(null=True, blank=True)
    per_word_topics = models.BooleanField(default=False)
    passes = models.IntegerField(null=True, blank=True)  # Added passes as an integer field
    update_every = models.IntegerField(null=True, blank=True)  # Added update_every as an integer field
    alpha = models.FloatField(null=True, blank=True)  # Added alpha as a float field
    eta = models.FloatField(null=True, blank=True)  # Added eta as a float field

    def __str__(self):
        return f"LDA Model - {self.num_topics} topics, {self.chunksize} chunksize \
            , {self.decay} decay"


    
class PlsaModel(models.Model):
    num_topics = models.IntegerField()
    passes = models.IntegerField()
    # Other fields as needed

    def __str__(self):
        return f"PLSA Model - {self.num_topics} topics, {self.passes} passes"
    
class NmfModel(models.Model):
    num_topics = models.IntegerField()
    chunksize = models.IntegerField()
    passes = models.IntegerField(null=True, blank=True)
    kappa = models.FloatField(null=True, blank=True)
    minimum_probability = models.FloatField(null=True, blank=True)
    w_max_iter = models.IntegerField(null=True, blank=True)
    w_stop_condition = models.FloatField(null=True, blank=True)
    h_max_iter = models.IntegerField(null=True, blank=True)
    h_stop_condition = models.FloatField(null=True, blank=True)
    eval_every = models.IntegerField(null=True, blank=True)
    normalize = models.BooleanField(null=True, blank=True)
    random_state = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"NMF Model - {self.num_topics} topics, {self.chunksize} chunksize"
    
class HdpModel(models.Model):
    max_chunks = models.IntegerField(null=True, blank=True)
    max_time = models.IntegerField(null=True, blank=True)
    chunksize = models.IntegerField(null=True, blank=True)
    kappa = models.FloatField(null=True, blank=True)
    tau = models.FloatField(null=True, blank=True)
    K = models.IntegerField(null=True, blank=True)
    T = models.IntegerField(null=True, blank=True)
    alpha = models.IntegerField(null=True, blank=True)
    gamma = models.IntegerField(null=True, blank=True)
    eta = models.FloatField(null=True, blank=True)
    scale = models.FloatField(null=True, blank=True)
    var_converge = models.FloatField(null=True, blank=True)
    outputdir = models.CharField(max_length=255, null=True, blank=True)
    random_state = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return "HDP Model"
    
class Corpus(models.Model):
    name = models.CharField(max_length=255)
