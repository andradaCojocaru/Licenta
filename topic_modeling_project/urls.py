from django.contrib import admin
from django.urls import path
from topic_modeling_app.views import lda_visualization, home, choose_model, model_detail,\
      selected_parameters, add_corpus, process_corpus, train_button

urlpatterns = [
    path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('select/', choose_model, name='select_options'),
    path('select/<str:selected_model>/', model_detail, name='model_detail'),
    path('selected_parameters/<str:selected_model>/', selected_parameters, name='selected_parameters'),
    path('add_corpus/', add_corpus, name='add_corpus'),
    path('process_corpus/', process_corpus, name='process_corpus'),
    path('train_button/', train_button, name='train_button'),
    path('lda-visualization/', lda_visualization, name='lda_visualization'),
]
