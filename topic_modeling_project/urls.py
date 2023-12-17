from django.contrib import admin
from django.urls import path
from topic_modeling_app.views import select_options, bar_graph, topic_circles, lda_visualization, home

urlpatterns = [
    path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('select/', select_options, name='select_options'),
    path('bar_graph/', bar_graph, name='bar_graph'),
    path('topic_circles/', topic_circles, name='topic_circles'),
    path('lda-visualization/', lda_visualization, name='lda_visualization'),
]
