from django.urls import path, include
from . import views


#URLConf

urlpatterns = [
    path('register/', views.register, name = 'register'),
    path('profile/', views.profile_view, name = 'profile'),
    path('profile/speech/', views.speech, name = "speech"),
    path('profile/documents/', views.documents_view, name = "documents")
    #path point to the view function that handles 
]
