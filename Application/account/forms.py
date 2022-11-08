from django.contrib.auth.forms import UserCreationForm
from django.forms import forms
from .models import User
from django import forms

class RegisterForm(UserCreationForm):
    '''
    A form class that created for designing the form of registering
    
    '''

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "first_name", "last_name")


class DocumentForm(forms.Form):
    '''
    A form class that created for designing the form for the client to fill in the document title, content, tags, and level
    and then submit to the server
    
    '''
    title = forms.CharField()
    tag = forms.ChoiceField(choices= (
        ("A", "Articles"),
        ("S", "Study"), 
        ("E", "Entertainment"),
        ("J", "Journal"), 
        ("N", "Novel"),
    ))
    level = forms.ChoiceField(choices=(
        ("H", "High"),
        ("M", "Medium"),
        ("L", "Low"),
    ))

    documents = forms.CharField(widget=forms.Textarea)

