from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse, response
import torch
import sys
import time

from .recognition import model_decode
from .models import User,TextDocument
from .forms import DocumentForm, RegisterForm
from django.contrib.auth.decorators import login_required


def register(request):

    '''
    The view function register handles the logic of user registering for Geass Speech by process on the request and 
    return the corresponding response

    '''

    #only post request means registration
    redirect_to = request.POST.get('next',request.GET.get('next',''))
    if request.method == 'POST':
        #initialize the form instance
        form = RegisterForm(request.POST)

        #validate
        if form.is_valid():
            form.save()

            messages.success(request, "Successfully Registered")
            if redirect_to:
                return redirect(redirect_to)

            # return redirect('index')
            return redirect('login')

    else:
        #if the request is not post
        form = RegisterForm()

    return render(request, 'register.html', context = {'form':form, 'next': redirect_to, 'messages':None})

# def index(request):
#     return render(request, 'index.html')

@login_required
def profile_view(request):

    '''
    The view function profile_view handles the page when the user made to login and display the user's profile on 
    page and save user's new edited documents to the database conntected thourgh POST request

    '''
    profile = User.objects.get(username = request.user)

    form = DocumentForm()
    context = {
        'profile':profile,
        'form':form    
    }
    if request.method == 'POST':
        form = DocumentForm(request.POST)
        print(form)

    
        if form.is_valid():
            document = TextDocument()
            document.text_title = request.POST['title']
            document.text_tag = request.POST["tag"]
            document.text_level = request.POST["level"]
            document.text_content = request.POST["documents"]
            document.user = profile
            document.save()

            return render(request, 'profiles/profile.html',context)

        
    

    return render(request, 'profiles/profile.html', context)



@login_required
def speech(request):
    '''
    The function migrates the functionality of model predicting the speech into practical utilization as application 
    by loading the optimized model and call the implemented decoders to do the decoding

    '''
    path = sys.path[0] + '/account/recognition'
    print(path)
    ASR_model = torch.load(path + '/retrained_model_20.pth', map_location = torch.device('cpu'))
    speech_decoder = model_decode.SpeechDecoder()
    text = speech_decoder.Decode(ASR_model)
    print(text)

    context = {
        'text': text
    }

    return HttpResponse(text)
    
@login_required
def documents_view(request):

    '''
    The view function docuemtns_view retrieve the objects of Text Document under login user and display 
    all the documents belong to the user

    '''
    
    queryset = TextDocument.objects.all().filter(user = request.user)
    context = {
        'documents':queryset
    }
    return render(request, 'profiles/documents.html', context)




