from django.shortcuts import render
from .forms import SongForm, URLForm
from ClassifierApp.main import *
from ClassifierApp.spotify_api import getdata

def index(request):
    if request.method == 'POST':
        url = URLForm(request.POST)
        form = SongForm(getdata(url.data))
        
    else:
        url = URLForm()
        form = SongForm()
    context = {
        'url_form':url,
        'form':form,
        'datasize':len(dataset.index),
        'trainsize':int(len(dataset.index)*0.9),
        'testsize':int(len(dataset.index)*0.1),
        'Accuracy': ['%.2f'%(a*100) for a in acc]
    }
    return render(request, 'home.html', context)

def predict(request):
    if request.method == 'POST':
        form = SongForm(request.POST)
        if form.is_valid():
            mood = predict_mood(form.cleaned_data)
            context = {'mood': mood}
            return render(request, 'result.html', context)