from django import forms
from django.forms.widgets import *

class URLForm(forms.Form):
    url= forms.URLField(widget=URLInput(attrs={'placeholder':'Enter Spotify URL To Get Track Details'}))

class SongForm(forms.Form):
    # pitch_mean= forms.FloatField(widget=NumberInput(attrs={ 'value' : 0,'id':'tempo_input', 'onkeyup':"SliderUpdate(this.value,'tempo_slider');",})) 
    # pitch_sd= forms.FloatField(widget=NumberInput(attrs={ 'value' : 0,'id':'tempo_input', 'onkeyup':"SliderUpdate(this.value,'tempo_slider');",}))
    # length= forms.FloatField(widget=NumberInput(attrs={ 'value' : 0,'id':'tempo_input', 'onkeyup':"SliderUpdate(this.value,'tempo_slider');",})) 
    # time_signature= forms.IntegerField() 
    # db_value= forms.FloatField(widget=NumberInput(attrs={ 'value' : 0,'id':'tempo_input', 'onkeyup':"SliderUpdate(this.value,'tempo_slider');",})) 
    # camelot= forms.CharField()
    tempo= forms.FloatField(widget=NumberInput(
        attrs={
            'value' : 0,'id':'input1', 'onkeyup':"SliderUpdate(this.value,'slider1');",
            }
        ))
    acousticness= forms.FloatField(widget=NumberInput(
        attrs={
            'value' : 0,'id':'input2', 'onkeyup':"SliderUpdate(this.value,'slider2');",
            }
        )) 
    danceability= forms.FloatField(widget=NumberInput(
        attrs={
             'value' : 0,'id':'input3', 'onkeyup':"SliderUpdate(this.value,'slider3');",
            }
        )) 
    energy= forms.FloatField(widget=NumberInput(
        attrs={
            
             'value' : 0,'id':'input4', 'onkeyup':"SliderUpdate(this.value,'slider4');",
            }
        )) 
    instrumentalness= forms.FloatField(widget=NumberInput(
        attrs={
             'value' : 0,'id':'input5', 'onkeyup':"SliderUpdate(this.value,'slider5');",
            }
        ))
    liveness= forms.FloatField(widget=NumberInput(
        attrs={
             'value' : 0,'id':'input6', 'onkeyup':"SliderUpdate(this.value,'slider6');",
            }
        )) 
    loudness= forms.FloatField(widget=NumberInput(
        attrs={
            'value' : -100,'id':'input7', 'onkeyup':"SliderUpdate(this.value,'slider7');",
            }
        ))
    speechiness= forms.FloatField(widget=NumberInput(
        attrs={
             'value' : 0,'id':'input8', 'onkeyup':"SliderUpdate(this.value,'slider8');",
            }
        ))
    valence= forms.FloatField(widget=NumberInput(
        attrs={
               'value' : 0,'id':'input9', 'onkeyup':"SliderUpdate(this.value,'slider9');",
            }
        ))
    ch = [('KNN','KNN'),('GNB','GNB'),('SVM','SVM')]
    classifier = forms.ChoiceField(widget=RadioSelect() , choices=ch)