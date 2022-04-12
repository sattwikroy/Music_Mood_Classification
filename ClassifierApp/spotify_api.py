import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def getdata(url_data):
    cid= '4301ed06ff4e4f7eab666ab9913f3221'
    secret = '4147648f43604287a15d6ed62835b4a1'
    client = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client)
    track_uri = url_data['url'] 
    #'https://open.spotify.com/track/5pyByucxNArkCMapbosKaP'
    return sp.audio_features(track_uri)[0]