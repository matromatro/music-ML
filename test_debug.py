import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

print("--- DEBUG SPOTIFY ---")
cid = os.getenv("SPOTIPY_CLIENT_ID") or input("Client ID: ").strip()
sec = os.getenv("SPOTIPY_CLIENT_SECRET") or input("Client Secret: ").strip()

os.environ["SPOTIPY_CLIENT_ID"] = cid
os.environ["SPOTIPY_CLIENT_SECRET"] = sec
os.environ["SPOTIPY_REDIRECT_URI"] = "http://127.0.0.1:8888/callback"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="user-top-read",
    open_browser=True
))

print("getting user...")
user = sp.current_user()
print(f"Logged in as {user['id']}")

print("Testing recommendations...")
try:
    # Use the same seed that appeared in the log
    seeds = ["3qm84nBOXUEQ2vnTfUTTFC", "4Cqia9vrAbm7ANXbJGXsTE"] 
    print(f"Seeds: {seeds}")
    
    res = sp.recommendations(seed_artists=seeds, limit=5)
    print("Success! Found", len(res['tracks']), "tracks.")
    print("First track:", res['tracks'][0]['name'])
except Exception as e:
    print("FAIL:", e)
    import traceback
    traceback.print_exc()
