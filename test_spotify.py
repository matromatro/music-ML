import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from pathlib import Path

# Load env
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

print(f"Testing with URI: {redirect_uri}")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-top-read",
    open_browser=True
))

try:
    print("User:", sp.current_user()["id"])
    print("Testing recommendations for 'The Beatles' (3WrFJ7ztbogyGnTHbHJFl2)...")
    res = sp.recommendations(seed_artists=["3WrFJ7ztbogyGnTHbHJFl2"], limit=1)
    if res['tracks']:
        print("✅ Correctly received track:", res['tracks'][0]['name'])
    else:
        print("⚠️ Received empty tracks list.")
except Exception as e:
    print("🔥 Error:", e)
