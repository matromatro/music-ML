# music-ML
Create an ML to create playlists and recommend new songs/bands


🎧 music-ML — Personalized Multi-Source Music Intelligence
🧭 Project goal

Build an intelligent, self-learning system that:

Analyzes your entire music history (starting from your Apple Music export).

Normalizes and learns your listening taste across platforms.

Creates automatic playlists and synchronizes them to Spotify.

Recommends new artists and songs — both familiar and completely new.

Learns continuously from your feedback and evolving habits.

Apple Music XML  →  load_library.py
                       ↓
genre normalization  →  genre_classifier.py
                       ↓
analysis + playlists →  analysis.py
                       ↓
Spotify sync         →  apple_to_spotify.py
                       ↓
recommendations      →  recommender.py
                       ↓
user ratings loop    →  updates taste model


music-ML/
│
├── main.py
├── load_library.py
├── genre_classifier.py
├── analysis.py
├── apple_to_spotify.py
├── recommender.py
├── requirements.txt
│
├── data/
│   ├── Biblioteca.xml
│   ├── library_clean.csv
│   ├── api_cache.json
│   ├── genre_review.csv
│   ├── user_ratings.csv
│   ├── spotify_export_log.json
│   ├── taste_model.json
│   ├── output/
│   │    ├── playlists/
│   │    └── charts/
│   └── external/
│        ├── soundcloud_likes.csv
│        └── shazam_history.csv
│
└── PROJECT_OVERVIEW.txt


🧩 Script summaries
1️⃣ load_library.py

Purpose:
Parse and clean your Apple Music (or any other) export into a unified format.

Functions:

Loads Biblioteca.xml with plistlib.

Normalizes columns (Name, Artist, Genre, Play Count, Loved, Date Added, etc.).

Removes non-music entries (e.g., Nova Gravação).

Adds source and mode tags (Source='Apple', Mode='past').

Outputs data/library_clean.csv.

Future:
Will also load data from Spotify, SoundCloud, and Shazam for a unified library.

2️⃣ genre_classifier.py

Purpose:
Normalize messy genre labels into umbrella categories using a hybrid system:

Regex + fuzzy rules

MusicBrainz / Last.fm API enrichment

Sentence-Transformer embeddings

Features:

Caches API responses (api_cache.json).

Logs low-confidence cases for manual correction (genre_review.csv).

Learns new mappings via manual updates.

Provides a class GenreClassifier for reuse across the project.

3️⃣ analysis.py

Purpose:
Explore and visualize your library.

Functions:

Calculates genre and artist statistics.

Computes “preference scores” based on plays, recency, and likes.

Generates plots (top genres, top artists, taste evolution).

Builds playlist CSVs by genre or by ML clustering (vibes).

Saves results to data/output/playlists/.

Future option:
Automatic playlist creation on Spotify through the API.

4️⃣ apple_to_spotify.py

Purpose:
Sync Apple Music playlists to your Spotify account.

Functions:

Reads CSV playlists from analysis.py.

Authenticates through Spotify OAuth (using Spotipy).

Creates one Spotify playlist per CSV.

Searches each track (name + artist) and adds it automatically.

Logs everything in spotify_export_log.json.

Future:
Can be adapted later for Spotify → Apple direction if desired.


5️⃣ recommender.py

Purpose:
Generate new music and artist recommendations — both familiar and exploratory — and learn from your feedback.

Modes:

Familiar: “More of what you like.”

Explore: “Something new but musically coherent.”

Mixed: Combines both dynamically.

Loop:

Builds a taste vector per genre from your listening data.

Fetches candidates via Spotify / Last.fm APIs.

Filters out songs you already know.

Suggests one new artist or song per genre.

Saves to recommendations.csv.

You rate them (1–5) in user_ratings.csv.

Model updates taste_model.json accordingly.

Future:

Auto-create a “New This Week” playlist on Spotify.

Use ratings to refine exploration/exploitation balance.

Integrate new data sources (SoundCloud / Shazam).

🎵 Additional integrations
Source	Data	Purpose
SoundCloud	Likes, playlists, mixes	Expands discovery into underground/independent artists
Shazam	Your “identified” tracks	Captures spontaneous curiosity moments
Spotify	Live listening data	Serves as active, up-to-date taste profile
Apple Music	Historical data	Foundation of your long-term preferences

All sources feed into a unified library with columns:
Source, Name, Artist, Genre_norm, Play_Count, Loved, Date_Added, Mode (past/current).

🧠 Learning and recommendation logic

Build embeddings for genres and artists using sentence-transformers.

Compute a taste vector for each umbrella genre (average of liked songs).

Fetch external candidates (similar artists/tracks) via APIs.

Rank by similarity to your taste vector × freshness × popularity.

Occasionally insert exploration items (low similarity) to expand your range.

Update vectors using ratings:

v_user_new = v_user + α * rating_norm * (v_song - v_user)


Store updated vectors in taste_model.json.

🚀 Typical workflow
python load_library.py         → parse and clean Apple library
python genre_classifier.py     → normalize genres
python analysis.py             → analyze and create playlists
python apple_to_spotify.py     → export playlists to Spotify
python recommender.py --mode mixed  → get weekly recommendations


After listening:

Rate recommendations in user_ratings.csv.

Re-run recommender.py → the model adapts.

Long-term vision

Maintain a personal music brain that evolves with you.

Seamlessly merge past (Apple) and present (Spotify) listening data.

Encourage discovery with smart, feedback-driven recommendations.

Create a historical timeline of your musical identity — what you loved, when, and how it changed.