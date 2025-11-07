# ==========================================================
# genre_classifier.py
# ==========================================================
"""
GenreClassifier
---------------
Uses a hybrid rules + API + ML approach to map raw genres
(e.g. 'Hip Hop', 'Alternativa', 'Funk Carioca') to one of a
set of umbrella genres and learn from new data.
"""

import os
import re
import json
import time
import requests
import pandas as pdok
import numpy as np
from pathlib import Path
from unidecode import unidecode
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1️⃣  Umbrella genres (base categories you provided)
# ==========================================================

UMBRELLA_GENRES = [
    "Dance", "Hip-Hop/Rap", "Rock", "Alternative", "Pop",
    "Electronic", "Hard Rock", "Brazilian", "MPB", "House",
    "Metal", "R&B/Soul", "Indie Pop", "Worldwide",
    "Comedy", "Soundtrack", "Singer/Songwriter", "Latin",
    "Trance", "Folk", "Holiday", "Techno", "Country",
    "Reggae", "Soul", "Funk", "Dubstep", "K-Pop", "Punjabi"
]

# ==========================================================
# 2️⃣  Rule dictionary (PT/ES/EN variants)
# ==========================================================

RULES = {
    r"\balternativa\b": "Alternative",
    r"\balternative rock\b": "Alternative",
    r"\balternativo\b": "Alternative",
    r"\brock y alternativo\b": "Alternative",
    r"\brock\b": "Rock",
    r"\bhard rock\b": "Hard Rock",
    r"\bhip[- ]?hop\b": "Hip-Hop/Rap",
    r"\brap\b": "Hip-Hop/Rap",
    r"\bdanca|dança\b": "Dance",
    r"\bdance\b": "Dance",
    r"\belectronica|eletronica|electronic\b": "Electronic",
    r"\bhouse|techno|trance|edm\b": "Electronic",
    r"\bpop latino|latino\b": "Latin",
    r"\blatin\b": "Latin",
    r"\bmpb\b": "MPB",
    r"\bbossa nova|samba|pagode|sertanejo|baile funk\b": "Brazilian",
    r"\bbrasileir|brazilian\b": "Brazilian",
    r"\br&b|rnb|soul\b": "R&B/Soul",
    r"\bfolk\b": "Folk",
    r"\bcountry\b": "Country",
    r"\bmetal|grunge|punk\b": "Metal",
    r"\breggae\b": "Reggae",
    r"\bworldwide|urbano latino\b": "Worldwide",
    r"\bcomedy\b": "Comedy",
    r"\bsoundtrack\b": "Soundtrack",
    r"\bsinger|songwriter\b": "Singer/Songwriter",
    r"\bholiday|natal\b": "Holiday",
    r"\bk[- ]?pop\b": "K-Pop",
    r"\bpunjabi\b": "Punjabi"
}

# ==========================================================
# 3️⃣  API helpers
# ==========================================================

class APIFetcher:
    """Fetch genres/tags from MusicBrainz and Last.fm."""
    def __init__(self, lastfm_key=None, cache_path="api_cache.json"):
        self.lastfm_key = lastfm_key or os.getenv("LASTFM_API_KEY", "")
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def save_cache(self):
        self.cache_path.write_text(json.dumps(self.cache, indent=2))

    def _cache_key(self, artist, track):
        return f"{artist.lower()}::{track.lower() if track else ''}"

    def fetch(self, artist, track=None):
        key = self._cache_key(artist, track)
        if key in self.cache:
            return self.cache[key]

        tags = []

        # --- MusicBrainz ---
        try:
            query = f"https://musicbrainz.org/ws/2/artist/?query={artist}&fmt=json"
            resp = requests.get(query, headers={"User-Agent": "music-ml/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                if data.get("artists"):
                    first = data["artists"][0]
                    if "tags" in first:
                        tags.extend([t["name"] for t in first["tags"]])
        except Exception:
            pass

        # --- Last.fm ---
        if self.lastfm_key:
            try:
                url = "http://ws.audioscrobbler.com/2.0/"
                params = {
                    "method": "artist.gettoptags",
                    "artist": artist,
                    "api_key": self.lastfm_key,
                    "format": "json"
                }
                r = requests.get(url, params=params)
                if r.status_code == 200:
                    j = r.json()
                    tags.extend([t["name"] for t in j.get("toptags", {}).get("tag", [])])
            except Exception:
                pass

        tags = list(set(tags))
        self.cache[key] = tags
        self.save_cache()
        time.sleep(1.0)  # be polite
        return tags

# ==========================================================
# 4️⃣  GenreClassifier core
# ==========================================================

class GenreClassifier:
    def __init__(self, umbrella_genres=None, rules=None, lastfm_key=None):
        self.umbrella = umbrella_genres or UMBRELLA_GENRES
        self.rules = rules or RULES
        self.api = APIFetcher(lastfm_key=lastfm_key)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Precompute embeddings for umbrella genres
        self.umbrella_embs = self.model.encode(self.umbrella, normalize_embeddings=True)

        self.manual_map = {}  # user-validated corrections

    # ------------------------------------------
    # Text cleaning
    # ------------------------------------------
    @staticmethod
    def clean(s):
        if not isinstance(s, str):
            return ""
        return unidecode(s.strip().lower())

    # ------------------------------------------
    # Rule-based matching
    # ------------------------------------------
    def rule_match(self, genre):
        g = self.clean(genre)
        for pat, repl in self.rules.items():
            if re.search(pat, g):
                return repl, 0.95
        # Fuzzy match fallback
        match, score, _ = process.extractOne(g, self.umbrella, scorer=fuzz.ratio)
        if score >= 80:
            return match, score/100
        return None, 0.0

    # ------------------------------------------
    # Internet enrichment
    # ------------------------------------------
    def enrich(self, artist, track=None):
        tags = self.api.fetch(artist, track)
        return [self.clean(t) for t in tags if isinstance(t, str)]

    # ------------------------------------------
    # Embedding-based semantic match
    # ------------------------------------------
    def semantic_match(self, genre_or_tags):
        if isinstance(genre_or_tags, str):
            texts = [genre_or_tags]
        else:
            texts = genre_or_tags

        emb = self.model.encode(texts, normalize_embeddings=True)
        sims = cosine_similarity(emb, self.umbrella_embs)
        # take best match among all tags
        max_idx = np.unravel_index(np.argmax(sims, axis=None), sims.shape)
        best_genre = self.umbrella[max_idx[1]]
        confidence = float(np.max(sims))
        return best_genre, confidence

    # ------------------------------------------
    # Full pipeline
    # ------------------------------------------
    def classify(self, genre, artist=None, track=None):
        # 1. Manual map (learned corrections)
        if genre in self.manual_map:
            return self.manual_map[genre], 1.0

        # 2. Rule-based
        match, conf = self.rule_match(genre)
        if conf >= 0.8:
            return match, conf

        # 3. Internet enrichment
        tags = self.enrich(artist or "", track)
        if tags:
            match, conf = self.semantic_match(tags)
            if conf >= 0.6:
                return match, conf

        # 4. Fallback: direct semantic
        match, conf = self.semantic_match(genre)
        return match, conf

    # ------------------------------------------
    # Batch mode
    # ------------------------------------------
    def classify_dataframe(self, df, col_genre="Genre", col_artist="Artist", col_track="Name"):
        results = []
        for _, row in df.iterrows():
            genre = row.get(col_genre, "")
            artist = row.get(col_artist, "")
            track = row.get(col_track, "")
            g, conf = self.classify(genre, artist, track)
            results.append({"Original": genre, "Suggested": g, "Confidence": conf})
        return pd.DataFrame(results)

    # ------------------------------------------
    # Manual learning update
    # ------------------------------------------
    def update_manual(self, original, corrected):
        self.manual_map[original] = corrected
        print(f"✅ Learned mapping: '{original}' → '{corrected}'")

# ==========================================================
# Usage example
# ==========================================================
if __name__ == "__main__":
    clf = GenreClassifier()
    print("Example:", clf.classify("Hip Hop"))
    print("Example:", clf.classify("Eletrônica", artist="Alok"))
