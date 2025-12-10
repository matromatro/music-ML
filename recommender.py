import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import random
import json

# ==========================================================
# CONFIG
# ==========================================================
SCOPE = "user-top-read" # To optionally fetch top tracks directly
CACHE_PATH = ".spotify_cache"

class Recommender:
    def __init__(self):
        # 1. Setup Spotify
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
        
        if not client_id or not client_secret:
            raise ValueError("❌ Missing SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET for recommendations.")
            
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPE,
            cache_path=CACHE_PATH,
            open_browser=True
        ))
        
        # 2. Setup Embedding Model
        print("🧠 Loading embedding model (all-MiniLM-L6-v2) (this may take a moment)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    # ----------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------
    def load_library(self):
        path = Path("data/library_classified.csv")
        if not path.exists():
            print("⚠️ Library not found. Run analysis first.")
            return None
        return pd.read_csv(path)

    # ----------------------------------------------------
    # PROFILE BUILDING
    # ----------------------------------------------------
    def build_profiles(self, df):
        """
        Creates a 'Taste Vector' for each genre based on highly-rated songs.
        """
        print("📊 Building taste profiles...")
        # Filter for "liked" or high play count songs
        liked = df[
            (df['Loved'] == True) | 
            (df['preference_score'] > df['preference_score'].quantile(0.7))
        ].copy()
        
        profiles = {}
        
        # Group by genre to make genre-specific vectors
        for genre, group in liked.groupby("Genre_norm"):
            if len(group) < 5: continue # Skip tiny genres
            
            # Combine Artist + Name for semantic embedding
            texts = (group["Artist"] + " " + group["Name"]).tolist()
            embeddings = self.model.encode(texts)
            
            # Average vector = The "Vibe" of this genre for you
            mean_vector = np.mean(embeddings, axis=0)
            profiles[genre] = {
                "vector": mean_vector,
                "top_artists": group["Artist"].value_counts().head(5).index.tolist(),
                "sample_tracks": group["Name"].head(5).tolist() # simplified
            }
            
        return profiles

    # ----------------------------------------------------
    # CANDIDATE GENERATION (Spotify API)
    # ----------------------------------------------------
    def fetch_candidates(self, profiles, limit_per_genre=5):
        candidates = []
        
        for genre, profile in profiles.items():
            print(f"   ✨ Fetching for {genre}...")
            seeds = profile["top_artists"][:2] # Spotify allows max 5 seeds total
            
            # We need Spotify IDs for seeds. Search them.
            seed_ids = []
            for artist in seeds:
                res = self.sp.search(q=f"artist:{artist}", type="artist", limit=1)
                if res["artists"]["items"]:
                    seed_ids.append(res["artists"]["items"][0]["id"])
            
            if not seed_ids: continue
            
            try:
                # Get recommendations
                recs = self.sp.recommendations(seed_artists=seed_ids, limit=limit_per_genre)
                
                for track in recs["tracks"]:
                    candidates.append({
                        "Genre_Focus": genre,
                        "Name": track["name"],
                        "Artist": track["artists"][0]["name"],
                        "Album": track["album"]["name"],
                        "Spotify_URI": track["uri"],
                        "Spotify_URL": track["external_urls"]["spotify"]
                    })
            except Exception as e:
                print(f"      ⚠️ API Error for seeds {[s for s in seeds]}: {e}")
                import traceback
                traceback.print_exc()
                
        return pd.DataFrame(candidates)

    # ----------------------------------------------------
    # RANKING & FILTERING
    # ----------------------------------------------------
    def rank_candidates(self, candidates_df, library_df, profiles):
        if candidates_df.empty: return candidates_df
        
        print("⚖️  Ranking & Filtering...")
        
        # 1. Filter out existing library songs (Name + Artist match)
        # Create unique keys
        lib_keys = set((library_df["Name"] + " " + library_df["Artist"]).str.lower())
        
        cand_keys = (candidates_df["Name"] + " " + candidates_df["Artist"]).str.lower()
        candidates_df = candidates_df[~cand_keys.isin(lib_keys)].copy()
        
        if candidates_df.empty: return candidates_df

        # 2. Semantic Scoring
        # Compare candidate (Name + Artist) vs Profile Vector of its Genre
        results = []
        
        # Batch encode candidates to save time
        cand_texts = (candidates_df["Artist"] + " " + candidates_df["Name"]).tolist()
        cand_embs = self.model.encode(cand_texts)
        
        for idx, row in candidates_df.reset_index(drop=True).iterrows():
            genre = row["Genre_Focus"]
            if genre in profiles:
                # Cosine sim with user's taste vector for this genre
                user_vec = profiles[genre]["vector"].reshape(1, -1)
                song_vec = cand_embs[idx].reshape(1, -1)
                
                sim = cosine_similarity(user_vec, song_vec)[0][0]
                row["Match_Score"] = sim
                results.append(row)
        
        final_df = pd.DataFrame(results)
        return final_df.sort_values("Match_Score", ascending=False)

    # ----------------------------------------------------
    # MAIN
    # ----------------------------------------------------
    # ----------------------------------------------------
    # FEEDBACK & LEARNING
    # ----------------------------------------------------
    def load_ratings(self):
        p = Path("data/user_ratings.csv")
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame(columns=["Name", "Artist", "Rating", "Timestamp"])

    def save_ratings(self, new_ratings):
        df = self.load_ratings()
        df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index=True)
        df.to_csv("data/user_ratings.csv", index=False)
        return df

    def save_profiles(self, profiles):
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for genre, data in profiles.items():
            serializable[genre] = {
                "vector": data["vector"].tolist(),
                "top_artists": data["top_artists"],
                "sample_tracks": data["sample_tracks"]
            }
        
        with open("data/taste_model.json", "w") as f:
            json.dump(serializable, f)

    def load_profiles(self):
        p = Path("data/taste_model.json")
        if not p.exists():
            return None
            
        with open(p, "r") as f:
            data = json.load(f)
            
        profiles = {}
        for genre, content in data.items():
            profiles[genre] = {
                "vector": np.array(content["vector"]),
                "top_artists": content["top_artists"],
                "sample_tracks": content["sample_tracks"]
            }
        return profiles

    def update_model(self, profiles, ratings_df):
        print("🧠 Updating model based on recent ratings...")
        alpha = 0.2 # Learning rate
        
        # Simple update: move genre vector towards rated song vector
        for _, row in ratings_df.iterrows():
            genre = row.get("Genre_Focus")
            if not genre or genre not in profiles: continue
            
            rating = float(row["Rating"])
            # Normalize rating: 1..5 -> -1..1
            score = (rating - 3) / 2.0 
            
            # Re-encode song to get vector
            text = f"{row['Artist']} {row['Name']}"
            song_vec = self.model.encode(text)
            
            # Current user vector
            user_vec = profiles[genre]["vector"]
            
            # Update: v_new = v_old + alpha * score * (v_song - v_old)
            # If (score > 0), move towards song. If (score < 0), move away.
            new_vec = user_vec + alpha * score * (song_vec - user_vec)
            
            # Normalize to keep it unit length (optional but good for cosine sim)
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
                
            profiles[genre]["vector"] = new_vec
            
        self.save_profiles(profiles)
        print("✅ Model updated and saved.")

    # ----------------------------------------------------
    # INTERACTIVE
    # ----------------------------------------------------
    def interactive_rate(self, recommendations):
        print("\n📝 Interactive Rating Mode")
        print("Rate the following recommendations (1-5) or press Enter to skip.")
        
        new_ratings = []
        
        for idx, row in recommendations.iterrows():
            print(f"\n🎵 {row['Name']} - {row['Artist']} ({row['Genre_Focus']})")
            print(f"   Link: {row['Spotify_URL']}")
            
            choice = input("   Rating (1-5) [Skip]: ").strip()
            if choice in ['1','2','3','4','5']:
                new_ratings.append({
                    "Name": row["Name"],
                    "Artist": row["Artist"],
                    "Genre_Focus": row["Genre_Focus"],
                    "Rating": int(choice),
                    "Timestamp": pd.Timestamp.now()
                })
        
        if new_ratings:
            self.save_ratings(new_ratings)
            return pd.DataFrame(new_ratings)
        return pd.DataFrame()

    def run(self):
        df = self.load_library()
        if df is None: return
        
        # Try loading existing model first
        profiles = self.load_profiles()
        if not profiles:
            profiles = self.build_profiles(df)
            self.save_profiles(profiles) # Initial save
        else:
            print("💾 Loaded existing taste model.")
        
        # Generate
        candidates = self.fetch_candidates(profiles, limit_per_genre=7)
        recommendations = self.rank_candidates(candidates, df, profiles)
        
        # Save output
        out_path = Path("data/recommendations.csv")
        recommendations.to_csv(out_path, index=False)
        print(f"\n✅ Saved {len(recommendations)} recommendations to {out_path}")
        if not recommendations.empty:
            print(recommendations[["Name", "Artist", "Genre_Focus", "Match_Score"]].head(10))
            
            # Ask for feedback
            print("\n")
            do_rate = input("Do you want to rate these songs now? [y/N]: ").lower()
            if do_rate == 'y':
                recent_ratings = self.interactive_rate(recommendations.head(10))
                if not recent_ratings.empty:
                    self.update_model(profiles, recent_ratings)    
        
    def run_wrapper(self):
         self.run()

if __name__ == "__main__":
    import sys
    try:
        rec = Recommender()
        rec.run()
    except Exception as e:
        print(f"\n🔥 Error: {e}")
        sys.exit(1)
