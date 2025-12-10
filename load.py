import plistlib, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# --- Load Biblioteca export (plist) ---
path = next((p for p in [Path("Biblioteca"), Path("Biblioteca.xml"), Path("Biblioteca.XML")] if p.exists()), None)
assert path is not None, "Couldn't find 'Biblioteca' export in current folder."
with open(path, "rb") as f:
    pl = plistlib.load(f)

tracks = list(pl.get("Tracks", {}).values())
df = pd.DataFrame(tracks)

# --- Keep the most useful columns ---
keep = [
    'Name','Artist','Album','Album Artist','Genre','Year',
    'Play Count','Skip Count','Loved','Disliked','Explicit','Compilation',
    'Date Added','Play Date UTC','Total Time','Track Number','Disc Number',
    'Kind','Persistent ID','Location'
]
keep = [c for c in keep if c in df.columns]
df = df[keep].copy()

# --- Type fixes ---
num_cols = [c for c in ['Year','Play Count','Skip Count','Total Time','Track Number','Disc Number'] if c in df.columns]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

for c in [c for c in ['Loved','Disliked','Explicit','Compilation'] if c in df.columns]:
    # plistlib already yields booleans, but guard anyway
    df[c] = df[c].map({True: True, False: False}).fillna(False)

for c in [c for c in ['Date Added','Play Date UTC'] if c in df.columns]:
    df[c] = pd.to_datetime(df[c], errors='coerce')

# --- Helpful derived fields ---
if 'Total Time' in df.columns:
    df['duration_min'] = df['Total Time'] / 60000.0  # ms → minutes

# ✅ Create UTC-aware 'today'
today = pd.Timestamp.now(tz="UTC")

# ✅ Ensure datetime columns are UTC-aware before subtracting
for col in ['Date Added', 'Play Date UTC']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        if df[col].dt.tz is None:  # if tz-naive, localize to UTC
            df[col] = df[col].dt.tz_localize('UTC')

# ✅ Now safe to compute time differences
if 'Date Added' in df.columns:
    df['age_days'] = (today - df['Date Added']).dt.days
if 'Play Date UTC' in df.columns:
    df['last_play_days'] = (today - df['Play Date UTC']).dt.days


# Normalize safe numbers for modeling
def nz(x): 
    return np.where(np.isfinite(x), x, np.nan)

df['plays']  = nz(df.get('Play Count', np.nan))
df['skips']  = nz(df.get('Skip Count', np.nan))
df['year']   = nz(df.get('Year', np.nan))
df['age_days'] = nz(df.get('age_days', np.nan))
df['last_play_days'] = nz(df.get('last_play_days', np.nan))

# --- Preference score (simple, tunable) ---
# Heuristic: more plays ↑, fewer skips ↑, more recent plays ↑, Loved ↑
# Recency factor decays with time since last play
recency = np.exp(-np.clip(df['last_play_days'].fillna(365)/180.0, 0, 10))
loved_bonus = df['Loved'].astype(float) * 1.0
skip_penalty = (df['skips'].fillna(0))**0.7
play_gain = (df['plays'].fillna(0))**0.7

df['preference_score'] = (0.7*play_gain*recency - 0.3*skip_penalty + 0.8*loved_bonus)
df['preference_score'] = df['preference_score'].fillna(df['preference_score'].median())

# ==========================================================
# 🧹 REMOVE INVALID / SPECIFIC ROWS
# ==========================================================
invalid_names = [
    "Henrique Mat 7 Maio",
    "IA PHY F 257",
    "IA PHY F 256",
    "IA PHY F 320",
    "PHY IA F 384",
    "IA phy f 480",
    "Esponja f1"
]

before = len(df)

# 1️⃣ Remove the "Nova Gravação" dummy voice memo
df = df[~((df["Artist"] == "iPhone de Henrique"))]

# 2️⃣ Remove the known invalid rows by name
df = df[~df["Name"].isin(invalid_names)].copy()


# ==========================================================
# 🌍 EXTERNAL SOURCES (SoundCloud / Shazam)
# ==========================================================
def load_external(path_str, source_name, col_map):
    p = Path(path_str)
    if not p.exists():
        print(f"ℹ️  No external data found at {p}, skipping {source_name}.")
        return pd.DataFrame()
    
    try:
        ext = pd.read_csv(p)
        # Rename columns to match core schema
        ext = ext.rename(columns=col_map)
        
        # Add missing core cols
        for c in ['Genre', 'Album', 'Play Count', 'Skip Count', 'Loved']:
            if c not in ext.columns:
                ext[c] = np.nan
                
        # Fill defaults
        ext['Source'] = source_name
        ext['Play Count'] = ext['Play Count'].fillna(1) # assume at least 1 play
        
        # Keep only relevant
        keep_cols = [c for c in df.columns if c in ext.columns] + ['Source']
        return ext[keep_cols]
    except Exception as e:
        print(f"⚠️ Error loading {source_name}: {e}")
        return pd.DataFrame()

# SoundCloud Schema Map (Standard export headers -> Our Schema)
sc_map = {
    "Title": "Name", 
    "User": "Artist", 
    "Created At": "Date Added"
}

# Shazam Schema Map (Standard export headers -> Our Schema)
shz_map = {
    "Title": "Name", 
    "Artist": "Artist", 
    "Date Saved": "Date Added"
}

df_sc = load_external("data/external/soundcloud_likes.csv", "SoundCloud", sc_map)
df_sh = load_external("data/external/shazam_history.csv", "Shazam", shz_map)

# Default main source
df['Source'] = 'Apple'

# Merge all
df = pd.concat([df, df_sc, df_sh], ignore_index=True)

print(f"Start rows: {before} | Apple: {len(df[df['Source']=='Apple'])} | SC: {len(df_sc)} | Shazam: {len(df_sh)} | Total: {len(df)}")





print("Rows:", len(df))
print("Top columns:", df.columns.tolist()[:10], "…")
print(df[['Name','Artist','Genre','Play Count','Skip Count','Date Added','Play Date UTC','preference_score']].head(10))




# Top genres by play count
top_genres = (
    df.groupby('Genre')['Play Count']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_genres.plot(kind='bar')
plt.title("Top Genres by Play Count")
plt.ylabel("Total Plays")
plt.xlabel("Genre")
plt.tight_layout()
plt.show()

# Same for top artists
top_artists = (
    df.groupby('Artist')['Play Count']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

genre_counts = (df['Genre']
                .fillna('Unknown')
                .str.strip()
                .value_counts()
                .reset_index())
genre_counts.columns = ['Genre', 'Count']

print(genre_counts.head(50))



plt.figure(figsize=(10,5))
top_artists.plot(kind='bar')
plt.title("Top Artists by Play Count")
plt.ylabel("Total Plays")
plt.xlabel("Artist")
plt.tight_layout()
plt.show()

# ==========================================================
# 📦 EXPORT CLEAN DATA
# ==========================================================
out_path = Path("data") / "library_clean.csv"
out_path.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(out_path, index=False)
print(f"\n✅ Clean library exported to {out_path.resolve()}")
