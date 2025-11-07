import plistlib
import pandas as pd
from pathlib import Path

# 1) Find your export (handles Biblioteca, Biblioteca.xml, etc.)
candidates = [Path("Biblioteca"), Path("Biblioteca.xml"), Path("Biblioteca.XML")]
for p in Path(".").glob("Biblioteca*"):
    if p.is_file() and p.suffix.lower() in ("", ".xml"):
        candidates.insert(0, p)
biblioteca_path = next((p for p in candidates if p.exists()), None)
if biblioteca_path is None:
    raise FileNotFoundError("Couldn't find your 'Biblioteca' export in the current folder.")

# 2) Load via plistlib (preserves the structure correctly)
with open(biblioteca_path, "rb") as f:
    plist = plistlib.load(f)

# 3) Apple stores tracks under the "Tracks" key: {track_id: {track_fields...}}
tracks_dict = plist.get("Tracks", {})
songs = list(tracks_dict.values())  # each is a clean dict of fields

# 4) Make a DataFrame and inspect headers
df = pd.DataFrame(songs)

print("Number of tracks:", len(df))
print("Columns found (sorted):")
print(sorted(df.columns.tolist()))

# 5) Optional: peek at one full record to understand field names/values
# (Pick a non-empty row index if 0 looks odd)
sample_idx = df.first_valid_index()
print("\nSample record keys:")
print(sorted([k for k in df.columns if pd.notna(df.loc[sample_idx, k])]))

print("\nSample record (truncated to common fields):")
common = ['Name','Artist','Album','Genre','Play Count','Year','Date Added','Total Time','Bit Rate','Disc Number','Track Number']
print(df.loc[sample_idx, [c for c in common if c in df.columns]])
