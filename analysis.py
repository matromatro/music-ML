# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 1️⃣ Load data
df = pd.read_csv("data/library_classified.csv")

# ==========================================================
# 🎧 MULTI-CONTEXT PLAYLIST GENERATION
# ==========================================================

# 2️⃣ Clean + ensure correct types
df["Genre_norm"] = df["Genre_norm"].fillna("Unknown")
df["country"] = df["country"].fillna("").str.upper()

# 3️⃣ Derive contextual tags
def derive_context(row):
    tags = []
    if row["country"] == "BR":
        tags.append("Brazilian")
    # optional: heuristic Latin detection for non-BR
    if isinstance(row["Name"], str) and any(c in row["Name"].lower() for c in ["ñ", "ó", "á", "é"]) and row["country"] != "BR":
        tags.append("Latin")
    return tags

df["context_tags"] = df.apply(derive_context, axis=1)

# 4️⃣ Build composite genre labels
df["composite_genres"] = df.apply(lambda r: [
    f"{ctx} {r['Genre_norm']}" for ctx in r["context_tags"]
] + [r["Genre_norm"]], axis=1)

# 5️⃣ Generate overlapping playlists
from pathlib import Path
Path("data/output/playlists").mkdir(parents=True, exist_ok=True)

exploded = df.explode("composite_genres")

for genre, subdf in exploded.groupby("composite_genres"):
    subdf = subdf.sort_values("preference_score", ascending=False)
    playlist = subdf[["Name", "Artist", "Album", "preference_score"]]
    out_path = Path(f"data/output/playlists/{genre.replace('/', '_')}.csv")
    playlist.to_csv(out_path, index=False)
    print(f"🎧 Saved {len(playlist)} songs to playlist: {genre}")


# 6️⃣ Visualizations (composite version)
summary = (
    exploded.groupby("composite_genres")
    .agg(total_plays=("Play Count", "sum"))
    .sort_values("total_plays", ascending=False)
)

plt.figure(figsize=(10,5))
summary.head(10)["total_plays"].plot(kind="bar", color="salmon")
plt.title("Top 10 Composite Genres by Play Count")
plt.ylabel("Total Plays")
plt.xlabel("Composite Genre")
plt.tight_layout()
plt.savefig(out_dir / "top_composite_genres.png")
plt.show()

print("✅ Charts generated for composite genres")
