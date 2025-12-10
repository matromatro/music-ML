# main.py
import argparse, subprocess, sys
from pathlib import Path

# ---------- helpers ----------
def run(script):
    print(f"\n🚀 Running {script} ...")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"⚠️  {script} exited with code {result.returncode}")
    else:
        print(f"✅  {script} finished.\n")

def ensure_file(path, build_script):
    """Check if path exists; if not, run build_script. 
       If it exists, ask if user wants to overwrite."""
    p = Path(path)
    if p.exists():
        choice = input(f"⚙️  '{path}' already exists. Overwrite? [y/N]: ").strip().lower()
        if choice != "y":
            print(f"➡️  Keeping existing '{path}'.")
            return True
    else:
        print(f"📦 '{path}' not found, will run {build_script} to create it.")
    run(build_script)
    run(build_script)
    return Path(path).exists()

def check_dependencies():
    """Checks if critical packages are installed."""
    required = ["spotipy", "sentence_transformers", "pandas", "numpy", "spotipy"]
    missing = []
    
    # Simple import check
    import importlib.util
    for package in required:
        # map package name to import name if different
        import_name = package.replace("-", "_")
        if importlib.util.find_spec(import_name) is None:
            missing.append(package)
            
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("📦 Installing dependencies from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed.\n")

def ensure_credentials():
    """Prompts for Spotify credentials if missing, with .env support."""
    import os
    from pathlib import Path

    env_path = Path(".env")
    
    # 1. Try loading from .env
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v
                    
    # 2. Check if loaded
    if not os.getenv("SPOTIPY_CLIENT_ID") or not os.getenv("SPOTIPY_CLIENT_SECRET"):
        print("\n🔑 Spotify Credentials Missing!")
        print("   To generate recommendations, we need your Spotify API keys.")
        print("   (Create an app at https://developer.spotify.com/dashboard)")
        
        cid = input("   👉 Enter Client ID: ").strip()
        sec = input("   👉 Enter Client Secret: ").strip()
        
        if cid and sec:
            os.environ["SPOTIPY_CLIENT_ID"] = cid
            os.environ["SPOTIPY_CLIENT_SECRET"] = sec
            os.environ["SPOTIPY_REDIRECT_URI"] = "http://127.0.0.1:8888/callback"
            print("✅ Credentials set for this session.")
            
            # 3. Ask to save
            save = input("   💾 Save these to .env for future runs? [y/N]: ").lower()
            if save == "y":
                with open(".env", "w") as f:
                    f.write(f"SPOTIPY_CLIENT_ID={cid}\n")
                    f.write(f"SPOTIPY_CLIENT_SECRET={sec}\n")
                    f.write(f"SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback\n")
                print("   ✅ Saved to .env (added to .gitignore).")
        else:
            print("⚠️  Skipping credentials. Some features will fail.\n")

# ---------- pipeline ----------
def main():
    check_dependencies()
    ensure_credentials()
    parser = argparse.ArgumentParser(description="music-ML master pipeline")
    parser.add_argument("--refresh-library", action="store_true", help="run load.py")
    parser.add_argument("--classify", action="store_true", help="run genre_classifier.py")
    parser.add_argument("--analyze", action="store_true", help="run analysis.py")
    parser.add_argument("--export-spotify", action="store_true", help="run spotify_manager.py (Sync playlists)")
    parser.add_argument("--recommend", action="store_true", help="run recommender.py")
    args = parser.parse_args()

    data_dir = Path("data")
    clean_file = data_dir / "library_clean.csv"
    classified_file = data_dir / "library_classified.csv"

    # --- step 1: load library ---
    if args.refresh_library:
        run("load.py")

    # --- step 2: genre classification ---
    if args.classify:
        if ensure_file(clean_file, "load.py"):
            run("genre_classifier.py")

    # --- step 3: analysis ---
    if args.analyze:
        if ensure_file(classified_file, "genre_classifier.py"):
            run("analysis.py")

    # --- step 4: export playlists (organized) ---
    if args.export_spotify:
        run("spotify_manager.py")

    # --- step 5: recommendations ---
    if args.recommend:
        if ensure_file(classified_file, "genre_classifier.py"):
            run("recommender.py")

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
