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
    return Path(path).exists()

# ---------- pipeline ----------
def main():
    parser = argparse.ArgumentParser(description="music-ML master pipeline")
    parser.add_argument("--refresh-library", action="store_true", help="run load.py")
    parser.add_argument("--classify", action="store_true", help="run genre_classifier.py")
    parser.add_argument("--analyze", action="store_true", help="run analysis.py")
    parser.add_argument("--export-spotify", action="store_true", help="run apple_to_spotify.py")
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

    # --- step 4: export playlists ---
    if args.export_spotify:
        run("apple_to_spotify.py")

    # --- step 5: recommendations ---
    if args.recommend:
        if ensure_file(classified_file, "genre_classifier.py"):
            run("recommender.py")

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
