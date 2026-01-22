from pathlib import Path
import importlib

# Determine the correct games directory. The repository layout can be either:
# - <repo>/codingame_gym/games/backtrack.jar  (when package files are at repo root)
# - <repo>/codingame_gym/codingame_gym/games/backtrack.jar (when package is nested)
# Try several locations and pick the first that exists.
root = Path(__file__).resolve().parent
candidate_paths = [
    root / 'games',                     # nested package: codingame_gym/games
    root.parent / 'games',              # top-level: codingame_gym/games
    # fallback: try importlib.resources if available
]

games_path = None
for p in candidate_paths:
    if p.exists():
        games_path = p
        break

if games_path is None:
    try:
        # importlib.resources.files may point to the package root; handle it safely
        from importlib.resources import files
        pkg_root = Path(files('codingame_gym'))
        alt = pkg_root / 'games'
        if alt.exists():
            games_path = alt
    except Exception:
        pass

if games_path is None:
    # Last resort: assume nested package path (helps when packaging is different)
    games_path = root / 'games'

envs_catalog = {
    'backtrack': {
        'jar_path': games_path / 'backtrack.jar',
    },
}