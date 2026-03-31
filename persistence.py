# persistence.py
import os, json, base64, requests, logging, time, threading
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GH_PAT_TOKEN", "")  # Uses your custom Render token name
GITHUB_REPO  = os.getenv("GITHUB_REPO", "Elliot14R/Crypto_AI_bot")
GITHUB_BRANCH= os.getenv("GITHUB_BRANCH", "main")

GITHUB_API   = "https://api.github.com"
HEADERS      = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

PERSISTENT_FILES = [
    "trades.json",
    "trade_history.json",
    "signals.json",
    "scan_mode.json",
]

# 🚦 Thread lock to prevent 409 SHA conflicts when saving
_save_lock = threading.Lock()

def _get_file_sha(filename: str) -> str | None:
    """Get current SHA of a file in GitHub."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("sha")
    except Exception:
        pass
    return None

def save_json(filename: str, data: dict | list) -> bool:
    """
    Thread-safe save to GitHub repo under /data/ folder.
    Uses a lock and retry loop to permanently stop 409 Conflicts.
    """
    # If a full path is passed, extract just the filename
    filename = Path(filename).name 
    
    if not GITHUB_TOKEN:
        log.warning("No GH_PAT_TOKEN found, skipping GitHub save.")
        return False

    url     = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
    content = base64.b64encode(json.dumps(data, indent=2, default=str).encode()).decode()
    
    with _save_lock:
        for attempt in range(5):
            sha = _get_file_sha(filename)
            payload = {
                "message": f"bot: update {filename} [{datetime.now(timezone.utc).strftime('%H:%M UTC')}]",
                "content": content,
                "branch":  GITHUB_BRANCH,
            }
            if sha:
                payload["sha"] = sha

            try:
                r = requests.put(url, headers=HEADERS, json=payload, timeout=15)
                if r.status_code in (200, 201):
                    return True
                
                if r.status_code == 409:
                    log.warning(f"GitHub 409 Conflict for {filename} (Attempt {attempt+1}/5). Retrying in 2s...")
                    time.sleep(2)
                    continue
                    
                log.error(f"GitHub save failed for {filename}: {r.status_code} {r.text[:100]}")
                break
            except Exception as e:
                log.error(f"GitHub request error for {filename}: {e}")
                break
        
    return False

def load_from_github(filename: str, default):
    """Load JSON data from GitHub repo, fallback to local."""
    filename = Path(filename).name
    if GITHUB_TOKEN:
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                content = r.json().get("content", "")
                decoded = base64.b64decode(content).decode()
                return json.loads(decoded)
        except Exception as e:
            log.debug(f"GitHub load failed for {filename}: {e}")

    try:
        p = Path(filename)
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except Exception:
        pass
    return default

def load_json(path: str, default):
    return load_from_github(path, default)

def pull_all_from_github():
    pulled = 0
    for filename in PERSISTENT_FILES:
        data = load_from_github(filename, None)
        if data is not None:
            try:
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                pulled += 1
                log.info(f"  Restored {filename} from GitHub")
            except Exception as e:
                log.warning(f"  Failed to restore {filename}: {e}")
    return pulled
