"""Persistent storage: SQLite for users/sessions/metadata, readable folders on disk for data.

Layout (results sit next to the dataset they came from):

    data/
      geoai.db
      <username>/
        <dataset-name>_<id>/
          <original file>
          results/
            <tool>_<YYYY-MM-DD_HHMMSS>_<id>/
              result.json        full response, re-displayed without recomputing
              <output files>     GeoTIFF / PNG / CSV downloads
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager

DATA_DIR = os.path.abspath(os.environ.get("GEOAI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")))
DB_PATH = os.path.join(DATA_DIR, "geoai.db")
SESSION_SECONDS = 7 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, ext TEXT NOT NULL, kind TEXT NOT NULL, size INTEGER NOT NULL, meta TEXT NOT NULL,
    dir TEXT NOT NULL, stored_name TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE, tool TEXT NOT NULL,
    params TEXT NOT NULL, params_key TEXT NOT NULL, summary TEXT NOT NULL, dir TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outputs (
    id TEXT PRIMARY KEY, result_id TEXT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, label TEXT NOT NULL, media TEXT NOT NULL, size INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS files_by_user ON files(user_id, created_at);
CREATE INDEX IF NOT EXISTS results_lookup ON results(user_id, file_id, tool, params_key);
"""


def _id():
    return uuid.uuid4().hex[:12]


def safe_name(name, fallback="file"):
    """Filesystem-safe, readable name: no separators, no leading dots."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name or "")).strip("._")[:80]
    return cleaned or fallback


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db():
    conn = _connect()
    try:
        with conn:  # commits, or rolls back on error
            yield conn
    finally:
        conn.close()


def init():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        conn.executescript(SCHEMA)
    finally:
        conn.close()


def _abs(rel):
    path = os.path.normpath(os.path.join(DATA_DIR, rel))
    if not path.startswith(DATA_DIR + os.sep):  # defence in depth; rel paths are built from safe names
        raise ValueError("Invalid storage path")
    return path


# ================== USERS & SESSIONS ==================
def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        _, n, r, p, salt, digest = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                              dklen=len(digest) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(calc.hex(), digest)


_DUMMY_HASH = hash_password(secrets.token_hex(8))


def create_user(username, password):
    if not USERNAME_RE.match(username or ""):
        raise ValueError("Username must be 3-32 characters: letters, numbers, dot, dash or underscore")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                            (username, hash_password(password), time.time()))
            return {"id": cur.lastrowid, "username": username}
    except sqlite3.IntegrityError:
        raise ValueError("That username is taken") from None


def authenticate(username, password):
    with db() as c:
        row = c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username or "",)).fetchone()
    # Same work whether or not the user exists, so timing doesn't reveal valid usernames
    ok = verify_password(password or "", row["password_hash"] if row else _DUMMY_HASH)
    return {"id": row["id"], "username": row["username"]} if row and ok else None


def set_password(user_id, password):
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    with db() as c:
        c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), user_id))
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))  # sign out everywhere else


def rename_user(old, new):
    """Rename an account, moving its data folder and stored paths with it."""
    if not USERNAME_RE.match(new or ""):
        raise ValueError("Username must be 3-32 characters: letters, numbers, dot, dash or underscore")
    with db() as c:
        row = c.execute("SELECT id, username FROM users WHERE username = ?", (old,)).fetchone()
        clash = c.execute("SELECT 1 FROM users WHERE username = ? AND id != ?", (new, row["id"] if row else -1)).fetchone()
    if not row:
        raise ValueError(f"No user named {old}")
    if clash:
        raise ValueError("That username is taken")
    src, dst = _abs(row["username"]), _abs(new)
    moved = os.path.isdir(src) and src != dst
    if moved:
        os.rename(src, dst)
    try:
        prefix, n = row["username"] + os.sep, len(row["username"])
        with db() as c:
            c.execute("UPDATE users SET username = ? WHERE id = ?", (new, row["id"]))
            for table in ("files", "results"):
                c.execute(f"UPDATE {table} SET dir = ? || substr(dir, ?) WHERE user_id = ? AND dir LIKE ?",
                          (new, n + 1, row["id"], prefix.replace("%", r"\%") + "%"))
    except Exception:
        if moved:
            os.rename(dst, src)
        raise


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id):
    token = secrets.token_urlsafe(32)
    with db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        c.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_token_hash(token), user_id, time.time() + SESSION_SECONDS))
    return token


def user_for_session(token):
    if not token:
        return None
    with db() as c:
        row = c.execute("SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                        "WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), time.time())).fetchone()
    return dict(row) if row else None


def delete_session(token):
    with db() as c:
        c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token or ""),))


# ================== DATASETS ==================
def _file_dict(row):
    d = dict(row)
    d["meta"] = json.loads(d["meta"])
    return d


def save_file(user, name, ext, kind, data, meta):
    """Write an uploaded dataset into its own folder and register it."""
    fid = _id()
    stored = safe_name(name, f"upload.{ext}")
    rel = os.path.join(user["username"], f"{safe_name(os.path.splitext(name)[0], 'dataset')}_{fid}")
    os.makedirs(os.path.join(_abs(rel), "results"), exist_ok=True)
    with open(os.path.join(_abs(rel), stored), "wb") as fh:
        fh.write(data)
    with db() as c:
        c.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (fid, user["id"], os.path.basename(name), ext, kind, len(data), json.dumps(meta), rel, stored, time.time()))
    return get_file(user, fid)


def list_files(user):
    with db() as c:
        files = [_file_dict(r) for r in c.execute("SELECT * FROM files WHERE user_id = ? ORDER BY created_at", (user["id"],))]
    return files


def get_file(user, file_id):
    with db() as c:
        row = c.execute("SELECT * FROM files WHERE id = ? AND user_id = ?", (file_id, user["id"])).fetchone()
    return _file_dict(row) if row else None


def file_path(f):
    return os.path.join(_abs(f["dir"]), f["stored_name"])


def delete_file(user, file_id):
    """Remove a dataset, its folder and every result stored with it."""
    f = get_file(user, file_id)
    if not f:
        return False
    with db() as c:
        c.execute("DELETE FROM files WHERE id = ? AND user_id = ?", (file_id, user["id"]))  # cascades to results/outputs
    shutil.rmtree(_abs(f["dir"]), ignore_errors=True)
    return True


# ================== RESULTS ==================
def params_key(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _result_dict(row, outputs):
    d = dict(row)
    d["params"], d["summary"] = json.loads(d["params"]), json.loads(d["summary"])
    d["downloads"] = [{"label": o["label"], "url": f"/api/outputs/{o['id']}", "size": o["size"]} for o in outputs]
    d.pop("dir"), d.pop("params_key"), d.pop("user_id")
    return d


def find_result(user, file_id, tool, key):
    with db() as c:
        row = c.execute("SELECT id FROM results WHERE user_id = ? AND file_id = ? AND tool = ? AND params_key = ? "
                        "ORDER BY created_at DESC LIMIT 1", (user["id"], file_id, tool, key)).fetchone()
    return row["id"] if row else None


def save_result(user, f, tool, params, summary, payload, outputs):
    """Store a run next to its dataset, replacing an older run with identical settings.

    outputs: [(bytes, filename, label, media)]. Returns the stored payload (with download links).
    """
    key = params_key(params)
    rid, now = _id(), time.time()
    rel = os.path.join(f["dir"], "results", f"{tool}_{time.strftime('%Y-%m-%d_%H%M%S', time.localtime(now))}_{rid}")
    os.makedirs(_abs(rel), exist_ok=True)
    rows, used = [], set()
    for data, filename, label, media in outputs:
        name = safe_name(filename, "output")
        while name in used or name == "result.json":
            name = f"{_id()[:4]}_{name}"
        used.add(name)
        with open(os.path.join(_abs(rel), name), "wb") as fh:
            fh.write(data)
        rows.append((_id(), rid, name, label, media, len(data)))
    payload = payload | {"downloads": [{"label": r[3], "url": f"/api/outputs/{r[0]}"} for r in rows],
                         "result": {"id": rid, "tool": tool, "file_id": f["id"], "file_name": f["name"],
                                    "created_at": now, "params": params, "cached": False}}
    with open(os.path.join(_abs(rel), "result.json"), "w") as fh:
        json.dump(payload, fh, default=float)

    old = find_result(user, f["id"], tool, key)
    with db() as c:
        c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (rid, user["id"], f["id"], tool, json.dumps(params, default=str), key,
                   json.dumps(summary, default=float), rel, now))
        c.executemany("INSERT INTO outputs VALUES (?, ?, ?, ?, ?, ?)", rows)
    if old:
        delete_result(user, old)
    return payload


def load_payload(user, result_id):
    with db() as c:
        row = c.execute("SELECT dir FROM results WHERE id = ? AND user_id = ?", (result_id, user["id"])).fetchone()
    if not row:
        return None
    with open(os.path.join(_abs(row["dir"]), "result.json")) as fh:
        return json.load(fh)


def list_results(user, file_id=None):
    sql, args = "SELECT * FROM results WHERE user_id = ?", [user["id"]]
    if file_id:
        sql, args = sql + " AND file_id = ?", args + [file_id]
    with db() as c:
        rows = c.execute(sql + " ORDER BY created_at DESC", args).fetchall()
        outs = {}
        for o in c.execute("SELECT o.* FROM outputs o JOIN results r ON r.id = o.result_id WHERE r.user_id = ?", (user["id"],)):
            outs.setdefault(o["result_id"], []).append(o)
    return [_result_dict(r, outs.get(r["id"], [])) for r in rows]


def get_output(user, output_id):
    """(absolute path, filename, media) of an output owned by user, else None."""
    with db() as c:
        row = c.execute("SELECT o.filename, o.media, r.dir FROM outputs o JOIN results r ON r.id = o.result_id "
                        "WHERE o.id = ? AND r.user_id = ?", (output_id, user["id"])).fetchone()
    return (os.path.join(_abs(row["dir"]), row["filename"]), row["filename"], row["media"]) if row else None


def delete_result(user, result_id):
    with db() as c:
        row = c.execute("SELECT dir FROM results WHERE id = ? AND user_id = ?", (result_id, user["id"])).fetchone()
        if not row:
            return False
        c.execute("DELETE FROM results WHERE id = ?", (result_id,))
    shutil.rmtree(_abs(row["dir"]), ignore_errors=True)
    return True
