# Download MERIT archives from password-protected Dropbox shared folders
'''
MERIT DEM and MERIT Hydro are distributed through password-protected Dropbox shared-folder links. This module logs
in to such a link the way the browser does, lists the files in it, and downloads single files.

Dropbox web flow (as found on the password page, 2026-11):
1. GET the shared link -> redirect to /sm/password?cont=<link path>&content_id=<id>; sets the CSRF cookie "t".
2. POST /sm/auth with is_xhr, t, content_id, password and url=<cont>  -> JSON {"status": "authed"} on success.
   The session cookies then give access to the folder.
3. The folder is listed through the JSON call the web page itself uses (list_shared_link_folder_entries), and a
   file is downloaded from its entry link with dl=1.

These are Dropbox's internal web calls, not a public API: they can change without notice. When they do, the
errors say which step failed, and a CSV of per-file links (name,url) can be passed instead of listing.

Passwords are never logged, printed or written anywhere. read_env_file reads them from a git-ignored .env file.
'''
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from tqdm.auto import tqdm

PathLike = Union[str, Path]
WEB = "https://www.dropbox.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
CHUNK = 8 * 2**20


class DropboxError(RuntimeError):
    """A step of the Dropbox web flow failed (the message names the step, never the password)."""


@dataclass
class RemoteFile:
    name: str
    url: str
    size: Optional[int] = None          # bytes, when the listing gives it


def read_env_file(fp: PathLike, keys=None) -> Dict[str, str]:
    """KEY=VALUE pairs from a .env file (comments and blank lines skipped, surrounding quotes removed).

    Values already set in os.environ win over the file. Returns only `keys` when given.
    """
    values = {}
    fp = Path(fp)
    if fp.exists():
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k in values or (keys and k in keys)})
    return {k: v for k, v in values.items() if keys is None or k in keys}


def _with_query(url: str, **params) -> str:
    """url with query parameters added or replaced."""
    p = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    q.update({k: str(v) for k, v in params.items()})
    return urlunparse(p._replace(query=urlencode(q)))


class DropboxSharedFolder:
    """A password-protected Dropbox shared-folder link (https://www.dropbox.com/scl/fo/<key>/<hash>[/sub]?rlkey=...)."""

    def __init__(self, url: str, password: Optional[str], session: Optional[requests.Session] = None):
        self.url = url
        self._password = password
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.authenticated = False

    def __repr__(self):
        return f"DropboxSharedFolder({self.url!r})"                # never shows the password

    # ---- 1-2. login -----------------------------------------------------------------------------------------------
    def authenticate(self) -> None:
        r = self.session.get(self.url, timeout=60)
        r.raise_for_status()
        if "/sm/password" not in r.url:                            # no password page: link is open
            self.authenticated = True
            return
        if not self._password:
            raise DropboxError(f"{self.url} needs a password")
        q = parse_qs(urlparse(r.url).query)
        content_id, cont = q.get("content_id", [None])[0], q.get("cont", [None])[0]
        csrf = self.session.cookies.get("t")
        if not (content_id and cont and csrf):
            raise DropboxError("password page: content_id, cont or the CSRF cookie 't' not found "
                               "(Dropbox may have changed the page)")
        resp = self.session.post(f"{WEB}/sm/auth", timeout=60, data=dict(
            is_xhr="true", t=csrf, content_id=content_id, password=self._password, url=cont))
        try:
            status = resp.json()
        except ValueError:
            raise DropboxError(f"password step: unexpected reply (HTTP {resp.status_code})") from None
        if status.get("status") != "authed":
            raise DropboxError(f"password step: {status.get('status')!r} {status.get('message') or ''}".strip())
        self.authenticated = True

    # ---- 3. listing -----------------------------------------------------------------------------------------------
    def _link_parts(self):
        """(link_key, secure_hash, sub_path, rlkey) from a /scl/fo/ URL."""
        p = urlparse(self.url)
        parts = [x for x in p.path.split("/") if x]
        if parts[:2] != ["scl", "fo"] or len(parts) < 4:
            raise DropboxError(f"not a /scl/fo/ shared-folder link: {self.url}")
        sub = "/" + "/".join(parts[4:]) if len(parts) > 4 else ""
        return parts[2], parts[3], sub, parse_qs(p.query).get("rlkey", [""])[0]

    def _list_dir(self, sub_path: str):
        link_key, secure_hash, _, rlkey = self._link_parts()
        entries, voucher = [], None
        while True:
            data = dict(is_xhr="true", t=self.session.cookies.get("t", ""), link_key=link_key,
                        secure_hash=secure_hash, link_type="c", rlkey=rlkey, sub_path=sub_path)
            if voucher:
                data["voucher"] = voucher
            resp = self.session.post(f"{WEB}/list_shared_link_folder_entries", data=data, timeout=60)
            try:
                js = resp.json()
            except ValueError:
                raise DropboxError(f"listing {sub_path or '/'}: unexpected reply (HTTP {resp.status_code})") from None
            if "entries" not in js:
                raise DropboxError(f"listing {sub_path or '/'}: no entries in reply (keys {sorted(js)})")
            entries += js["entries"]
            voucher = js.get("next_request_voucher")
            if not js.get("has_more_entries") or not voucher:
                return entries

    def list_files(self, recursive: bool = True, max_depth: int = 3) -> Dict[str, RemoteFile]:
        """{file name: RemoteFile} for the folder (and its sub-folders when recursive)."""
        if not self.authenticated:
            self.authenticate()
        files, todo = {}, [(self._link_parts()[2], 0)]
        while todo:
            sub, depth = todo.pop()
            for e in self._list_dir(sub):
                name = e.get("filename") or e.get("name")
                if e.get("is_dir"):
                    if recursive and depth < max_depth:
                        todo.append((f"{sub}/{name}", depth + 1))
                    continue
                href = e.get("href")
                if name and href:
                    files[name] = RemoteFile(name, href, e.get("bytes") or e.get("size"))
        return files

    # ---- download -------------------------------------------------------------------------------------------------
    def download(self, remote: RemoteFile, dest: PathLike, retries: int = 3,
                 log: Callable[[str], None] = print) -> Path:
        """Stream a file to dest (via dest.part, resumed when the server allows it); checks the size when known."""
        if not self.authenticated:
            self.authenticate()
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        url = _with_query(remote.url, dl=1)
        for attempt in range(1, retries + 1):
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with self.session.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
                    if r.status_code == 416:                       # already complete
                        break
                    r.raise_for_status()
                    if done and r.status_code != 206:              # no resume support: start over
                        done = 0
                    ctype = r.headers.get("Content-Type", "")
                    if "text/html" in ctype:
                        raise DropboxError(f"{remote.name}: got a web page instead of the file (login expired?)")
                    total = int(r.headers.get("Content-Length", 0)) + done or remote.size
                    with open(part, "ab" if done else "wb") as fh, tqdm(
                            total=total, initial=done, unit="B", unit_scale=True, desc=remote.name,
                            mininterval=5) as bar:
                        for chunk in r.iter_content(CHUNK):
                            fh.write(chunk)
                            bar.update(len(chunk))
                break
            except (requests.RequestException, DropboxError) as exc:
                if attempt == retries or isinstance(exc, DropboxError):
                    raise
                log(f"  {remote.name}: {exc.__class__.__name__}, retry {attempt}/{retries - 1} in {30 * attempt} s")
                time.sleep(30 * attempt)
        size = part.stat().st_size
        if remote.size and size != remote.size:
            raise DropboxError(f"{remote.name}: downloaded {size} bytes, listing says {remote.size}")
        part.replace(dest)
        return dest


def read_file_list(fp: PathLike) -> Dict[str, RemoteFile]:
    """{name: RemoteFile} from a CSV with columns name,url (the fallback when listing fails)."""
    with open(fp, newline="", encoding="utf-8") as fh:
        return {row["name"].strip(): RemoteFile(row["name"].strip(), row["url"].strip())
                for row in csv.DictReader(fh) if row.get("name") and row.get("url")}
