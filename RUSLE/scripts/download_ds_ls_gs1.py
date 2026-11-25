"""List and download DS-LS-GS1 (global 1-arcsec LS factor; Scientific Data 2024, doi 10.11888/Terre.tpdc.300613)
from the TPDC FTP server.

Credentials go in the git-ignored .env at the repo root (never in this file), as issued by TPDC:
    TPDC_FTP_HOST=...
    TPDC_FTP_PORT=21
    TPDC_FTP_USER=...
    TPDC_FTP_PASSWORD=...
    TPDC_FTP_PATH=...        (optional; default: the folder the server opens after login)

Usage (from the repo root):
    python RUSLE/scripts/download_ds_ls_gs1.py                     # list only: remote tree with sizes
    python RUSLE/scripts/download_ds_ls_gs1.py --index             # tiles inside each zip (reads zip directories only)
    python RUSLE/scripts/download_ds_ls_gs1.py --tile N44E6 N48E6  # extract single tiles from the remote zips (resumable)
    python RUSLE/scripts/download_ds_ls_gs1.py --get FILE [FILE...]  # download these remote paths
    python RUSLE/scripts/download_ds_ls_gs1.py --get-from list.txt   # one remote path per line

The listing is saved to RUSLE/outputs/ds_ls_gs1_ftp_listing.csv. Downloads go to
data/soil_erosion/LS_external/DS-LS-GS1/, keep the remote folder structure, resume partial files and skip
files already complete. TPDC notes that bandwidth is best between 02:00 and 10:00 (China time, UTC+8).
"""
import argparse
import csv
import ftplib
import sys
import time
from pathlib import PurePosixPath, Path

from sbtn_leaf.merit_download import read_env_file

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["TPDC_FTP_HOST", "TPDC_FTP_PORT", "TPDC_FTP_USER", "TPDC_FTP_PASSWORD", "TPDC_FTP_PATH"]
OUT_DIR = ROOT / "data" / "soil_erosion" / "LS_external" / "DS-LS-GS1"
LISTING_FP = ROOT / "RUSLE" / "outputs" / "ds_ls_gs1_ftp_listing.csv"


def connect() -> ftplib.FTP:
    env = read_env_file(ROOT / ".env", keys=KEYS)
    missing = [k for k in KEYS[:4] if not env.get(k)]
    if missing:
        raise SystemExit(f"Missing in {ROOT / '.env'}: {', '.join(missing)}")
    ftp = ftplib.FTP(timeout=120, encoding="utf-8")
    ftp.connect(env["TPDC_FTP_HOST"], int(env["TPDC_FTP_PORT"]))
    ftp.login(env["TPDC_FTP_USER"], env["TPDC_FTP_PASSWORD"])
    ftp.set_pasv(True)
    if env.get("TPDC_FTP_PATH"):
        ftp.cwd(env["TPDC_FTP_PATH"])
    print(f"Connected. Start folder: {ftp.pwd()}")
    return ftp


def _entries(ftp: ftplib.FTP, path: str):
    """(name, is_dir, size) for each entry of a remote folder; MLSD if the server has it, LIST otherwise."""
    try:
        for name, facts in ftp.mlsd(path, facts=["type", "size"]):
            if name in (".", ".."):
                continue
            yield name, facts.get("type") == "dir", int(facts.get("size", 0) or 0)
        return
    except ftplib.error_perm:
        pass
    lines = []
    ftp.retrlines(f"LIST {path}", lines.append)
    for line in lines:                       # unix-style: perms links owner group size mon day time name
        parts = line.split(None, 8)
        if len(parts) < 9 or parts[8] in (".", ".."):
            continue
        yield parts[8], line.startswith("d"), int(parts[4]) if parts[4].isdigit() else 0


def walk(ftp: ftplib.FTP, path: str, rows: list, depth: int = 0, max_depth: int = 6):
    for name, is_dir, size in _entries(ftp, path):
        full = str(PurePosixPath(path) / name)
        if is_dir:
            if depth < max_depth:
                walk(ftp, full, rows, depth + 1, max_depth)
        else:
            rows.append((full, size))


def do_list(ftp: ftplib.FTP):
    start = ftp.pwd()
    rows = []
    walk(ftp, start, rows)
    LISTING_FP.parent.mkdir(parents=True, exist_ok=True)
    with open(LISTING_FP, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["remote_path", "size_bytes"])
        w.writerows(rows)
    total = sum(s for _, s in rows)
    print(f"{len(rows):,} files, {total / 1e9:.2f} GB in total. Listing saved to {LISTING_FP.relative_to(ROOT)}")
    for p, s in rows[:15]:
        print(f"  {s / 1e6:10.1f} MB  {p}")
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15:,} more")


class FtpRangeFile:
    """Read-only, seekable view of a remote file through FTP REST, so zipfile can read the central
    directory at the end of a huge archive without downloading it."""

    def __init__(self, ftp: ftplib.FTP, path: str):
        self.ftp, self.path, self.pos = ftp, path, 0
        self.ftp.voidcmd("TYPE I")
        self.size = self.ftp.size(path)

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        conn = self.ftp.transfercmd(f"RETR {self.path}", rest=self.pos)
        chunks, got = [], 0
        while got < n:
            b = conn.recv(min(1 << 16, n - got))
            if not b:
                break
            chunks.append(b)
            got += len(b)
        conn.close()
        try:                                   # the server reports the aborted transfer (426/451) or success
            self.ftp.voidresp()
        except ftplib.all_errors:
            pass
        self.pos += got
        return b"".join(chunks)


def do_index(ftp: ftplib.FTP, retries: int = 3) -> ftplib.FTP:
    """Member names and sizes of every remote zip, read from the zip central directories.

    Resumable: zips already in the index file are skipped. A timeout reconnects and retries."""
    import zipfile
    zips = [r for r, _ in _read_listing() if r.lower().endswith(".zip") and "LS-TOOL" not in r]
    index_fp = LISTING_FP.with_name("ds_ls_gs1_zip_index.csv")
    done = set()
    if index_fp.exists():
        with open(index_fp, encoding="utf-8") as f:
            done = {r["zip"] for r in csv.DictReader(f)}
    new_file = not index_fp.exists()
    with open(index_fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["zip", "member", "size_bytes"])
        for z in zips:
            if z in done:
                print(f"  {z}: already indexed")
                continue
            for attempt in range(1, retries + 1):
                t0 = time.time()
                try:
                    with zipfile.ZipFile(FtpRangeFile(ftp, z)) as zf:
                        infos = zf.infolist()
                    break
                except (*ftplib.all_errors, zipfile.BadZipFile) as e:
                    print(f"  {z}: attempt {attempt} failed ({type(e).__name__}); reconnecting")
                    try:
                        ftp.close()
                    except Exception:
                        pass
                    ftp = connect()
            else:
                print(f"  {z}: skipped after {retries} attempts; rerun --index to retry")
                continue
            w.writerows((z, i.filename, i.file_size) for i in infos)
            f.flush()
            print(f"  {z}: {len(infos)} members ({time.time() - t0:.0f} s)")
    print(f"Index saved to {index_fp.relative_to(ROOT)}")
    return ftp


def _zip_of(member: str) -> str:
    index_fp = LISTING_FP.with_name("ds_ls_gs1_zip_index.csv")
    if not index_fp.exists():
        raise SystemExit("Run --index first")
    with open(index_fp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["member"] == member:
                return r["zip"]
    raise SystemExit(f"{member} is not in the zip index (run --index again to finish indexing)")


def fetch_tile(ftp: ftplib.FTP, member: str, retries: int = 50) -> ftplib.FTP:
    """Download one tile out of its remote zip: only the member's compressed bytes are transferred (FTP REST),
    to <member>.part, resuming after timeouts or restarts. Then it is inflated and checked against the zip CRC."""
    import zipfile
    import zlib
    out = OUT_DIR / member
    if out.exists():
        print(f"{member}: already downloaded")
        return ftp
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = _zip_of(member)
    for attempt in range(1, retries + 1):        # locate the member's data inside the remote zip
        try:
            rf = FtpRangeFile(ftp, z)
            with zipfile.ZipFile(rf) as zf:
                info = zf.getinfo(member)
            rf.seek(info.header_offset)
            hdr = rf.read(30)
            data_start = info.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little")
            break
        except (*ftplib.all_errors, zipfile.BadZipFile) as e:
            print(f"{member}: locating attempt {attempt} failed ({type(e).__name__}); reconnecting", flush=True)
            ftp = _reconnect(ftp)
    else:
        raise SystemExit(f"{member}: could not read the zip directory of {z}")
    length = info.compress_size
    part = OUT_DIR / f"{member}.part"
    print(f"{member}: in {z}, {length / 1e9:.2f} GB compressed ({'deflate' if info.compress_type else 'stored'})", flush=True)

    attempt, t0, last = 0, time.time(), 0.0
    while (part.stat().st_size if part.exists() else 0) < length:
        have = part.stat().st_size if part.exists() else 0
        try:
            ftp.voidcmd("TYPE I")
            conn = ftp.transfercmd(f"RETR {z}", rest=data_start + have)
            with open(part, "ab") as f:
                while have < length:
                    b = conn.recv(min(1 << 20, length - have))
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 60:
                        print(f"  {member}: {have / 1e6:,.0f} / {length / 1e6:,.0f} MB", flush=True)
                        last = time.time()
            conn.close()
            try:
                ftp.voidresp()
            except ftplib.all_errors:
                pass
            attempt = 0
        except ftplib.all_errors as e:
            attempt += 1
            if attempt > retries:
                raise SystemExit(f"{member}: giving up after {retries} failed attempts; rerun to resume")
            print(f"  {member}: transfer interrupted ({type(e).__name__}) at {have / 1e6:,.0f} MB; "
                  f"reconnecting (attempt {attempt})", flush=True)
            time.sleep(min(30 * attempt, 300))
            ftp = _reconnect(ftp)

    print(f"{member}: transfer complete in {(time.time() - t0) / 3600:.1f} h; unpacking", flush=True)
    crc = 0
    tmp = OUT_DIR / f"{member}.tmp"
    with open(part, "rb") as src, open(tmp, "wb") as dst:
        d = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SystemExit(f"{member}: unsupported compression type {info.compress_type}")
        while chunk := src.read(1 << 22):
            data = d.decompress(chunk) if d else chunk
            crc = zlib.crc32(data, crc)
            dst.write(data)
        if d:
            data = d.flush()
            crc = zlib.crc32(data, crc)
            dst.write(data)
    if crc != info.CRC or tmp.stat().st_size != info.file_size:
        raise SystemExit(f"{member}: CRC or size mismatch; delete {part.name} and rerun")
    tmp.rename(out)
    part.unlink()
    print(f"{member}: OK, {out.stat().st_size / 1e9:.2f} GB, CRC verified -> {out.relative_to(ROOT)}", flush=True)
    return ftp


def _reconnect(ftp: ftplib.FTP) -> ftplib.FTP:
    try:
        ftp.close()
    except Exception:
        pass
    for attempt in range(1, 11):
        try:
            return connect()
        except ftplib.all_errors as e:
            print(f"  reconnect attempt {attempt} failed ({type(e).__name__})", flush=True)
            time.sleep(60)
    raise SystemExit("could not reconnect to the FTP server")


def _read_listing():
    if not LISTING_FP.exists():
        raise SystemExit("Run the listing first (no arguments)")
    with open(LISTING_FP, encoding="utf-8") as f:
        return [(r["remote_path"], int(r["size_bytes"])) for r in csv.DictReader(f)]


def download(ftp: ftplib.FTP, remote: str, start: str):
    rel = PurePosixPath(remote).relative_to(start) if remote.startswith(start) else PurePosixPath(remote).name
    local = OUT_DIR / Path(*PurePosixPath(rel).parts)
    local.parent.mkdir(parents=True, exist_ok=True)
    ftp.voidcmd("TYPE I")
    size = ftp.size(remote)
    have = local.stat().st_size if local.exists() else 0
    if size is not None and have == size:
        print(f"  complete, skipped: {rel}")
        return
    t0 = time.time()
    with open(local, "ab" if have else "wb") as f:
        ftp.retrbinary(f"RETR {remote}", f.write, blocksize=1 << 20, rest=have or None)
    got = local.stat().st_size
    print(f"  {got / 1e6:,.1f} MB in {time.time() - t0:.0f} s: {rel}")
    if size is not None and got != size:
        print(f"  WARNING: size {got} != remote {size}; rerun to resume")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--get", nargs="+", metavar="REMOTE_PATH", help="remote files to download")
    ap.add_argument("--get-from", type=Path, help="text file with one remote path per line")
    ap.add_argument("--index", action="store_true",
                    help="list the members of every remote zip (reads only the zip directories)")
    ap.add_argument("--tile", nargs="+", metavar="TILE",
                    help="tiles to extract from their remote zips, e.g. N44E6.tif (resumable)")
    args = ap.parse_args()

    ftp = connect()
    try:
        if args.index:
            ftp = do_index(ftp)
            return
        if args.tile:
            for t in args.tile:
                ftp = fetch_tile(ftp, t if t.endswith(".tif") else f"{t}.tif")
            return
        targets = list(args.get or [])
        if args.get_from:
            targets += [l.strip() for l in args.get_from.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not targets:
            do_list(ftp)
            return
        start = ftp.pwd()
        print(f"Downloading {len(targets)} file(s) to {OUT_DIR.relative_to(ROOT)}")
        for remote in targets:
            download(ftp, remote, start)
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()


if __name__ == "__main__":
    sys.exit(main())
