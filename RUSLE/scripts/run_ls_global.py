"""Global RUSLE LS from MERIT: download each 30-degree box, process it, delete the tars, then mosaic.

Writes up to three global LS layers in one pass per box (see RUSLE/README.md, LS Layer):
- default, LS1 = correct Desmet & Govers (area_term "area"):  RUSLE/outputs/rusle_LS_all_5m.tif
- GloSEM-compatible (area_term "sca"):                        RUSLE/outputs/rusle_LS_all_5m_glosem_compat.tif
- LS2, slope-class exponent + Liu S (area_term "ls2"):        RUSLE/outputs/rusle_LS_all_5m_ls2.tif
Parts and manifests go to RUSLE/outputs/ls_parts_glosem, ls_parts_glosem_sca and ls_parts_ls2 (the notebook's
folders), so the run is resumable: rerun the same command and finished boxes are skipped. --variants limits the run
to some layers (a box is downloaded only if one of them still needs it).

Passwords: put them in a .env file at the repo root (git-ignored), never in this file:
    MERIT_HYDRO_PASSWORD=...
    MERIT_DEM_PASSWORD=...

Usage (from the repo root, with the project venv):
    python RUSLE/scripts/run_ls_global.py --dry-run          # log in, list, show what would run
    python RUSLE/scripts/run_ls_global.py --boxes s30e030    # one box
    python RUSLE/scripts/run_ls_global.py                    # everything, then mosaic
    python RUSLE/scripts/run_ls_global.py --variants ls2     # only the LS2 layer
    python RUSLE/scripts/run_ls_global.py --fresh            # from scratch: old parts folders moved aside first
    python RUSLE/scripts/run_ls_global.py --mosaic-only --mosaic-partial
"""
import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import sbtn_leaf.rusle_ls as rls
from sbtn_leaf.merit_download import DropboxError, DropboxSharedFolder, read_env_file, read_file_list

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "RUSLE" / "outputs"
MERIT_DIR = ROOT / "data" / "soil_erosion" / "MERIT"
CROP_FP = ROOT / "data" / "soil_erosion" / "GloSEM_v1.3" / "ESA100_Cropland_fraction.tif"

LINKS = {   # Dropbox shared folders from the MERIT authors; which is which is read from the file names
    "MERIT_HYDRO_PASSWORD": "https://www.dropbox.com/scl/fo/fw8qrf73zhzmo6p92gbxv/AFmj4KtFj5i_zYOrwPANWQg/v1.0.1"
                            "?dl=0&rlkey=a0be5opyb2kpkf0ssbvau8y9q",
    "MERIT_DEM_PASSWORD": "https://www.dropbox.com/scl/fo/bjsd03zm1ari3x7zuf2ac/AE7JCRVuU8cE-6_HnC5zzOE/v1.0.3"
                          "?dl=0&rlkey=pnobdlzoclhdkpsbpctnmer2n",
}
DEM_NAME, UPA_NAME = "dem_tif_{box}.tar", "upa_{box}.tar"       # as DEM_NAME / UPA_NAME in the notebook
BOX_RE = r"([ns]\d{2}[ew]\d{3})"

# run settings: the notebook's (GloSEM v1.1 limits); both outputs share them
SETTINGS = dict(out_res_deg=rls.FIVE_MIN_DEG, lambda_max=122, max_slope_deg=26.6, channel_upa_km2=None)
OUTPUTS = {   # area_term -> (parts folder, mosaic suffix)
    "area": (OUT_DIR / "ls_parts_glosem", ""),
    "sca": (OUT_DIR / "ls_parts_glosem_sca", "_glosem_compat"),
    "ls2": (OUT_DIR / "ls_parts_ls2", "_ls2"),
}

log = logging.getLogger("ls_global")


def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT_DIR / "ls_global_run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def gb(n):
    return f"{(n or 0) / 2**30:.1f} GB"


def remote_files(args):
    """{file name: (folder, RemoteFile)} from both Dropbox links (or from --file-list)."""
    env = read_env_file(ROOT / ".env", keys=list(LINKS))
    files = {}
    for key, url in LINKS.items():
        if not env.get(key):
            raise SystemExit(f"{key} is not set: add it to {ROOT / '.env'} (see the docstring of this script)")
        folder = DropboxSharedFolder(url, env[key])
        folder.authenticate()
        log.info(f"logged in: {url.split('?')[0]}")
        listing = read_file_list(args.file_list) if args.file_list else folder.list_files()
        files.update({name: (folder, rf) for name, rf in listing.items()})
    return files


def box_index(files):
    """{box: {"dem": (folder, RemoteFile), "upa": (...)}} for boxes with both archives."""
    out = {}
    for name, item in files.items():
        for kind, pattern in (("dem", DEM_NAME), ("upa", UPA_NAME)):
            m = re.fullmatch(pattern.format(box=BOX_RE), name)
            if m:
                out.setdefault(m.group(1), {})[kind] = item
    return {b: v for b, v in out.items() if len(v) == 2}


def manifest_done(box, variants):
    """True when the box is done in the parts folder of every selected variant."""
    return all(rls._load_manifest(OUTPUTS[v][0] / "manifest.json").get(box) == "done" for v in variants)


def move_aside(variants):
    """--fresh: rename the selected parts folders to <name>_old_<timestamp> (nothing is deleted)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for v in variants:
        d = OUTPUTS[v][0]
        if d.exists():
            d.rename(d.with_name(f"{d.name}_old_{stamp}"))
            log.info(f"fresh run: moved {d.name} to {d.name}_old_{stamp}")


def fetch(folder, remote, dest):
    """Download one archive, logging in again once if the session expired."""
    try:
        return folder.download(remote, dest, log=log.info)
    except DropboxError as exc:
        log.info(f"  {remote.name}: {exc}; logging in again")
        folder.authenticate()
        return folder.download(remote, dest, log=log.info)


def process_box(box, pair, args, crop_kw, variants):
    merit_dir = Path(args.merit_dir)
    merit_dir.mkdir(parents=True, exist_ok=True)
    targets = {kind: merit_dir / (DEM_NAME if kind == "dem" else UPA_NAME).format(box=box) for kind in pair}
    need = sum(pair[k][1].size or 0 for k in pair if not targets[k].exists())
    free = shutil.disk_usage(merit_dir).free
    if need * 1.2 > free:
        raise SystemExit(f"{box}: needs {gb(need)} for the download, only {gb(free)} free in {merit_dir}; stopping")

    downloaded = []
    for kind, (folder, remote) in pair.items():
        dest = targets[kind]
        if dest.exists() and (remote.size is None or dest.stat().st_size == remote.size):
            log.info(f"  {dest.name}: already on disk, kept")
            continue
        t0 = time.time()
        fetch(folder, remote, dest)
        downloaded.append(dest)
        log.info(f"  {dest.name}: downloaded {gb(dest.stat().st_size)} in {(time.time() - t0) / 60:.1f} min")

    t0 = time.time()
    # only the variants this box still needs (the others are done, maybe by an earlier run)
    todo = tuple(v for v in variants if rls._load_manifest(OUTPUTS[v][0] / "manifest.json").get(box) != "done")
    manifests, _ = rls.run_local(merit_dir, {t: OUTPUTS[t][0] for t in todo}, boxes=[box],
                                 dem_name=DEM_NAME, upa_name=UPA_NAME, area_term=todo,
                                 show_progress=False, **SETTINGS, **crop_kw)
    status = {t: m.get(box) for t, m in manifests.items()}
    if all(s == "done" for s in status.values()):
        log.info(f"  processed in {(time.time() - t0) / 60:.1f} min")
        to_delete = list(targets.values()) if args.delete_existing else downloaded
        if not args.keep_tars:
            for fp in to_delete:
                fp.unlink(missing_ok=True)
                log.info(f"  deleted {fp.name}")
        return True
    log.info(f"  FAILED, tars kept for inspection: {status}")
    return False


def mosaic(crop_kw, partial, variants, expected=None):
    for term in variants:
        parts, suffix = OUTPUTS[term]
        done = {b for b, s in rls._load_manifest(parts / "manifest.json").items() if s == "done"}
        missing = sorted(set(expected or []) - done)
        if missing and not partial:
            log.info(f"mosaic {term}: {len(missing)} boxes not done yet ({', '.join(missing[:8])}...); "
                     "skipped (use --mosaic-partial)")
            continue
        all_fp = OUT_DIR / f"rusle_LS_all_5m{suffix}.tif"
        crop_fp = OUT_DIR / f"rusle_LS_crop_5m{suffix}.tif" if crop_kw else None
        rls.mosaic_ls(parts, all_fp, crop_fp, res_deg=SETTINGS["out_res_deg"])
        log.info(f"mosaic {term}: {len(done)} boxes -> {all_fp.relative_to(ROOT)}"
                 + (f" and {crop_fp.name}" if crop_fp else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", nargs="+", help="only these boxes (e.g. n30e000 s30e030)")
    ap.add_argument("--dry-run", action="store_true", help="log in and list; download and process nothing")
    ap.add_argument("--mosaic-only", action="store_true", help="skip downloads and processing")
    ap.add_argument("--mosaic-partial", action="store_true", help="mosaic even if some boxes are not done")
    ap.add_argument("--variants", nargs="+", choices=list(OUTPUTS), default=list(OUTPUTS),
                    help="LS layers to compute and mosaic (default: all)")
    ap.add_argument("--fresh", action="store_true",
                    help="start from scratch: move the selected parts folders aside (renamed, not deleted)")
    ap.add_argument("--keep-tars", action="store_true", help="do not delete the tars after processing")
    ap.add_argument("--delete-existing", action="store_true",
                    help="also delete tars that were already on disk (default: only the ones this run downloaded)")
    ap.add_argument("--file-list", help="CSV name,url of per-file links, used instead of listing the folders")
    ap.add_argument("--merit-dir", default=str(MERIT_DIR), help=f"download folder (default {MERIT_DIR})")
    ap.add_argument("--crop-scale", type=float, default=1.0, help="multiplier turning crop values into 0-1")
    args = ap.parse_args()

    setup_logging()
    crop_kw = dict(crop_fp=CROP_FP, crop_scale=args.crop_scale) if CROP_FP.exists() else {}
    log.info(f"settings: {SETTINGS}, variants {args.variants}, crop weights: "
             f"{CROP_FP.name if crop_kw else 'none (LS_all only)'}")

    variants = tuple(args.variants)
    if args.mosaic_only:
        mosaic(crop_kw, args.mosaic_partial, variants)
        return

    boxes = box_index(remote_files(args))
    wanted = [b for b in rls.merit_box_names() if b in boxes and (not args.boxes or b in args.boxes)]
    unknown = sorted(set(args.boxes or []) - set(boxes))
    if unknown:
        log.info(f"not available on Dropbox (ocean-only or misspelt): {unknown}")
    if args.fresh and not args.dry_run:
        move_aside(variants)
    todo = [b for b in wanted if args.fresh or not manifest_done(b, variants)]
    size = {b: sum(rf.size or 0 for _, rf in boxes[b].values()) for b in todo}
    log.info(f"{len(boxes)} boxes on Dropbox, {len(wanted)} selected, {len(wanted) - len(todo)} already done, "
             f"{len(todo)} to run ({gb(sum(size.values()))} to download)")
    if args.dry_run:
        for b in todo:
            log.info(f"  {b}: " + ", ".join(f"{rf.name} {gb(rf.size)}" for _, rf in boxes[b].values()))
        return

    failed = []
    for i, box in enumerate(todo, 1):
        log.info(f"[{i}/{len(todo)}] {box} ({gb(size[box])})")
        try:
            if not process_box(box, boxes[box], args, crop_kw, variants):
                failed.append(box)
        except (DropboxError, OSError) as exc:
            log.info(f"  FAILED: {exc}")
            failed.append(box)
    if failed:
        log.info(f"failed boxes (rerun to retry): {failed}")
    mosaic(crop_kw, args.mosaic_partial or bool(args.boxes), variants, expected=wanted)


if __name__ == "__main__":
    main()
