"""Download CRISPR-StAR (Elling et al. 2024) CRISPR screen data from GEO.

GEO accession: GSE262309
Dataset: Genome-wide (~22k gene) CRISPR screen in Yumm1.7 450R melanoma cells,
         paired in vitro and in vivo (subcutaneous syngeneic tumour, C57BL/6).

Downloads:
    - GEO SOFT file (GSE262309_family.soft.gz)
    - All supplementary files (gene-level scores)

Output directory layout:
    data/elling2024/raw/
        GSE262309_family.soft.gz
        <supplementary files>

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/01_download.py
    python scripts/elling2024/01_download.py --out-dir /path/to/custom/dir
"""
import argparse
import ftplib
import gzip
import hashlib
import logging
import tarfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEO_ACCESSION = "GSE262309"
GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
GEO_FTP_DIR = "/geo/series/GSE262nnn/GSE262309"
GEO_HTTP_BASE = f"https://ftp.ncbi.nlm.nih.gov{GEO_FTP_DIR}"
DEFAULT_OUT_DIR = Path("data/elling2024/raw")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 3) -> Path:
    """Download a file to dest, skipping if already present."""
    if dest.exists():
        log.info("Already exists: %s", dest)
        return dest
    log.info("Downloading %s -> %s", url, dest)
    for attempt in range(1, retries + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def list_supplementary_files() -> list[str]:
    """List supplementary file names for GSE262309 via NCBI FTP."""
    log.info("Listing supplementary files via FTP: %s/%s/suppl/", GEO_FTP_HOST, GEO_FTP_DIR)
    ftp = ftplib.FTP(GEO_FTP_HOST, timeout=60)
    ftp.login()
    suppl_dir = f"{GEO_FTP_DIR}/suppl"
    filenames: list[str] = []
    try:
        ftp.cwd(suppl_dir)
        filenames = ftp.nlst()
    except ftplib.error_perm as e:
        log.warning("FTP listing failed (%s); supplementary files may need manual download", e)
    finally:
        ftp.quit()
    log.info("Found %d supplementary files", len(filenames))
    return filenames


def download_soft(out_dir: Path) -> Path:
    """Download the GEO SOFT file."""
    url = f"{GEO_HTTP_BASE}/soft/{GEO_ACCESSION}_family.soft.gz"
    dest = out_dir / f"{GEO_ACCESSION}_family.soft.gz"
    return download_file(url, dest)


def download_supplementary(out_dir: Path) -> list[Path]:
    """Download all supplementary files."""
    filenames = list_supplementary_files()
    paths = []
    for fname in filenames:
        url = f"{GEO_HTTP_BASE}/suppl/{fname}"
        dest = out_dir / fname
        paths.append(download_file(url, dest))
    return paths


def try_geoparse_download(out_dir: Path) -> bool:
    """Attempt download via GEOparse as an alternative to direct FTP.

    Returns True if successful, False if GEOparse is not available.
    """
    try:
        import GEOparse  # type: ignore[import]
    except ImportError:
        log.info("GEOparse not available; using direct FTP/HTTP download")
        return False
    log.info("Downloading %s via GEOparse", GEO_ACCESSION)
    GEOparse.get_GEO(geo=GEO_ACCESSION, destdir=str(out_dir), silent=False)
    return True


def extract_raw_tar(out_dir: Path) -> None:
    """Extract GSE262309_RAW.tar into out_dir if not already extracted."""
    tar_path = out_dir / f"{GEO_ACCESSION}_RAW.tar"
    if not tar_path.exists():
        return
    # Check if already extracted (any .txt.gz file present)
    if list(out_dir.glob("*.txt.gz")):
        log.info("RAW tar already extracted (*.txt.gz files found)")
        return
    log.info("Extracting %s into %s", tar_path.name, out_dir)
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    extracted = list(out_dir.glob("*.txt.gz"))
    log.info("Extracted %d files from %s", len(extracted), tar_path.name)


def verify_downloads(out_dir: Path) -> None:
    """Basic sanity check on downloaded files."""
    files = list(out_dir.iterdir())
    if not files:
        raise RuntimeError(f"No files found in {out_dir}")
    log.info("Files in %s:", out_dir)
    for f in sorted(files):
        log.info("  %s  %.1f MB", f.name, f.stat().st_size / 1e6)

    # Verify gzip integrity of .gz files
    for f in out_dir.glob("*.gz"):
        try:
            with gzip.open(f, "rb") as gz:
                gz.read(1024)
        except Exception as exc:
            log.warning("Gzip integrity check failed for %s: %s", f.name, exc)
        else:
            log.info("  %s: gzip OK", f.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CRISPR-StAR Elling 2024 data from GEO (GSE262309)."
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for raw data (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--skip-geoparse",
        action="store_true",
        help="Skip GEOparse and use direct FTP/HTTP download only",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.resolve())

    # Attempt GEOparse first (handles supplementary files automatically)
    used_geoparse = False
    if not args.skip_geoparse:
        used_geoparse = try_geoparse_download(out_dir)

    # Fall back to direct download
    if not used_geoparse:
        log.info("Downloading SOFT file...")
        download_soft(out_dir)

        log.info("Downloading supplementary files...")
        suppl_paths = download_supplementary(out_dir)
        if not suppl_paths:
            log.warning(
                "No supplementary files downloaded via FTP. "
                "Check GEO manually: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s",
                GEO_ACCESSION,
            )

    extract_raw_tar(out_dir)
    verify_downloads(out_dir)
    log.info("Download complete. Next step: python scripts/elling2024/02_parse_scores.py")


if __name__ == "__main__":
    main()
