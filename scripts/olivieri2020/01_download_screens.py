"""Download Olivieri 2020 genotoxic CRISPR screens from BioGRID ORCS.

Fetches 30 screens (screen 1328 ICRF-187 excluded — QC fail) via the
BioGRID ORCS DataTables server-side API and writes:
  - data/olivieri2020/normz_matrix.parquet  (genes × screens, DrugZ NormZ)
  - data/olivieri2020/normz_long.parquet    (long format)
  - data/olivieri2020/screen_metadata.parquet
  - data/olivieri2020/gene_entrez.parquet   (gene_symbol → entrez_id lookup)
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from crispr_al.io import save_parquet

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_GENE_ID_RE = re.compile(r"/Gene/(\d+)")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Screen 1328 (ICRF-187) excluded — failed QC in Olivieri et al. 2020.
SCREENS = {
    1313: ("Cisplatin_TKOv2",       "Cisplatin",      "TKOv2"),
    1314: ("Camptothecin_TKOv2",    "Camptothecin",   "TKOv2"),
    1315: ("Etoposide_TKOv2",       "Etoposide",      "TKOv2"),
    1316: ("Hydroxyurea_TKOv2",     "Hydroxyurea",    "TKOv2"),
    1317: ("IonizingRadiation_TKOv2", "IonizingRadiation", "TKOv2"),
    1318: ("Doxorubicin_TKOv2",     "Doxorubicin",    "TKOv2"),
    1319: ("H2O2_TKOv2",            "H2O2",           "TKOv2"),
    1320: ("MMS_TKOv2",             "MMS",            "TKOv2"),
    1321: ("Pyridostatin_TKOv2",    "Pyridostatin",   "TKOv2"),
    1322: ("UV_TKOv2",              "UV",             "TKOv2"),
    1323: ("Bleomycin_TKOv3",       "Bleomycin",      "TKOv3"),
    1324: ("Olaparib_TKOv3",        "Olaparib",       "TKOv3"),
    1325: ("AZD6738_TKOv3",         "AZD6738",        "TKOv3"),
    1326: ("Cisplatin2_repA_TKOv3", "Cisplatin",      "TKOv3"),
    1327: ("Cisplatin2_repB_TKOv3", "Cisplatin",      "TKOv3"),
    1329: ("Formaldehyde_TKOv3",    "Formaldehyde",   "TKOv3"),
    1330: ("PhenDC3_TKOv3",         "PhenDC3",        "TKOv3"),
    1331: ("DuocarmycinSA_TKOv3",   "DuocarmycinSA",  "TKOv3"),
    1332: ("Trabectedin_TKOv3",     "Trabectedin",    "TKOv3"),
    1333: ("Calicheamicin_TKOv3",   "Calicheamicin",  "TKOv3"),
    1334: ("Gemcitabine_TKOv2",     "Gemcitabine",    "TKOv2"),
    1335: ("IlludinS_TKOv3",        "IlludinS",       "TKOv3"),
    1336: ("MLN4924_TKOv3",         "MLN4924",        "TKOv3"),
    1337: ("MNNG_TKOv3",            "MNNG",           "TKOv3"),
    1338: ("KBrO3_TKOv3",           "KBrO3",          "TKOv3"),
    1339: ("CD437_TKOv3",           "CD437",          "TKOv3"),
    1340: ("Camptothecin2_TKOv3",   "Camptothecin",   "TKOv3"),
    1341: ("BPDE_TKOv3",            "BPDE",           "TKOv3"),
    1342: ("Hydroxyurea2_TKOv3",    "Hydroxyurea",    "TKOv3"),
    1343: ("PladienolideB_TKOv3",   "PladienolideB",  "TKOv3"),
}

BASE_URL = "https://orcs.thebiogrid.org"
DATATABLES_URL = f"{BASE_URL}/scripts/datatableTools.php"


def _fetch_screen(session: requests.Session, screen_id: int, page_size: int = 5000) -> pd.DataFrame:
    """Fetch all gene scores for one BioGRID ORCS screen.

    Returns DataFrame with columns: gene_symbol, entrez_id, normz.
    Rows with empty gene_symbol or non-finite normz are dropped.
    Duplicate gene_symbols are collapsed by mean (6 genes appear twice).
    """
    r0 = session.get(f"{BASE_URL}/Screen/{screen_id}", timeout=20)
    r0.raise_for_status()
    soup = BeautifulSoup(r0.text, "html.parser")

    hidden = {
        (inp.get("id") or inp.get("name")): inp.get("value", "")
        for inp in soup.find_all("input", type="hidden")
        if inp.get("id") or inp.get("name")
    }
    total_records = next(
        (inp.get("value", "20000") for inp in soup.find_all("input", type="hidden")
         if not inp.get("id") and not inp.get("name")),
        "20000",
    )

    col_payload = json.dumps({
        "tool": "serverSideHeader", "type": "scores",
        "screenID": str(screen_id),
        "isHit": hidden.get("isHit", "false"),
        "significanceIndicator": hidden.get("significanceIndicator", ""),
        "significanceDetails": hidden.get("significanceDetails", ""),
        "scoreCols": hidden.get("scoreCols", "1"),
    })
    col_defs = session.post(DATATABLES_URL, data={"expData": col_payload}, timeout=30).json()
    n_cols = len(col_defs)

    def _build_payload(start: int) -> dict:
        d = {
            "draw": "1", "start": str(start), "length": str(page_size),
            "search[value]": "", "search[regex]": "false",
            "tool": "serverSideRows", "totalRecords": total_records,
            "checkedBoxes": "{}", "type": "scores",
            "screenID": str(screen_id),
            "isHit": hidden.get("isHit", "false"),
            "significanceIndicator": hidden.get("significanceIndicator", ""),
            "significanceDetails": hidden.get("significanceDetails", ""),
            "scoreCols": hidden.get("scoreCols", "1"),
        }
        for i in range(n_cols):
            d[f"columns[{i}][data]"] = str(i)
            d[f"columns[{i}][name]"] = ""
            d[f"columns[{i}][searchable]"] = "true"
            d[f"columns[{i}][orderable]"] = "true"
            d[f"columns[{i}][search][value]"] = ""
            d[f"columns[{i}][search][regex]"] = "false"
        hit_col = hidden.get("hitCol", "5")
        score_dir = hidden.get("scoreOrderDir", "asc")
        if hidden.get("isHit") == "true":
            d["order[0][column]"] = hit_col
            d["order[0][dir]"] = "asc"
            d["order[1][column]"] = "4"
            d["order[1][dir]"] = score_dir
        else:
            d["order[0][column]"] = "4"
            d["order[0][dir]"] = score_dir
        return d

    resp = session.post(DATATABLES_URL, data={"expData": json.dumps(_build_payload(0))}, timeout=120).json()
    all_rows = resp["data"]
    total = int(resp["recordsTotal"])

    start = page_size
    while start < total:
        batch = session.post(
            DATATABLES_URL,
            data={"expData": json.dumps(_build_payload(start))},
            timeout=120,
        ).json()["data"]
        all_rows.extend(batch)
        start += page_size
        time.sleep(0.2)

    records = []
    for row in all_rows:
        gene = _HTML_TAG_RE.sub("", str(row[0])).strip()
        m = _GENE_ID_RE.search(str(row[0]))
        entrez = m.group(1) if m else ""
        score = _HTML_TAG_RE.sub("", str(row[4])).strip()
        records.append({"gene_symbol": gene, "entrez_id": entrez, "normz": score})

    df = pd.DataFrame(records)
    df["normz"] = pd.to_numeric(df["normz"], errors="coerce")
    df = df[df["gene_symbol"].str.strip() != ""].dropna(subset=["normz"])
    # Collapse the 6 genes that appear twice; entrez_id kept from first occurrence
    entrez_map = df.drop_duplicates("gene_symbol").set_index("gene_symbol")["entrez_id"]
    df = df.groupby("gene_symbol")["normz"].mean().reset_index()
    df["entrez_id"] = df["gene_symbol"].map(entrez_map)
    return df


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    })

    screen_series = {}
    gene_entrez = None  # populated from the first screen; all screens share same gene universe
    for screen_id, (label, drug, library) in SCREENS.items():
        logger.info("Fetching screen %d (%s)", screen_id, label)
        session.headers["Referer"] = f"{BASE_URL}/Screen/{screen_id}"
        df = _fetch_screen(session, screen_id)
        screen_series[label] = df.set_index("gene_symbol")["normz"].rename(label)
        if gene_entrez is None:
            gene_entrez = (
                df[["gene_symbol", "entrez_id"]]
                .dropna(subset=["entrez_id"])
                .query("entrez_id != ''")
                .reset_index(drop=True)
            )
        time.sleep(0.5)

    normz_matrix = pd.concat(screen_series.values(), axis=1, join="outer")
    logger.info("normz_matrix shape: %s, missing: %.2f%%",
                normz_matrix.shape,
                normz_matrix.isna().mean().mean() * 100)

    normz_long = normz_matrix.reset_index().melt(
        id_vars="gene_symbol", var_name="screen_label", value_name="normz"
    ).dropna()

    meta_rows = [
        {"screen_id": sid, "screen_label": label, "drug": drug, "library": lib}
        for sid, (label, drug, lib) in SCREENS.items()
    ]
    screen_meta = pd.DataFrame(meta_rows)

    save_parquet(normz_matrix, str(out / "normz_matrix.parquet"))
    save_parquet(normz_long, str(out / "normz_long.parquet"))
    save_parquet(screen_meta, str(out / "screen_metadata.parquet"))
    save_parquet(gene_entrez, str(out / "gene_entrez.parquet"))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="data/olivieri2020",
        help="Output directory for downloaded data (default: data/olivieri2020)",
    )
    args = parser.parse_args()
    main(args.out_dir)
