"""Download GRACE/GRACE-FO monthly spherical-harmonic solutions and ISIMIP3a model TWS.

Sources are provisional (design doc, 00-design.md): we use what we can obtain
now and keep the source behind a small table so it can be swapped.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

ITSG = "https://ftp.tugraz.at/pub/ITSG/GRACE"
SERIES = {
    # name: (listing URL, file regex). Unconstrained (unfiltered) monthly solutions.
    "itsg2018": (f"{ITSG}/ITSG-Grace2018/monthly/monthly_n{{n}}/",
                 r"ITSG-Grace2018_n{n}_(\d{{4}}-\d{{2}})\.gfc"),
    "itsg_operational": (f"{ITSG}/ITSG-Grace_operational/monthly/monthly_n{{n}}/",
                         r"ITSG-Grace_operational_n{n}_(\d{{4}}-\d{{2}})\.gfc"),
}


def list_files(series: str, nmax: int = 120) -> list[tuple[str, str]]:
    """Return (month 'YYYY-MM', url) pairs for one series."""
    base, pattern = SERIES[series]
    base, pattern = base.format(n=nmax), pattern.format(n=nmax)
    html = requests.get(base, timeout=60).text
    names = sorted(set(re.findall(r'href="(%s)"' % pattern.replace("(\\d{4}-\\d{2})", "[0-9-]+"), html)))
    out = []
    for name in names:
        m = re.fullmatch(pattern, name)
        if m:
            out.append((m.group(1), base + name))
    return out


def fetch_series(series: str, dest: Path, nmax: int = 120, pause: float = 0.2) -> int:
    """Download all monthly files of a series into dest/series/n{nmax}/. Skips existing files."""
    folder = dest / series / f"n{nmax}"
    folder.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for month, url in list_files(series, nmax):
        path = folder / f"{month}.gfc"
        if path.exists() and path.stat().st_size > 0:
            continue
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        text = r.text
        if "end_of_head" not in text:
            raise RuntimeError(f"{url}: not a gfc file")
        tmp = path.with_suffix(".part")
        tmp.write_text(text)
        tmp.rename(path)
        n_new += 1
        time.sleep(pause)
    return n_new


# ISIMIP3a global water models: total water storage, GSWP3-W5E5 obsclim, histsoc, default
# runs, monthly 1901-2019, 0.5 deg (design doc 01, section 3). Public, no login.
ISIMIP_API = "https://data.isimip.org/api/v1/datasets/"
ISIMIP_QUERY = dict(simulation_round="ISIMIP3a", sector="water_global", variable="tws",
                    climate_forcing="gswp3-w5e5", climate_scenario="obsclim",
                    soc_scenario="histsoc", sens_scenario="default", page_size=100)


def list_isimip() -> list[tuple[str, str]]:
    """Return (model, url) for every ISIMIP3a TWS dataset matching ISIMIP_QUERY."""
    r = requests.get(ISIMIP_API, params=ISIMIP_QUERY, timeout=60)
    r.raise_for_status()
    out = []
    for d in r.json()["results"]:
        for f in d["files"]:
            out.append((d["specifiers"]["model"], f["file_url"]))
    return sorted(out)


def fetch_isimip(dest: Path) -> int:
    """Download the ISIMIP3a TWS files into dest/isimip/. Skips existing files."""
    folder = dest / "isimip"
    folder.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for model, url in list_isimip():
        path = folder / url.rsplit("/", 1)[1]
        if path.exists() and path.stat().st_size > 0:
            continue
        tmp = path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.rename(path)
        print(model, path.name, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
        n_new += 1
    return n_new


if __name__ == "__main__":
    import sys

    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    which = sys.argv[2:] or ["grace"]
    if "grace" in which:
        for s in SERIES:
            print(s, "new files:", fetch_series(s, dest))
    if "isimip" in which:
        print("isimip new files:", fetch_isimip(dest))
