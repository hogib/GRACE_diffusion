"""Collect per-paper status from the survey notes and merge the BibTeX files.

Usage: python -I collect.py NOTES_DIR OUT_DIR

Writes:
  OUT_DIR/papers.tsv        key, source, status, doi, eprint, title
  OUT_DIR/references.bib    deduplicated BibTeX (all verified entries)
  OUT_DIR/keymap.tsv        duplicate key -> canonical key
"""
import re
import sys
from pathlib import Path

SOURCES = ["grace_background", "dl_grace", "gravimetry_ml_processing",
           "diffusion_core", "diffusion_geoscience", "imonet"]
STATUSES = ["FULLTEXT_AVAILABLE_UNREAD", "FULLTEXT_READ", "PAYWALLED",
            "ABSTRACT_ONLY", "UNVERIFIED"]


def split_entries(text):
    """Return list of (type, key, raw) for each @entry, brace-balanced."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        if m.group(1).lower() in ("comment", "string", "preamble"):
            continue
        depth, i = 0, text.index("{", m.start())
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append((m.group(1).lower(), m.group(2), text[m.start():j + 1]))
                    break
    return out


def field(raw, name):
    m = re.search(r"\b%s\s*=\s*[{\"](.*?)[}\"]\s*[,}]\s*" % name, raw, re.I | re.S)
    if not m:
        return ""
    return re.sub(r"[{}\s]+", " ", m.group(1)).strip()


def norm_doi(d):
    d = d.lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:60]


def status_for(md, key, keys):
    """Status of `key`: first status token after the key's first mention in
    the Papers section, before the next paper key is mentioned."""
    start = md.find("## Papers")
    body = md[start:] if start >= 0 else md
    end_sec = body.find("## Paywalled")
    if end_sec > 0:
        body = body[:end_sec]
    pos = body.find(key)
    if pos < 0:
        return "NOT_IN_NOTES"
    nxt = len(body)
    for k in keys:
        if k == key:
            continue
        p = body.find(k, pos + len(key))
        # only a *different* key that is not a prefix/suffix collision
        if p > pos and p < nxt and not body[p - 1].isalnum() and not body[p + len(k):p + len(k) + 1].isalnum():
            nxt = p
    chunk = body[pos:nxt]
    found = [(chunk.find(s), s) for s in STATUSES if s in chunk]
    if not found:
        return "UNKNOWN"
    return min(found)[1]


def main(notes, out):
    notes, out = Path(notes), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, canon, keymap, merged = [], {}, [], []
    rank = {"FULLTEXT_READ": 0, "FULLTEXT_AVAILABLE_UNREAD": 1, "PAYWALLED": 2,
            "ABSTRACT_ONLY": 3, "UNVERIFIED": 4, "UNKNOWN": 5, "NOT_IN_NOTES": 6}
    for src in SOURCES:
        bib, md = notes / f"{src}.bib", notes / f"{src}.md"
        if not bib.exists():
            print(f"missing {bib}", file=sys.stderr)
            continue
        entries = split_entries(bib.read_text())
        mdtext = md.read_text() if md.exists() else ""
        keys = [k for _, k, _ in entries]
        for typ, key, raw in entries:
            doi = norm_doi(field(raw, "doi"))
            eprint = field(raw, "eprint")
            title = field(raw, "title")
            st = "IN_HOUSE" if src == "imonet" else status_for(mdtext, key, keys)
            ident = ("doi:" + doi) if doi else ("arxiv:" + eprint) if eprint else ("t:" + norm_title(title))
            if ident in canon:
                ck = canon[ident]
                keymap.append((key, ck))
                # keep the best status across sources
                for r in rows:
                    if r["key"] == ck and rank.get(st, 9) < rank.get(r["status"], 9):
                        r["status"] = st
                        r["source"] += "+" + src
                continue
            if any(r["key"] == key for r in rows):
                key2 = key + "_" + src[:3]
                raw = raw.replace(key, key2, 1)
                keymap.append((key, key2))
                key = key2
            canon[ident] = key
            # escape bare ampersands (e.g. "Earth & Environment") outside url/doi fields
            raw = re.sub(r"(?m)^(\s*(?!url|doi)\w+\s*=.*?)(?<!\\)&", r"\1\\&", raw)
            raw = re.sub(r"(?<![\\\w])&(?=\s)", r"\\&", raw)
            rows.append(dict(key=key, source=src, status=st, doi=doi, eprint=eprint, title=title))
            merged.append(raw)
    with open(out / "papers.tsv", "w") as f:
        f.write("key\tsource\tstatus\tdoi\teprint\ttitle\n")
        for r in rows:
            f.write("\t".join(r[c] for c in ["key", "source", "status", "doi", "eprint", "title"]) + "\n")
    with open(out / "keymap.tsv", "w") as f:
        for a, b in keymap:
            f.write(f"{a}\t{b}\n")
    (out / "references.bib").write_text(
        "% Merged from survey/notes/*.bib by survey/tools/collect.py. Do not edit by hand.\n\n"
        + "\n\n".join(merged) + "\n")
    from collections import Counter
    print(Counter(r["status"] for r in rows))
    print("entries", len(rows), "duplicates", len(keymap))


if __name__ == "__main__":
    main(*sys.argv[1:3])
