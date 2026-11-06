#!/usr/bin/env python3
"""Pack the thesis into a deterministic Overleaf upload.

The bundle must be self-contained: `main.tex`, `refs.bib`, every section and every
generated table fragment.  It must also be reproducible byte for byte, so the
zip is written with fixed timestamps and permissions and is accompanied by a
SHA-256 manifest of exactly the files inside it.

macOS `._*` resource-fork files are never shipped: they are not UTF-8, they are
not LaTeX, and a build that includes them behaves differently from one that does
not.  Figures and front matter are shipped with the source, and `--verify` fails
on a missing figure, an overfull box or an empty list of figures as well as on
a LaTeX error or an undefined reference.

Usage:
    python3 scripts/pack_overleaf.py            # write dist/overleaf
    python3 scripts/pack_overleaf.py --verify   # also rebuild in a temp dir
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "thesis"
OUT = ROOT / "dist" / "overleaf"
ZIP_NAME = "azurite-thesis-overleaf.zip"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def included() -> list[Path]:
    files: list[Path] = [THESIS / "main.tex", THESIS / "refs.bib"]
    for pattern in (
        "frontmatter/*.tex",
        "sections/*.tex",
        "tables/*.tex",
        "figures/*.pdf",
    ):
        files.extend(sorted(THESIS.glob(pattern)))
    for extra in THESIS.glob("*.sty"):
        files.append(extra)
    out = []
    for p in files:
        if not p.exists() or p.name.startswith("."):
            continue
        if p.name.startswith("._"):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(THESIS).as_posix())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(verify: bool) -> int:
    files = included()
    for name in ("main.tex", "refs.bib"):
        if not any(p.name == name for p in files):
            raise SystemExit(f"missing required file: thesis/{name}")

    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT / ZIP_NAME
    manifest = OUT / "MANIFEST.sha256"
    if zip_path.exists():
        zip_path.unlink()

    lines = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arc = path.relative_to(THESIS).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  ./{arc}")
    manifest.write_text("\n".join(lines) + "\n")

    print(f"packed {len(files)} files into {zip_path.relative_to(ROOT)}")
    print(f"wrote {manifest.relative_to(ROOT)} ({len(lines)} entries)")

    if verify:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            engine = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
            if not Path(engine).exists():
                print("verify: no tectonic on PATH; bundle not compiled", file=sys.stderr)
                return 2
            proc = subprocess.run(
                [engine, "-X", "compile", "main.tex", "--keep-logs", "--keep-intermediates"],
                cwd=tmp, capture_output=True, text=True,
            )
            log = Path(tmp) / "main.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            bad = [ln for ln in text.splitlines() if ln.startswith("!")]
            undefined = [
                ln for ln in text.splitlines()
                if "Undefined control sequence" in ln
                or ("undefined" in ln and ("Citation" in ln or "Reference" in ln))
                or "There were undefined references" in ln
                or "Missing character" in ln
            ]
            overfull = [ln for ln in text.splitlines() if ln.startswith("Overfull")]
            pdf = Path(tmp) / "main.pdf"
            pages = 0
            if pdf.exists():
                out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
                for line in out.splitlines():
                    if line.startswith("Pages:"):
                        pages = int(line.split()[1])
            lof = Path(tmp) / "main.lof"
            figures = 0
            if lof.exists():
                figures = sum(
                    1 for ln in lof.read_text(errors="replace").splitlines()
                    if "contentsline {figure}" in ln
                )
            wanted: set[str] = set()
            for source in Path(tmp).rglob("*.tex"):
                for line in source.read_text(errors="replace").splitlines():
                    if "\\includegraphics" not in line or "%" in line.split("\\includegraphics")[0]:
                        continue
                    start, end = line.find("{", line.find("\\includegraphics")), line.rfind("}")
                    if start != -1 and end > start:
                        name = line[start + 1:end].strip()
                        if name and not name.startswith("#"):
                            wanted.add(name if name.endswith(".pdf") else name + ".pdf")
            missing_figures = sorted(
                name for name in wanted if not (Path(tmp) / "figures" / name).is_file()
            )
            print(
                f"verify: exit={proc.returncode} pages={pages} figures={figures} "
                f"errors={len(bad)} undefined={len(undefined)} overfull={len(overfull)}"
            )
            if proc.returncode != 0 or bad or undefined or overfull or figures == 0 or missing_figures:
                print(f"verify: missing figure assets: {missing_figures}", file=sys.stderr)
                print(text[-3000:])
                return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="compile the extracted bundle")
    args = ap.parse_args()
    return build(args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
