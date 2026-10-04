# downloads the pwtt benchmark dataset (~139mb zipped): building shapes, damage
# labels and sentinel-1 stats for 24 cities in ukraine, gaza, syria and iraq.
# no account needed, it's a public google drive link.
#
# source: https://github.com/oballinger/PWTT
#
# usage:
#   python scripts/download_benchmark.py

from __future__ import annotations

import zipfile
from pathlib import Path

import gdown

DRIVE_FILE_ID = "12RsrfU8m-cvtONohD6FBcF21OPoEA_Mf"
OUT_DIR = Path("data/benchmark")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "pwtt_benchmark.zip"

    if (OUT_DIR / "one_month").exists():
        print(f"{OUT_DIR / 'one_month'} already there, skipping download.")
        return

    gdown.download(id=DRIVE_FILE_ID, output=str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
    zip_path.unlink()

    # zip has some mac os junk files in it, drop them
    macosx_dir = OUT_DIR / "__MACOSX"
    if macosx_dir.exists():
        import shutil
        shutil.rmtree(macosx_dir)

    print(f"extracted to {OUT_DIR}")


if __name__ == "__main__":
    main()
