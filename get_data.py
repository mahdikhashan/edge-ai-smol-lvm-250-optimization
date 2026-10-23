"""Download the parts of the two EVLM datasets (Farahani et al., EDGE 2026, Section V-B) that the evaluation needs.

- clips:   UZH power line dataset (Xing et al., IROS 2023, ref. [35]). 720x480 frames at 30 fps. Each environment
           folder is one continuous flight that was split at random into train/val/test, so the frames are sorted
           by frame number and cut into 4 s clips (120 frames).
- insplad: InsPLAD-fault supervised fault classification crops (Vieira e Silva et al., 2023, ref. [43]).

Both servers accept HTTP range requests, so only the needed files are downloaded (not the 25 GB and 6.4 GB zips).
Standard library only.
"""

import argparse
import io
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UZH_URL = "https://download.ifi.uzh.ch/rpg/powerline_tracking_dataset/power_line_dataset.zip"
INSPLAD_URL = (
    "https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded"
)
# paper environment (Fig. 3) -> (UZH folder, position of the clip in the flight). The paper does not name the
# folders, so this mapping is ours. powerlines_16_city flies over the city only near the start (grass and fog later).
ENVS = {
    "forest": ("powerline_1_forest", 0.5),
    "city1": ("powerlines_16_city", 0.1),
    "mountain": ("powerline_7_mountain", 0.5),
    "city2": ("powerline_5_japan", 0.5),
    "sand": ("powerline_3_sand", 0.5),
    "warehouse": ("powerline_11_warehouse", 0.5),
    "wasteland": ("powerline_4_wasteland", 0.5),
    "industrial": ("powerline_2_industrial", 0.5),
}


class Remote(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    HEADERS = {"User-Agent": "curl/8"}  # the Mendeley front end rejects the default urllib agent

    def __init__(self, url):
        self.src, self.pos = url, 0
        self._resolve()

    def _resolve(self):  # follow redirects once; signed S3 links expire, so this runs again on a 403
        req = urllib.request.Request(self.src, headers={**self.HEADERS, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.url, self.size = r.url, int(r.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(2):
            req = urllib.request.Request(self.url, headers={**self.HEADERS, "Range": f"bytes={self.pos}-{end}"})
            try:
                with urllib.request.urlopen(req) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 403 or attempt:
                    raise
                self._resolve()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url, buffer_mb=1):
    return zipfile.ZipFile(io.BufferedReader(Remote(url), buffer_size=buffer_mb << 20))


def get_clips(out, envs, clips, frames):
    z = remote_zip(UZH_URL)
    for env in envs:
        folder, where = ENVS[env]
        names = {}
        for info in z.infolist():
            m = re.match(rf"datasets/{folder}/images/\w+/frame_(\d+)\.png$", info.filename)
            if m:
                names[int(m.group(1))] = info.filename
        order = sorted(names)
        spots = [where] if clips == 1 else [(k + 1) / (clips + 1) for k in range(clips)]  # more clips: spread out
        for k, spot in enumerate(spots):
            start = max(0, int(len(order) * spot) - frames // 2)
            dst = out / env / f"clip_{k}"
            dst.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(order[start : start + frames]):
                path = dst / f"{i:04d}.png"
                if not path.exists():
                    path.write_bytes(z.read(names[n]))
            print(f"clips: {env} ({folder}) clip_{k}: {frames} frames from frame {order[start]}")


def get_insplad(out):
    out.mkdir(parents=True, exist_ok=True)
    inner = out / "supervised_fault_classification.zip"
    if not inner.exists():
        with remote_zip(INSPLAD_URL, buffer_mb=8).open("supervised_fault_classification.zip") as src:
            with open(inner.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(src, f, 8 << 20)
        inner.with_suffix(".part").rename(inner)
    with zipfile.ZipFile(inner) as z:
        z.extractall(out)
    inner.unlink()
    print(f"insplad: extracted to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--what", default="clips,insplad")
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--clips", type=int, default=1, help="4 s clips per environment")
    ap.add_argument("--frames", type=int, default=120, help="frames per clip (120 = 4 s at 30 fps)")
    args = ap.parse_args()

    out = Path(args.out)
    what = args.what.split(",")
    if "clips" in what:
        get_clips(out / "powerline_clips", args.envs.split(","), args.clips, args.frames)
    if "insplad" in what:
        get_insplad(out / "insplad")


if __name__ == "__main__":
    main()
