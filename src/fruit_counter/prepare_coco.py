"""Build a YOLO-format orange/apple dataset from COCO 2017 annotations.

Only images containing an orange or apple are downloaded, one by one from
images.cocodataset.org (about 3.4k images, ~0.55 GB) instead of the 19 GB zip.
COCO annotations are CC BY 4.0; each image keeps its own Flickr license, which
is recorded in ``licenses.csv``.

Splits: COCO train2017 -> ``train`` and ``val`` (a deterministic hash of the
image id holds out ``val_fraction``); COCO val2017 -> ``test``.
Images with a crowd box (one box around a pile of fruit) are skipped: keeping
them without that box would teach the model that the pile is background.
"""

import csv
import hashlib
import json
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TO_CLASS = {"orange": 0, "apple": 1}


def _download(url: str, path: Path, attempts: int = 4) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fruit-counter"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as file:
                while chunk := response.read(1 << 20):
                    file.write(chunk)
            temporary.replace(path)
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def _load_annotations(cache: Path, coco_split: str) -> dict:
    json_path = cache / f"instances_{coco_split}.json"
    if not json_path.exists():
        zip_path = cache / "annotations_trainval2017.zip"
        print(f"downloading {ANNOTATIONS_URL} (253 MB)", flush=True)
        _download(ANNOTATIONS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            for name in ("instances_train2017.json", "instances_val2017.json"):
                with archive.open(f"annotations/{name}") as source, (cache / name).open("wb") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
        zip_path.unlink()
    return json.loads(json_path.read_text())


def _is_val(image_id: int, fraction: float) -> bool:
    digest = hashlib.sha256(str(image_id).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < fraction


def select_images(coco: dict) -> dict[int, dict]:
    """Return {image_id: {"image": ..., "boxes": [(class_id, x, y, w, h), ...]}}."""
    categories = {c["id"]: COCO_TO_CLASS[c["name"]] for c in coco["categories"] if c["name"] in COCO_TO_CLASS}
    images = {image["id"]: image for image in coco["images"]}
    boxes = defaultdict(list)
    crowded = set()
    for annotation in coco["annotations"]:
        class_id = categories.get(annotation["category_id"])
        if class_id is None:
            continue
        if annotation.get("iscrowd"):
            crowded.add(annotation["image_id"])
        else:
            boxes[annotation["image_id"]].append((class_id, *annotation["bbox"]))
    return {
        image_id: {"image": images[image_id], "boxes": items}
        for image_id, items in boxes.items()
        if image_id not in crowded
    }


def yolo_lines(boxes: list, width: int, height: int) -> list[str]:
    lines = []
    for class_id, x, y, w, h in boxes:
        if w < 1 or h < 1:
            continue
        lines.append(
            f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    return lines


def prepare(output: Path, cache: Path, val_fraction: float = 0.1, workers: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    summary = defaultdict(lambda: {"images": 0, "orange": 0, "apple": 0})
    license_rows = []
    for coco_split in ("train2017", "val2017"):
        coco = _load_annotations(cache, coco_split)
        licenses = {item["id"]: item["name"] for item in coco["licenses"]}
        for image_id, item in sorted(select_images(coco).items()):
            image = item["image"]
            if coco_split == "val2017":
                split = "test"
            else:
                split = "val" if _is_val(image_id, val_fraction) else "train"
            stem = f"coco_{image_id:012d}"
            label_path = output / split / "labels" / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(yolo_lines(item["boxes"], image["width"], image["height"])) + "\n")
            jobs.append((image["coco_url"], output / split / "images" / f"{stem}.jpg"))
            summary[split]["images"] += 1
            for class_id, *_ in item["boxes"]:
                summary[split]["orange" if class_id == 0 else "apple"] += 1
            license_rows.append((split, stem, image["coco_url"], licenses.get(image["license"], ""),
                                 image.get("flickr_url", "")))
        del coco

    print(f"downloading {len(jobs)} images", flush=True)
    failures = []

    def fetch(job):
        url, path = job
        try:
            _download(url, path)
        except OSError as error:
            failures.append((url, str(error)))

    with ThreadPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(fetch, jobs), start=1):
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed, e.g. {failures[0]}; rerun to resume")

    with (output / "licenses.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "file", "coco_url", "license", "flickr_url"])
        writer.writerows(license_rows)
    result = {"source": "COCO 2017 (annotations CC BY 4.0)", "splits": dict(summary)}
    (output / "dataset.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
