"""Create CVAT tasks for fixtures and export their tracks (MOT 1.1).

  uv run scripts/cvat_tasks.py create fast_motion [single_target ...]
  uv run scripts/cvat_tasks.py export fast_motion
  uv run scripts/cvat_tasks.py list
  uv run scripts/cvat_tasks.py import-prelabel non_coco --label object [--occluded 1:0,2:40]

Credentials: tools/cvat_admin.txt. Exports land in data/fixtures/<name>.gt.zip.
"""

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS = ROOT / "tools" / "cvat_admin.txt"
FIXTURES = ROOT / "data" / "fixtures"
PROJECT = "VectraX fixtures"
LABELS = ["cup", "person", "object"]  # object: outside COCO classes (R1)
EXPORT_FORMAT = "MOT 1.1"
IMAGE_QUALITY = 95
POLL_S = 2
TIMEOUT_S = 900


class Cvat:
    def __init__(self):
        cred = dict(line.split(": ", 1) for line in CREDENTIALS.read_text().splitlines())
        self._url = cred["url"]
        key = self._call("POST", "/api/auth/login", {"username": cred["username"], "password": cred["password"]})["key"]
        self._headers = {"Authorization": f"Token {key}"}

    def _call(self, method, path, body=None, raw=None, content_type="application/json"):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self._url + path, data=data, method=method)
        for k, v in getattr(self, "_headers", {}).items():
            req.add_header(k, v)
        if data is not None:
            req.add_header("Content-Type", content_type)

        with urllib.request.urlopen(req) as resp:
            payload = resp.read()

        return json.loads(payload) if payload else None

    def download(self, path, dest):
        req = urllib.request.Request(self._url + path, headers=self._headers)
        with urllib.request.urlopen(req) as resp:
            dest.write_bytes(resp.read())

    def project_id(self):
        found = self._call("GET", f"/api/projects?name={urllib.request.quote(PROJECT)}")["results"]
        if found:
            return found[0]["id"]

        return self._call("POST", "/api/projects", {"name": PROJECT, "labels": [{"name": n} for n in LABELS]})["id"]

    def task_id(self, name):
        found = self._call("GET", f"/api/tasks?name={urllib.request.quote(name)}")["results"]
        return found[0]["id"] if found else None

    def wait(self, rq_id):
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            status = self._call("GET", f"/api/requests/{rq_id}")
            if status["status"] == "finished":
                return status

            if status["status"] == "failed":
                raise SystemExit(f"CVAT request failed: {status.get('message')}")

            time.sleep(POLL_S)

        raise SystemExit(f"CVAT request {rq_id} timed out")

    def create(self, name):
        if self.task_id(name) is not None:
            print(f"{name}: task exists, skipped")
            return

        video = FIXTURES / f"{name}.mp4"
        task = self._call("POST", "/api/tasks", {"name": name, "project_id": self.project_id()})
        body, ctype = _multipart({"image_quality": str(IMAGE_QUALITY)}, video)
        t0 = time.monotonic()
        rq = self._call("POST", f"/api/tasks/{task['id']}/data", raw=body, content_type=ctype)
        self.wait(rq["rq_id"])
        frames = self._call("GET", f"/api/tasks/{task['id']}/data/meta")["size"]
        expected = json.loads(video.with_suffix(".json").read_text())["frames"]
        check = "ok" if frames == expected else f"MISMATCH (sidecar {expected})"
        print(f"{name}: task {task['id']}, {frames} frames {check}, processed in {time.monotonic() - t0:.1f}s")

    def export(self, name):
        task_id = self.task_id(name)
        if task_id is None:
            raise SystemExit(f"{name}: no task")

        fmt = urllib.request.quote(EXPORT_FORMAT)
        rq = self._call("POST", f"/api/tasks/{task_id}/dataset/export?format={fmt}&save_images=false")
        status = self.wait(rq["rq_id"])
        dest = FIXTURES / f"{name}.gt.zip"
        self.download(status["result_url"].removeprefix(self._url), dest)
        print(f"{name}: exported to {dest}")

    def import_prelabel(self, name, label, occluded):
        """SAM 2 keyframes (sam2_prelabel.py) → one track per object. A keyframe
        without a mask ends the track with an outside shape until it reappears."""
        task_id = self.task_id(name)
        job = self._call("GET", f"/api/jobs?task_id={task_id}")["results"][0]["id"]
        ann = self._call("GET", f"/api/jobs/{job}/annotations")
        if ann["tracks"] or ann["shapes"]:
            raise SystemExit(f"{name}: job {job} already has annotations; not overwriting")

        project = self._call("GET", f"/api/tasks/{task_id}")["project_id"]
        labels = {lb["name"]: lb["id"] for lb in self._call("GET", f"/api/labels?project_id={project}")["results"]}
        pre = json.loads((FIXTURES / f"{name}.sam2.json").read_text())
        tracks = []
        for obj, seq in pre["objects"].items():
            shapes, last = [], None
            for frame, (box, _) in zip(pre["keyframes"], seq, strict=True):
                if box is None:
                    if last is not None:
                        shapes.append(_rect(frame, last, outside=True))
                        last = None
                    continue

                last = [float(v) for v in box]
                shapes.append(_rect(frame, last, occluded=(obj, frame) in occluded))

            tracks.append({"frame": shapes[0]["frame"], "label_id": labels[label], "group": 0,
                           "source": "semi-auto", "shapes": shapes, "attributes": []})

        self._call("PATCH", f"/api/jobs/{job}/annotations?action=create",
                   {"version": ann["version"], "tags": [], "shapes": [], "tracks": tracks})
        for t in self._call("GET", f"/api/jobs/{job}/annotations")["tracks"]:
            outside = [s["frame"] for s in t["shapes"] if s["outside"]]
            print(f"{name}: job {job} track {t['id']} keyframes={len(t['shapes'])} outside_at={outside}")

    def list(self):
        for t in self._call("GET", "/api/tasks?page_size=100")["results"]:
            print(f"{t['id']:3d} {t['name']:18s} frames={t['size']} status={t['status']}")


def _rect(frame, points, outside=False, occluded=False):
    return {"type": "rectangle", "frame": frame, "points": points, "outside": outside,
            "occluded": occluded, "z_order": 0, "rotation": 0.0, "attributes": []}


def _multipart(fields, file_path):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())

    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="client_files[0]"; filename="{file_path.name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["create", "export", "list", "import-prelabel"])
    p.add_argument("names", nargs="*")
    p.add_argument("--label", choices=LABELS, help="import-prelabel: label for all tracks")
    p.add_argument("--occluded", default="", help="import-prelabel: obj:frame[,obj:frame] keyframes to flag")
    args = p.parse_args()

    cvat = Cvat()
    if args.action == "list":
        cvat.list()
        return

    if args.action == "import-prelabel":
        occluded = {tuple(item.split(":")) for item in args.occluded.split(",") if item}
        occluded = {(obj, int(frame)) for obj, frame in occluded}
        for name in args.names:
            cvat.import_prelabel(name, args.label, occluded)
        return

    if not args.names:
        sys.exit("give fixture names")

    for name in args.names:
        getattr(cvat, args.action)(name)


if __name__ == "__main__":
    main()
