"""Create CVAT tasks for fixtures and export their tracks (MOT 1.1).

  uv run scripts/cvat_tasks.py create fast_motion [single_target ...]
  uv run scripts/cvat_tasks.py export fast_motion
  uv run scripts/cvat_tasks.py list

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

    def list(self):
        for t in self._call("GET", "/api/tasks?page_size=100")["results"]:
            print(f"{t['id']:3d} {t['name']:18s} frames={t['size']} status={t['status']}")


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
    p.add_argument("action", choices=["create", "export", "list"])
    p.add_argument("names", nargs="*")
    args = p.parse_args()

    cvat = Cvat()
    if args.action == "list":
        cvat.list()
        return

    if not args.names:
        sys.exit("give fixture names")

    for name in args.names:
        getattr(cvat, args.action)(name)


if __name__ == "__main__":
    main()
