"""
FlowGuard Penang: Zero-Dependency Municipal API & Live Web Server.

Bridges the FlowGuardEngine computer-vision backend to the interactive web UI
using Python's built-in http.server (no external framework installation needed).
Also exposes live Python source snippets via inspect.getsource() to satisfy the
Hackathon Anti-"Black Box" technical defense rule.
"""

from __future__ import annotations

import ast
import base64
import csv
import inspect
import json
import mimetypes
import os
import threading
import urllib.parse
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import model_engine
import train_model
from model_engine import FlowGuardEngine, ZONE_METADATA

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "sample_drains"
HAZARD_LOG_PATH = DATA_DIR / "hazard_log.json"
MANIFEST_PATH = DATA_DIR / "samples_manifest.json"
METRICS_PATH = DATA_DIR / "validation_metrics.json"
QA_LOG_PATH = DATA_DIR / "qa_debug_log.csv"

_DATA_LOCK = threading.Lock()
_ENGINE: Optional[FlowGuardEngine] = None


def get_engine() -> FlowGuardEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FlowGuardEngine(ROOT)
    return _ENGINE


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_qa_csv() -> List[Dict[str, str]]:
    if not QA_LOG_PATH.exists():
        return []
    with QA_LOG_PATH.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _decode_base64_image(data_str: str) -> bytes:
    if "," in data_str and data_str.strip().startswith("data:"):
        _, b64_part = data_str.split(",", 1)
    else:
        b64_part = data_str
    return base64.b64decode(b64_part, validate=True)


def _parse_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, str], Optional[bytes], str]:
    header_bytes = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=email_default_policy).parsebytes(header_bytes + body)
    fields: Dict[str, str] = {}
    file_bytes: Optional[bytes] = None
    filename = "upload.jpg"
    if msg.is_multipart():
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            fname = part.get_filename()
            payload = part.get_payload(decode=True)
            if fname is not None and payload is not None:
                file_bytes = payload
                filename = fname
            elif name and payload is not None:
                fields[name] = payload.decode("utf-8", errors="replace")
    return fields, file_bytes, filename


def _extract_function_source(file_path: Path, func_name: str) -> str:
    """Safely extract a function source block from a Python file without importing optional frameworks."""
    source_text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                return segment
    return f"# Function {func_name} not found in {file_path.name}"


def _build_source_inspection() -> Dict[str, Dict[str, str]]:
    """Return real source snippets from the running Python backend for Anti-Black-Box Q&A."""
    return {
        "frontend": {
            "title": "01. Streamlit / Web UI Cache Bridge",
            "file": "app.py",
            "symbol": "get_flowguard_engine()",
            "beginner_note": "Stores the AI model in memory once so clicking buttons stays instant instead of reloading files every time.",
            "expert_note": "Uses @st.cache_resource (and singleton _ENGINE in server.py) to persist weights and OpenCV cascade buffers across requests.",
            "code": _extract_function_source(ROOT / "app.py", "get_flowguard_engine"),
        },
        "preprocess": {
            "title": "02. Image Preprocessing (224x224 + LAB CLAHE)",
            "file": "model_engine.py",
            "symbol": "preprocess_drain_image()",
            "beginner_note": "Crops the middle square of the photo, resizes to 224x224, and evens out dark afternoon tree shadows.",
            "expert_note": "Converts BGR to LAB, lifts deep shadow troughs on L-channel via Gaussian background estimation + CLAHE, and normalizes RGB to [-1.0, 1.0].",
            "code": inspect.getsource(model_engine.preprocess_drain_image),
        },
        "model": {
            "title": "03. MobileNetV2 Transfer Learning & Texture Features",
            "file": "train_model.py / model_engine.py",
            "symbol": "build_mobilenetv2_keras_model() + FlowGuardEngine._predict_head()",
            "beginner_note": "Reads debris textures (crumpled bags, leaves, grate bars) instead of guessing from dark color alone.",
            "expert_note": "MobileNetV2 depthwise-separable backbone with Week 10 RandomBrightness(0.25)/RandomContrast(0.25) augmentation + 18-dim Gabor/Laplacian/FFT neural head.",
            "code": inspect.getsource(train_model.build_mobilenetv2_keras_model),
        },
        "privacy": {
            "title": "04. Privacy Shield & Municipal Dispatch Log",
            "file": "model_engine.py",
            "symbol": "enforce_privacy_shield()",
            "beginner_note": "Checks every photo for human faces and blurs them out before anything is displayed or saved.",
            "expert_note": "Executes cached OpenCV Haar Cascade frontal-face detection and adaptive Gaussian blur prior to tensor conversion.",
            "code": inspect.getsource(model_engine.enforce_privacy_shield),
        },
    }


class FlowGuardHandler(BaseHTTPRequestHandler):
    server_version = "FlowGuardPenang/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output concise and clean
        pass

    def _send_json(self, status: int, data: Any) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "File not found."})
            return
        mime, _ = mimetypes.guess_type(str(path))
        if path.suffix == ".js":
            mime = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif path.suffix == ".html":
            mime = "text/html; charset=utf-8"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            safe_path = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in safe_path.parents and safe_path != STATIC_DIR.resolve():
                self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Forbidden path."})
                return
            self._send_file(safe_path)
            return

        if path.startswith("/sample_drains/"):
            rel = path[len("/sample_drains/") :]
            safe_path = (SAMPLES_DIR / rel).resolve()
            if SAMPLES_DIR.resolve() not in safe_path.parents:
                self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Forbidden path."})
                return
            self._send_file(safe_path)
            return

        if path == "/api/health":
            engine = get_engine()
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "backend_mode": engine.backend_mode, "status": "ready"},
            )
            return

        if path == "/api/samples":
            samples = _read_json(MANIFEST_PATH, [])
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "samples": samples, "zones": ZONE_METADATA},
            )
            return

        if path == "/api/hazards":
            with _DATA_LOCK:
                hazards = _read_json(HAZARD_LOG_PATH, [])
            self._send_json(HTTPStatus.OK, {"ok": True, "hazards": hazards, "zones": ZONE_METADATA})
            return

        if path == "/api/metrics":
            metrics = _read_json(METRICS_PATH, {})
            qa_rows = _read_qa_csv()
            engine = get_engine()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "metrics": metrics,
                    "qa_log": qa_rows,
                    "backend_mode": engine.backend_mode,
                    "source_stages": _build_source_inspection(),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            self._handle_analyze()
            return

        if path == "/api/hazards":
            self._handle_create_hazard()
            return

        if path == "/api/benchmark":
            self._handle_benchmark()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown POST endpoint: {path}"})

    def do_PATCH(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/hazards/"):
            hazard_id = urllib.parse.unquote(path[len("/api/hazards/") :]).strip("/")
            self._handle_update_hazard(hazard_id)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown PATCH endpoint: {path}"})

    def _handle_analyze(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        raw_body = self._read_body_bytes()
        engine = get_engine()

        zone = "Jalan P. Ramlee"
        apply_shadow_fix = True
        source: Any = None
        source_name = "uploaded_photo.jpg"

        try:
            if "multipart/form-data" in content_type:
                fields, file_bytes, fname = _parse_multipart(content_type, raw_body)
                zone = fields.get("zone", zone)
                apply_shadow_fix = fields.get("apply_shadow_fix", "true").lower() != "false"
                sample_id = fields.get("sample_id", "")
                if file_bytes is not None:
                    source = file_bytes
                    source_name = fname
                elif sample_id:
                    manifest = _read_json(MANIFEST_PATH, [])
                    match = next((s for s in manifest if s["id"] == sample_id), None)
                    if match is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Unknown sample_id: {sample_id}"})
                        return
                    source = SAMPLES_DIR / match["filename"]
                    source_name = match["filename"]
            else:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                zone = str(payload.get("zone", zone))
                apply_shadow_fix = bool(payload.get("apply_shadow_fix", True))
                if payload.get("image_base64"):
                    source = _decode_base64_image(str(payload["image_base64"]))
                    source_name = str(payload.get("filename", "uploaded_drain.jpg"))
                elif payload.get("sample_id"):
                    sample_id = str(payload["sample_id"])
                    manifest = _read_json(MANIFEST_PATH, [])
                    match = next((s for s in manifest if s["id"] == sample_id), None)
                    if match is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Unknown sample_id: {sample_id}"})
                        return
                    source = SAMPLES_DIR / match["filename"]
                    source_name = match["filename"]
                else:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": "Provide either an uploaded photo or a Penang sample selection."},
                    )
                    return
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"Malformed analysis request: {exc}"},
            )
            return

        if zone not in ZONE_METADATA:
            zone = "Jalan P. Ramlee"

        result = engine.run_inference(source, zone=zone, apply_shadow_fix=apply_shadow_fix)
        if not result.get("ok"):
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, result)
            return

        result["source_photo"] = source_name
        self._send_json(HTTPStatus.OK, result)

    def _handle_create_hazard(self) -> None:
        try:
            payload = json.loads(self._read_body_bytes().decode("utf-8") or "{}")
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON body."})
            return

        zone = str(payload.get("zone", "")).strip()
        severity = str(payload.get("severity", "")).strip()
        if zone not in ZONE_METADATA:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Select a valid Penang corridor zone."})
            return
        if severity not in ("Clear", "Moderate", "Critical"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Severity must be Clear, Moderate, or Critical."})
            return

        with _DATA_LOCK:
            hazards: List[Dict[str, Any]] = _read_json(HAZARD_LOG_PATH, [])
            new_id = f"FG-2026-{101 + len(hazards)}"
            status = "Cleared" if severity == "Clear" else "Pending Dispatch"
            record = {
                "id": new_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M MYT"),
                "zone": zone,
                "severity": severity,
                "confidence": float(payload.get("confidence", 0.0)),
                "latency_ms": float(payload.get("latency_ms", 0.0)),
                "status": status,
                "source_photo": str(payload.get("source_photo", "field_upload.jpg")),
                "notes": str(payload.get("notes", "")).strip() or f"{severity} drain condition logged in {zone}.",
            }
            hazards.insert(0, record)
            _write_json(HAZARD_LOG_PATH, hazards)

        self._send_json(HTTPStatus.CREATED, {"ok": True, "record": record, "hazards": hazards})

    def _handle_update_hazard(self, hazard_id: str) -> None:
        try:
            payload = json.loads(self._read_body_bytes().decode("utf-8") or "{}")
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON body."})
            return

        new_status = str(payload.get("status", "")).strip()
        if new_status not in ("Pending Dispatch", "Crew Dispatched", "Cleared"):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Status must be Pending Dispatch, Crew Dispatched, or Cleared."},
            )
            return

        updated_record: Optional[Dict[str, Any]] = None
        with _DATA_LOCK:
            hazards: List[Dict[str, Any]] = _read_json(HAZARD_LOG_PATH, [])
            for item in hazards:
                if item.get("id") == hazard_id:
                    item["status"] = new_status
                    updated_record = item
                    break
            if updated_record is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Hazard ID {hazard_id} not found."})
                return
            _write_json(HAZARD_LOG_PATH, hazards)

        self._send_json(HTTPStatus.OK, {"ok": True, "record": updated_record, "hazards": hazards})

    def _handle_benchmark(self) -> None:
        engine = get_engine()
        manifest = _read_json(MANIFEST_PATH, [])
        runs: List[Dict[str, Any]] = []
        latencies: List[float] = []
        for _pass in range(2):
            for item in manifest:
                res = engine.run_inference(SAMPLES_DIR / item["filename"], zone=item["zone"], apply_shadow_fix=True)
                if res.get("ok"):
                    latencies.append(float(res["latency_ms"]))
                    if _pass == 1:
                        runs.append(
                            {
                                "filename": item["filename"],
                                "expected": item["expected"],
                                "predicted": res["label"],
                                "confidence": res["confidence"],
                                "latency_ms": res["latency_ms"],
                            }
                        )
        summary = {
            "ok": True,
            "runs_count": len(latencies),
            "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "min_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_ms": round(max(latencies), 2) if latencies else 0.0,
            "latest_pass": runs,
        }
        self._send_json(HTTPStatus.OK, summary)


def run_server(host: str = "127.0.0.1", port: int = 8501) -> None:
    # Pre-warm engine at startup before serving HTTP requests
    engine = get_engine()
    httpd = ThreadingHTTPServer((host, port), FlowGuardHandler)
    print(f"FlowGuard Penang server running at http://{host}:{port}")
    print(f"Inference backend: {engine.backend_mode}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FlowGuard Penang server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    env_port = int(os.environ.get("PORT", "8501"))
    run_server(port=env_port)
