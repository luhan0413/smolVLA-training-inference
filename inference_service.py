"""Local HTTP inference for the trained UR7e SmolVLA checkpoint."""
import argparse
import base64
import binascii
import io
import json
import logging
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "outputs/train/ur7e_20260911_053556_3425920/checkpoints/020000_64/pretrained_model"
MAX_BODY = 8 * 1024 * 1024


def decode_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, list) or len(state) != 7 or any(
        type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4e38 for x in state
    ):
        raise ValueError("state must contain 7 finite numbers: x,y,z,rx,ry,rz,gripper")
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip() or len(task) > 2000:
        raise ValueError("task must be a nonempty string, at most 2000 characters")
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str):
        raise ValueError("image_base64 must contain a base64 JPEG or PNG")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("JPEG", "PNG") or im.size != (640, 480):
                raise ValueError("left camera image must be a 640x480 JPEG or PNG")
            rgb = np.array(im.convert("RGB"), copy=True)
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid JPEG/PNG image_base64") from exc
    return {
        "observation.state": torch.tensor(state, dtype=torch.float32),
        "observation.images.left": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
        "task": task,
    }


class Engine:
    def __init__(self, checkpoint, device):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig

        self.checkpoint = str(Path(checkpoint).resolve())
        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.device = device
        cfg.load_vlm_weights = False  # All weights come from the fine-tuned checkpoint.
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint, config=cfg, strict=True).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device = device

    @torch.inference_mode()
    def predict(self, observation):
        start = time.perf_counter()
        self.policy.reset()
        batch = self.pre(observation)
        actions = self.post(self.policy.predict_action_chunk(batch))[0].cpu()
        if actions.shape != (50, 7) or not torch.isfinite(actions).all():
            raise RuntimeError("model returned invalid actions")
        return {"actions": actions.tolist(), "shape": list(actions.shape), "fps": 10,
                "action_names": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"],
                "inference_ms": round((time.perf_counter() - start) * 1000, 2)}


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "not found"})
            self.respond(200, {"status": "ready", "checkpoint": engine.checkpoint, "device": engine.device})

        def do_POST(self):
            if self.path != "/infer":
                return self.respond(404, {"error": "not found"})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported; send Content-Length")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, {"error": "body must be 1 byte to 8 MiB"})
                observation = decode_request(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeDecodeError) as exc:
                return self.respond(400, {"error": str(exc)})
            try:
                self.respond(200, engine.predict(observation))
            except Exception:
                logging.exception("Inference failed")
                self.respond(500, {"error": "inference failed; see service log"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = Engine(args.checkpoint, args.device)
    # Serial requests prevent concurrent mutation of policy/processor state.
    server = HTTPServer((args.host, args.port), handler_for(engine))
    logging.info("Ready at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
