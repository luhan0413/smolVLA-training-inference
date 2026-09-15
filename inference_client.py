"""Example: python inference_client.py --image left.png --state 0 0 0 0 0 0 0 --task 'open drawer'"""
import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8085")
    parser.add_argument("--image", required=True)
    parser.add_argument("--state", nargs=7, type=float, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    body = {"state": args.state, "task": args.task,
            "image_base64": base64.b64encode(Path(args.image).read_bytes()).decode()}
    request = Request(args.url.rstrip("/") + "/infer", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
