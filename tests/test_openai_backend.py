import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from indibench.config import load_config
from indibench.runner import Evaluator


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        user = body["messages"][-1]["content"]
        has_img = isinstance(user, list) and any(p["type"] == "image_url" for p in user)
        reply = "ஆம்" if has_img else "Answer: B"
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_compatible_backend(fixture_dir, tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = load_config(None, {
            "model.backend": "openai", "model.name_or_path": "fake-vlm",
            "model.api_base": f"http://127.0.0.1:{srv.server_port}/v1",
            "run.tasks": ["local_mcq_ta", "local_vqa_xl_ta"], "run.task_files": [str(fixture_dir / "tasks.yaml")],
            "run.output_dir": str(tmp_path), "run.run_name": "api"}, [])
        out = Evaluator(cfg).run()["results"]
        assert out["local_mcq_ta"]["n"] == 6
        assert out["local_vqa_xl_ta"]["metrics"]["script_fidelity"]["value"] == 100.0
        img_req = [b for b in _Handler.seen if isinstance(b["messages"][-1]["content"], list)]
        assert img_req and img_req[0]["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        srv.shutdown()
