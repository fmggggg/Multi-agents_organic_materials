"""JSON-lines client for the persistent tool workers (each lives in its own environment)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
from ..config import APPTAINER, APPTAINER_BINDS, MACE_CACHE, MACE_PY, MCRT_SIF


class Worker:
    def __init__(self, cmd: list[str], log_path: Path, env: dict | None = None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1, env=env)
        self.info = self._read()

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited (code {self.proc.poll()}); see {self.log.name}")
        return json.loads(line)

    def call(self, **req) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = self._read()
        if "error" in resp:
            raise RuntimeError(f"worker error: {resp['error']}\n{resp.get('traceback', '')}")
        return resp

    def close(self):
        try:
            self.proc.stdin.write(json.dumps(dict(cmd="quit")) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
        self.log.close()


def _gpu_env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "2")
    env["MPLCONFIGDIR"] = f"/tmp/mplconfig_{os.getuid()}"
    return env


def start_mace(log_dir: Path, gpu: str = "0", model: str = "small") -> Worker:
    env = _gpu_env(gpu)
    env["XDG_CACHE_HOME"] = MACE_CACHE
    return Worker([MACE_PY, "-u", str(HERE / "mace_worker.py"), "--model", model], log_dir / "mace_worker.log", env)


def start_mcrt(log_dir: Path, gpu: str = "0") -> Worker:
    env = _gpu_env(gpu)
    env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = gpu
    env["APPTAINERENV_OMP_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["APPTAINERENV_TRANSFORMERS_CACHE"] = f"/tmp/hf_{os.getuid()}"
    env["APPTAINERENV_MPLCONFIGDIR"] = f"/tmp/mplconfig_{os.getuid()}"
    binds = [p for p in (*APPTAINER_BINDS.split(","), f"/run/user/{os.getuid()}") if p and os.path.isdir(p)]
    return Worker([APPTAINER, "exec", "--nv", "--bind", ",".join(binds), MCRT_SIF, "python", "-u", str(HERE / "mcrt_worker.py")],
                  log_dir / "mcrt_worker.log", env)
