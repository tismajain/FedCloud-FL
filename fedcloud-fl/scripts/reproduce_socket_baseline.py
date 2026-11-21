"""Run original Flower or FedAsync processes with isolated logs and a timeout.

Unlike run_experiment.py, this wrapper never reuses upstream tracked log filenames.
It makes no changes to server/client implementations or their evaluation methodology.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--port", type=int, default=18391)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--mode", choices=["sync", "async"], default="sync")
    p.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    p.add_argument("--emulate-wan", action="store_true",
                   help="Use the upstream per-client WAN profiles instead of loopback timing")
    a = p.parse_args()
    if a.clients < 1 or a.rounds < 1:
        p.error("--clients and --rounds must be positive")
    out, parts = a.output.resolve(), a.parts.resolve()
    required_parts = [parts / "server_test.npz"] + [
        parts / f"client_{cid}.npz" for cid in range(a.clients)
    ]
    missing = [str(path) for path in required_parts if not path.is_file()]
    if missing:
        p.error("missing partition files: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=False)
    emulation = [] if a.emulate_wan else ["--no-emu"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               FLWR_TELEMETRY_ENABLED="0")
    common = ["--rounds", str(a.rounds), "--clients", str(a.clients), "--seed", str(a.seed),
              "--test", str(parts / "server_test.npz"), "--out", str(out / "run.json"), "--tag", str(parts.name)]
    py = sys.executable
    if a.mode == "sync":
        server = [py, "-m", "fedcloud.sync_server", "--address", f"127.0.0.1:{a.port}",
                  "--strategy", a.strategy] + common
        client = lambda cid: [py, "-m", "fedcloud.sync_client", "--server", f"127.0.0.1:{a.port}",
                              "--cid", str(cid), "--data", str(parts / f"client_{cid}.npz")] + emulation
    else:
        server = [py, "-m", "fedcloud.async_server", "--host", "127.0.0.1", "--port", str(a.port)] + common
        client = lambda cid: [py, "-m", "fedcloud.async_client", "--server", f"http://127.0.0.1:{a.port}",
                              "--cid", str(cid), "--data", str(parts / f"client_{cid}.npz")] + emulation
    processes, handles = [], []
    start = time.monotonic()
    try:
        for name, cmd in [("server", server)] + [(f"client_{cid}", client(cid)) for cid in range(a.clients)]:
            handle = (out / f"{name}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
            if name == "server":
                time.sleep(3)
                if processes[0].poll() is not None:
                    raise RuntimeError("Server exited before clients started; inspect server.log")
        rc = processes[0].wait(timeout=a.timeout)
        if rc:
            raise RuntimeError(f"Server failed with exit code {rc}; inspect server.log")
        for child in processes[1:]:
            if child.wait(timeout=30):
                raise RuntimeError("Client failed; inspect client logs")
        print(f"Original {a.mode} {a.strategy} completed in {time.monotonic()-start:.2f}s: {out / 'run.json'}")
    finally:
        for child in processes:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
