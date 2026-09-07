#!/usr/bin/env python3
"""Run the complete Round 10 train/validation workflow without supervision.

The runner is idempotent: completed stages are skipped, interrupted training is
resumed at an epoch boundary, and every subprocess has its own persistent log.
It deliberately evaluates the validation split only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT / "results/continuum_v1_macrostep_round10/formal"
)
ARMS = {
    "fno_single": "continuum_v1_macrostep_fno_single_dt0p1.json",
    "fno_history4": "continuum_v1_macrostep_fno_history4_dt0p1.json",
    "unet_history4": "continuum_v1_macrostep_unet_history4_dt0p1.json",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def gpu_free_gib(device: str) -> float | None:
    if not device.startswith("cuda:"):
        return None
    index = int(device.split(":", maxsplit=1)[1])
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip()) / 1024.0


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    marker: Path
    always_run: bool = False


class Workflow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.result_root.resolve()
        self.control = self.root / "_automation"
        self.logs = self.control / "logs"
        self.status_path = self.control / "status.json"
        self.lock_path = self.control / "runner.lock"
        self.state: dict[str, Any] = {
            "workflow": "continuum_v1_macrostep_round10",
            "state": "starting",
            "pid": os.getpid(),
            "device": args.device,
            "started_at": now(),
            "updated_at": now(),
            "diagnostic_test_opened": False,
            "tasks": {},
        }

    def save(self) -> None:
        self.state["updated_at"] = now()
        atomic_json(self.status_path, self.state)

    def acquire_lock(self) -> None:
        self.control.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            try:
                pid = int(self.lock_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                self.lock_path.unlink(missing_ok=True)
                return self.acquire_lock()
            raise RuntimeError(f"workflow is already running as PID {pid}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")

    def wait_for_gpu(self, task_name: str) -> None:
        threshold = self.args.minimum_free_gpu_gib
        if threshold <= 0.0 or not self.args.device.startswith("cuda:"):
            return
        while True:
            try:
                free = gpu_free_gib(self.args.device)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.state["gpu_check_warning"] = str(error)
                self.save()
                return
            self.state["free_gpu_memory_gib"] = free
            if free is not None and free >= threshold:
                self.save()
                return
            self.state["state"] = "waiting_for_gpu"
            self.state["current_task"] = task_name
            self.save()
            time.sleep(self.args.poll_seconds)

    def run_task(self, task: Task) -> None:
        if not task.always_run and valid_json(task.marker):
            self.state["tasks"][task.name] = {
                "state": "skipped_complete",
                "marker": str(task.marker),
                "updated_at": now(),
            }
            self.save()
            return

        self.wait_for_gpu(task.name)
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{task.name}.log"
        task_state = self.state["tasks"].setdefault(task.name, {})
        env = os.environ.copy()
        source_path = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        for attempt in range(1, self.args.retries + 2):
            task_state.update(
                {
                    "state": "running",
                    "attempt": attempt,
                    "command": list(task.command),
                    "log": str(log_path),
                    "started_at": now(),
                }
            )
            self.state["state"] = "running"
            self.state["current_task"] = task.name
            self.save()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\n[{now()}] attempt {attempt}: "
                    + " ".join(task.command)
                    + "\n"
                )
                log.flush()
                process = subprocess.Popen(
                    task.command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                task_state["subprocess_pid"] = process.pid
                self.save()
                while True:
                    try:
                        return_code = process.wait(timeout=self.args.heartbeat_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        task_state["heartbeat_at"] = now()
                        self.state["state"] = "running"
                        self.save()
            if return_code == 0 and valid_json(task.marker):
                task_state.update(
                    {
                        "state": "completed",
                        "completed_at": now(),
                        "marker": str(task.marker),
                        "return_code": return_code,
                    }
                )
                self.save()
                return
            task_state.update(
                {
                    "state": "retry_pending",
                    "return_code": return_code,
                    "updated_at": now(),
                }
            )
            self.save()
        task_state["state"] = "failed"
        raise RuntimeError(
            f"task {task.name} failed; see {log_path}"
        )

    def tasks(self) -> list[Task]:
        python = str(self.args.python.resolve())
        tasks: list[Task] = []
        for arm, config_name in ARMS.items():
            config = PROJECT_ROOT / "configs/training" / config_name
            for seed in self.args.seeds:
                seed_root = self.root / arm / f"seed{seed}"
                tasks.append(
                    Task(
                        name=f"train_{arm}_seed{seed}",
                        command=(
                            python,
                            str(PROJECT_ROOT / "scripts/train_continuum_v1_macrostep.py"),
                            "--config",
                            str(config),
                            "--output-dir",
                            str(seed_root),
                            "--seed",
                            str(seed),
                            "--device",
                            self.args.device,
                            "--resume",
                        ),
                        marker=seed_root / "summary.json",
                    )
                )
                rollout_root = seed_root / "validation_t80"
                tasks.append(
                    Task(
                        name=f"validate_{arm}_seed{seed}",
                        command=(
                            python,
                            str(
                                PROJECT_ROOT
                                / "scripts/run_continuum_v1_macrostep_rollouts.py"
                            ),
                            "--checkpoint",
                            str(seed_root / "best.pt"),
                            "--output-dir",
                            str(rollout_root),
                            "--split",
                            "validation",
                            "--maximum-time",
                            "80",
                            "--device",
                            self.args.device,
                        ),
                        marker=rollout_root / "summary.json",
                    )
                )

        closure_root = self.root / "closure_validation_benchmark"
        tasks.append(
            Task(
                name="benchmark_frozen_closure_validation",
                command=(
                    python,
                    str(
                        PROJECT_ROOT
                        / "scripts/benchmark_continuum_v1_closure_validation.py"
                    ),
                    "--dataset-root",
                    self.args.dataset_root,
                    "--checkpoint",
                    str(self.args.closure_checkpoint.resolve()),
                    "--output-dir",
                    str(closure_root),
                    "--device",
                    self.args.device,
                    "--max-time",
                    "80",
                ),
                marker=closure_root / "summary.json",
            )
        )
        summary_path = self.root / "round10_validation_summary.json"
        tasks.append(
            Task(
                name="summarize_validation",
                command=(
                    python,
                    str(PROJECT_ROOT / "scripts/summarize_continuum_v1_macrostep_round10.py"),
                    "--root",
                    str(self.root),
                    "--output",
                    str(summary_path),
                ),
                marker=summary_path,
                always_run=True,
            )
        )
        return tasks

    def run(self) -> None:
        self.acquire_lock()
        (self.control / "DONE").unlink(missing_ok=True)
        self.save()
        try:
            for task in self.tasks():
                self.run_task(task)
            self.state["state"] = "completed"
            self.state["current_task"] = None
            self.state["completed_at"] = now()
            self.save()
            (self.control / "DONE").write_text(f"completed {now()}\n", encoding="utf-8")
        except BaseException as error:
            self.state["state"] = "failed"
            self.state["error"] = f"{type(error).__name__}: {error}"
            self.state["failed_at"] = now()
            self.save()
            raise
        finally:
            self.lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            "/wangx/home/duxinxu/miniconda3/envs/landau-pic-surrogate/bin/python"
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--minimum-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dataset-root",
        default="/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1",
    )
    parser.add_argument(
        "--closure-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT / "results/continuum_v1_closure_baseline/seed1/best.pt"
        ),
    )
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.heartbeat_seconds <= 0.0:
        parser.error("--heartbeat-seconds must be positive")
    if not args.python.exists():
        parser.error(f"Python interpreter does not exist: {args.python}")
    return args


def main() -> None:
    Workflow(parse_args()).run()


if __name__ == "__main__":
    main()
