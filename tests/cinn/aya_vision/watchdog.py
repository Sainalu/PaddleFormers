"""Run a bounded child process group with exclusive evidence files."""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from common import ROOT, sha256, write_json


def stop(process, grace):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--grace-s", type=float, default=5)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.log.exists() or args.status.exists() or args.timeout_s <= 0:
        parser.error("positive deadline and new evidence paths required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("child command required")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"watchdog received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    timed_out = False
    interrupted = False
    with args.log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        next_heartbeat = started + 10
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= args.timeout_s:
                    timed_out = True
                    stop(process, args.grace_s)
                    break
                if now >= next_heartbeat:
                    print(f"HEARTBEAT elapsed={now - started:.1f}s log_bytes={args.log.stat().st_size}", flush=True)
                    next_heartbeat = now + 10
                try:
                    process.wait(timeout=min(1, max(0.01, args.timeout_s - (now - started))))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            interrupted = True
            stop(process, args.grace_s)
            raise
        finally:
            if process.poll() is None:
                stop(process, args.grace_s)
            report = {"command": command, "timeout_s": args.timeout_s, "timed_out": timed_out,
                      "interrupted": interrupted, "child_returncode": process.returncode,
                      "returncode": 124 if timed_out else process.returncode,
                      "elapsed_s": time.monotonic() - started,
                      "log": {"path": os.path.relpath(args.log, ROOT), "sha256": sha256(args.log)}}
            write_json(args.status, report)
            print(f"PROCESS_DONE returncode={report['returncode']} seconds={report['elapsed_s']:.2f}", flush=True)
    raise SystemExit(report["returncode"])


if __name__ == "__main__":
    main()
