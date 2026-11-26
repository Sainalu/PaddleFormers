"""Bound a new process session and retain logs, timeout status and real exit codes."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from run import write_json


def terminate_group(process, grace):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run(command, log, status, timeout, grace=3):
    if log.exists() or status.exists():
        raise ValueError("watchdog evidence path already exists")
    if timeout <= 0 or grace < 0:
        raise ValueError("invalid watchdog time budget")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"watchdog received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    timed_out = False
    interrupted = False
    next_heartbeat = started + 10
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    timed_out = True
                    terminate_group(process, grace)
                    break
                if now >= next_heartbeat:
                    print(f"WATCHDOG elapsed_s={now - started:.1f} budget_s={timeout:g} log_bytes={log.stat().st_size}",
                          flush=True)
                    next_heartbeat = now + 10
                try:
                    process.wait(timeout=min(1, max(0.01, timeout - (now - started))))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            interrupted = True
            terminate_group(process, grace)
            raise
        finally:
            if process.poll() is None:
                terminate_group(process, grace)
            result = {"command": command, "pid": process.pid, "process_group": process.pid,
                      "timeout_s": timeout, "timed_out": timed_out, "interrupted": interrupted,
                      "child_returncode": process.returncode, "elapsed_s": time.monotonic() - started,
                      "returncode": 124 if timed_out else process.returncode,
                      "log": str(log.resolve())}
            write_json(status, result)
            print(f"WATCHDOG_DONE returncode={result['returncode']} elapsed_s={result['elapsed_s']:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--grace-s", type=float, default=3)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("a child command is required")
    result = run(command, arguments.log, arguments.status, arguments.timeout_s, arguments.grace_s)
    raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
