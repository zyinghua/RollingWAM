"""Keep the SageMaker checkpoint prefix small enough to restore.

Why this exists
---------------
SageMaker restores the ENTIRE ``checkpoint_s3_uri`` prefix into
/opt/ml/checkpoints *before* the container starts. Past some size that restore
fails outright, and the job dies with a completely opaque

    InternalServerError: We encountered an internal error. Please try again.

with ZERO CloudWatch log streams, because the container never ran. Measured on
the libero job (93 GB per checkpoint):

    92 GB  (1 checkpoint)   restore OK, container starts, auto-resume works
    369 GB (4 checkpoints)  InternalServerError, container never starts

The trainer has no retention policy, so any long run walks itself into this.

Policy
------
Newest checkpoint stays in ``checkpoint_s3_uri`` (small restore). The previous
one is MOVED to ``<parent>/_archive/<job>/`` — outside the restore path, so it
costs nothing at startup, but is still there if the newest turns out to have
been cut mid-write by a spot reclaim. Without that spare, a corrupt newest makes
auto-resume reject everything and silently restart from step 0.

The archive deliberately does NOT live at ``<prefix>-archive/``. S3 prefix
matching is plain string matching and ``CheckpointConfig.S3Uri`` carries no
trailing slash, so ``<prefix>-archive/...`` starts with ``<prefix>`` and may be
pulled into the very restore the archive exists to stay out of. Measured on the
vlaspot run: 369 GB under the prefix proper, 646 GB once the sibling archive is
counted. Nesting it under a ``_archive/`` directory cannot collide either way.

Anything older than that is deleted. Local copies under /opt/ml/checkpoints are
removed too, otherwise SageMaker's sync just re-uploads what we pruned.

Runs as a forked daemon from entry.py (see ``spawn``) on EVERY host. Only the
main host mutates S3; the others just reconcile their own local directory
against it. That split matters on multi-node jobs: DeepSpeed writes shards from
every rank, so a main-host-only pruner deletes its own local copy and S3's, and
then the *other* hosts' untouched copies get synced straight back. The vlaspot
2-node run archived the same step_002500 every 22 minutes for a day and a half
— 303 log lines, prefix never shrank — while the 1-node libero runs pruned
correctly.

Every failure is swallowed: pruning must never be able to kill training.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"^step_(\d+)/?$")
LOCAL_STEP_RE = re.compile(r"^step_(\d+)$")


def _run(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def s3_steps(prefix: str) -> list[int]:
    """Step numbers of the ``step_NNNNNN/`` dirs directly under ``prefix``."""
    rc, out = _run(["aws", "s3", "ls", prefix.rstrip("/") + "/"])
    if rc != 0:
        return []
    steps = []
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "PRE":
            m = STEP_RE.match(parts[-1])
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def local_steps(local_state: Path) -> list[int]:
    """Step numbers of the ``step_NNNNNN/`` dirs that exist locally."""
    if not local_state.is_dir():
        return []
    steps = []
    try:
        for p in local_state.iterdir():
            m = LOCAL_STEP_RE.match(p.name)
            if m and p.is_dir():
                steps.append(int(m.group(1)))
    except OSError:
        return []
    return sorted(steps)


def drop_local(steps, local_state: Path, local_weights: Path, say) -> None:
    for step in steps:
        name = f"step_{step:06d}"
        try:
            d = local_state / name
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            f = local_weights / f"{name}.pt"
            if f.is_file():
                f.unlink()
        except OSError as exc:
            say(f"local cleanup of {name} failed (harmless): {exc}")


def reconcile_local(state_s3: str, local_state: Path, local_weights: Path,
                    verbose: bool = True) -> None:
    """Non-main hosts: drop local checkpoints the main host already pruned.

    S3 is the source of truth, since only the main host writes it. Deletes only
    steps strictly OLDER than the newest step present in S3 — a step newer than
    that has simply not been synced up yet, and removing it would throw away a
    fresh checkpoint.
    """
    def say(msg: str) -> None:
        if verbose:
            print(f"[prune] {msg}", flush=True)

    remote = s3_steps(state_s3)
    if not remote:
        return
    newest = max(remote)
    stale = [s for s in local_steps(local_state) if s < newest and s not in remote]
    if stale:
        say(f"local-only cleanup of {stale} (already pruned from S3 by the main host)")
        drop_local(stale, local_state, local_weights, say)


def prune_once(state_s3: str, weights_s3: str, archive_s3: str, local_state: Path,
               local_weights: Path, keep: int = 1, verbose: bool = True) -> None:
    """Keep ``keep`` newest in place, archive the next, delete the rest."""
    steps = s3_steps(state_s3)
    if len(steps) <= keep:
        return
    newest = steps[-keep:]
    to_archive = steps[-(keep + 1):-keep]      # exactly one spare
    to_delete = steps[:-(keep + 1)]

    def say(msg: str) -> None:
        if verbose:
            print(f"[prune] {msg}", flush=True)

    # Local copies go FIRST, before any S3 call. Archiving one ~81 GB DeepSpeed
    # state is an `aws s3 mv --recursive` that takes ~8 minutes, and SageMaker's
    # checkpoint sync keeps uploading during that window: anything still on disk
    # during that window can be put straight back into the prefix we just pruned.
    #
    # NOTE (measured on g1-puffs-rolling, 2026-09-12): ordering local-first is
    # necessary but NOT sufficient. That run still re-archived step_001000 every
    # ~21 minutes, and S3 kept showing the step back under the job prefix, so
    # something is still reinstating it. The three logged numbers below exist to
    # settle it on the next run: if `local before` does not contain the step, the
    # re-upload is not coming from this host's disk and the sync itself is
    # restoring it; if `local after` still contains it, the rmtree is failing
    # silently (drop_local passes ignore_errors=True).
    say(f"local before={local_steps(local_state)} "
        f"s3={steps} keep={newest} archive={to_archive} delete={to_delete}")
    drop_local([*to_archive, *to_delete], local_state, local_weights, say)
    say(f"local after={local_steps(local_state)}")

    for step in to_archive:
        name = f"step_{step:06d}"
        say(f"archiving {name} -> {archive_s3}")
        _run(["aws", "s3", "mv", f"{state_s3.rstrip('/')}/{name}/",
              f"{archive_s3.rstrip('/')}/state/{name}/", "--recursive", "--only-show-errors"])
        _run(["aws", "s3", "mv", f"{weights_s3.rstrip('/')}/{name}.pt",
              f"{archive_s3.rstrip('/')}/weights/{name}.pt", "--only-show-errors"])

    for step in to_delete:
        name = f"step_{step:06d}"
        say(f"deleting {name}")
        _run(["aws", "s3", "rm", f"{state_s3.rstrip('/')}/{name}/", "--recursive", "--only-show-errors"])
        _run(["aws", "s3", "rm", f"{weights_s3.rstrip('/')}/{name}.pt", "--only-show-errors"])

    if to_archive or to_delete:
        say(f"kept {newest}, archived {to_archive}, deleted {to_delete}")


def archive_uri(checkpoint_s3: str, rel: str) -> str:
    """``<parent>/_archive/<job>/<rel>`` — never a string prefix of the job's URI."""
    trimmed = checkpoint_s3.rstrip("/")
    parent, _, job = trimmed.rpartition("/")
    return f"{parent}/_archive/{job}/{rel}"


def _alive(pid: int) -> bool:
    """True while ``pid`` still exists (signal 0 only probes, never delivers)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _sleep_while_alive(seconds: int, parent_pid: int | None) -> bool:
    """Sleep up to ``seconds``, waking early if the training process exits.

    Returns False when the parent is gone and the caller should stop.
    """
    slept = 0
    step = 30
    while slept < seconds:
        if parent_pid is not None and not _alive(parent_pid):
            return False
        time.sleep(min(step, seconds - slept))
        slept += step
    return parent_pid is None or _alive(parent_pid)


def loop(output_dir: str, checkpoint_s3: str, keep: int, interval: int,
         manage_s3: bool = True, parent_pid: int | None = None) -> None:
    rel = ""
    try:
        rel = str(Path(output_dir).resolve().relative_to("/opt/ml/checkpoints"))
    except ValueError:
        return  # output_dir is not inside the mirrored dir; nothing to prune
    base = f"{checkpoint_s3.rstrip('/')}/{rel}/checkpoints"
    state_s3, weights_s3 = f"{base}/state", f"{base}/weights"
    archive_s3 = archive_uri(checkpoint_s3, rel)
    local = Path(output_dir) / "checkpoints"
    role = "main host: prunes S3" if manage_s3 else "worker: local cleanup only"
    print(f"[prune] watching {state_s3} (keep={keep}, every {interval}s, "
          f"archive -> {archive_s3}) [{role}]", flush=True)
    while True:
        if not _sleep_while_alive(interval, parent_pid):
            print("[prune] training process is gone; pruner exiting", flush=True)
            return
        try:
            if manage_s3:
                prune_once(state_s3, weights_s3, archive_s3, local / "state",
                           local / "weights", keep=keep)
            else:
                reconcile_local(state_s3, local / "state", local / "weights")
        except Exception as exc:  # never let pruning kill the job
            print(f"[prune] error (ignored): {exc}", flush=True)


def spawn(output_dir: str, checkpoint_s3: str, *, keep: int = 1,
          interval: int = 900, manage_s3: bool = True) -> None:
    """Fork a daemon that prunes periodically; returns immediately in the parent.

    entry.py calls this just before ``os.execvp`` on EVERY host. exec replaces
    the parent's process image but leaves children running, so the daemon
    outlives it. Pass ``manage_s3=False`` on non-main hosts: they must still
    clear their own local copies (otherwise the checkpoint sync re-uploads what
    the main host pruned) but must not race it writing S3.

    The daemon watches the pid it was forked from. exec keeps that pid, so it is
    the training process: when it goes, there is nothing left to prune and the
    daemon stops instead of holding the container open on a finished job.
    """
    if not checkpoint_s3:
        return
    trainer_pid = os.getpid()           # exec below keeps this pid
    try:
        if os.fork() != 0:
            return                      # parent: carry on and exec training
    except OSError as exc:
        print(f"[prune] fork failed, pruning disabled: {exc}", flush=True)
        return
    try:
        os.setsid()
    except OSError:
        pass
    try:
        loop(output_dir, checkpoint_s3, keep, interval, manage_s3=manage_s3,
             parent_pid=trainer_pid)
    except Exception as exc:
        print(f"[prune] daemon exiting: {exc}", flush=True)
    finally:
        os._exit(0)


if __name__ == "__main__":
    # Manual use: python prune_checkpoints.py <output_dir> <checkpoint_s3> [keep]
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    loop(a[0], a[1], int(a[2]) if len(a) > 2 else 1, 0)
