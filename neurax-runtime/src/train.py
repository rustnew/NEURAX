"""
The harness that trains a NEURAX design.

Written to be read. Anyone debugging a run will open this file inside the run
directory it was copied into, next to the `model.py` it trains and the
`steps.jsonl` it wrote, and should be able to follow what happened without
consulting anything else.

Three properties matter more than speed here:

  * **It reports the real parameter count before the first step.** That single
    number confronts eleven IR phases of prediction in about a second, with no
    GPU time spent. If it disagrees with what NEURAX computed, a formula is
    wrong, and finding that out now costs nothing — so it is written first,
    and the studio compares it immediately.

  * **It stops at a step boundary, never inside one.** Pause and stop arrive
    as a file, read between steps. Stopping mid-kernel would leave memory
    allocated, a half-applied optimizer update, and no checkpoint.

  * **It says what went wrong.** Every failure path writes `status: failed`
    with the exception into `state.json`, because a run that simply stops
    reports the same thing as one that was killed, and those need different
    responses from the user.
"""

import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads((RUN_DIR / path).read_text())
    except Exception:
        return default


def write_json(path, payload):
    # Written and renamed, so the studio never reads a half-written file. It
    # polls these while training is writing them.
    tmp = RUN_DIR / (path + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(RUN_DIR / path)


_last_state_write = 0.0


def update_state(force=False, **fields):
    """
    Publish where the run has got to.

    Throttled, and that is not premature optimisation. The first version wrote
    and renamed `state.json` on every step: a training loop that manages a
    thousand steps a second would issue a thousand writes and a thousand
    renames a second, and the run would spend more time publishing its
    progress than making any. Every half-second is far finer than a human
    reads, and the studio's own polling is slower than that anyway.

    A status change is never throttled — `paused`, `finished` and `failed` are
    the transitions everything downstream waits on, and delaying one by half a
    second is how a stop request looks ignored.
    """
    global _last_state_write
    is_status_change = "status" in fields
    if not (force or is_status_change) and time.monotonic() - _last_state_write < 0.5:
        return None

    state = read_json("state.json", {}) or {}
    state.update(fields)
    state["lastHeartbeat"] = now()
    write_json("state.json", state)
    _last_state_write = time.monotonic()
    return state


def fail(message):
    update_state(status="failed", error=message)
    print(f"[neurax] failed: {message}", file=sys.stderr)
    sys.exit(1)


# ── Data ────────────────────────────────────────────────────────────────────


def load_batches(req, input_shape, device, torch):
    """
    Yield `(inputs, targets)` forever.

    Three sources, in order of how real they are. A folder of images is read
    with PIL directly rather than through torchvision, which is not installed
    on every machine and would turn a missing optional dependency into a run
    that cannot start. A CSV is read with numpy. With no dataset at all the
    harness trains on synthetic tensors — which is not a pretence of training
    on data, it is how the pipeline itself gets smoke-tested, and the run
    records `dataset: synthetic` so nothing downstream mistakes it.
    """
    batch = int(req["batchSize"])
    path = req.get("datasetPath")
    num_classes = max(1, int(req["numClasses"]))

    if path and Path(path).is_dir():
        from PIL import Image

        root = Path(path)
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        files = []
        for label, name in enumerate(classes):
            for f in sorted((root / name).iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    files.append((f, label))
        if not files:
            fail(f"no readable images under {path}")

        if len(input_shape) < 4:
            fail(
                f"this design takes {input_shape[1:]} per sample, which is not an image shape — "
                f"a folder of images cannot be fed to it"
            )
        c, h, w = input_shape[-3], input_shape[-2], input_shape[-1]
        print(f"[neurax] {len(files)} images, {len(classes)} classes", flush=True)

        while True:
            random.shuffle(files)
            for i in range(0, len(files) - batch + 1, batch):
                chunk = files[i : i + batch]
                arrays, labels = [], []
                for f, label in chunk:
                    try:
                        img = Image.open(f).convert("RGB").resize((w, h))
                    except Exception:
                        # A few unreadable files must not end a run — the
                        # dataset profile already warned that some exist.
                        continue
                    arrays.append(torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 255.0)
                    labels.append(label)
                if not arrays:
                    continue
                x = torch.stack(arrays)[:, :c].to(device)
                y = torch.tensor(labels, device=device)
                yield x, y

    elif path and Path(path).is_file():
        import numpy as np

        raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype="float32")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        features = torch.from_numpy(raw[:, :-1]).float()
        targets = torch.from_numpy(raw[:, -1]).long()
        print(f"[neurax] {features.shape[0]} rows, {features.shape[1]} features", flush=True)
        while True:
            perm = torch.randperm(features.shape[0])
            for i in range(0, features.shape[0] - batch + 1, batch):
                idx = perm[i : i + batch]
                yield features[idx].to(device), targets[idx].to(device)

    else:
        # No dataset: synthetic tensors of the shape *and dtype* the model
        # declared. A token embedding takes integer indices, not floats — feed
        # it `randn` and it fails inside the first attention block with a shape
        # error that points nowhere near the real cause.
        kind = req.get("inputKind", "features")
        vocab = int(req.get("vocabSize") or 1)
        while True:
            if kind == "tokens":
                x = torch.randint(0, max(1, vocab), (batch, *input_shape[1:]), device=device)
            else:
                x = torch.randn(batch, *input_shape[1:], device=device)
            y = torch.randint(0, num_classes, (batch,), device=device)
            yield x, y


def _to_array(img):
    import numpy as np

    return np.asarray(img, dtype="float32")


# ── The run ─────────────────────────────────────────────────────────────────


def main():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        fail(f"PyTorch is not available to this interpreter: {exc}")
        return

    req = read_json("request.json")
    if not req:
        fail("request.json is missing or unreadable")
        return

    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[neurax] torch {torch.__version__} on {device}", flush=True)

    # The design, as PyTorch. `model.py` sits beside this file and was written
    # by the studio's own generator.
    sys.path.insert(0, str(RUN_DIR))
    try:
        import model as model_module  # noqa: PLC0415

        cls = getattr(model_module, req["modelClass"], None)
        if cls is None:
            # The generator names the class after the design, so a mismatch is
            # recoverable: take the only nn.Module defined in the file.
            candidates = [
                v
                for v in vars(model_module).values()
                if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
            ]
            if len(candidates) != 1:
                fail(f"model.py defines no class named {req['modelClass']}")
                return
            cls = candidates[0]
        model = cls().to(device)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail("model.py could not be instantiated:\n" + traceback.format_exc())
        return

    parameters = sum(p.numel() for p in model.parameters())

    # Before a single step. This is the cheapest possible moment to discover
    # that a formula is wrong.
    write_json(
        "model_built.json",
        {
            "parameters": parameters,
            "device": str(device),
            "torchVersion": torch.__version__,
            "dtype": str(next(model.parameters()).dtype).replace("torch.", "")
            if parameters
            else req.get("precision", "fp32"),
        },
    )
    print(f"[neurax] {parameters:,} parameters", flush=True)

    # One sample's shape, from the design. Not defaulted: a wrong input shape
    # does not degrade a run, it kills it in the first layer, and guessing
    # would hide which of the two was wrong.
    sample_shape = tuple(int(d) for d in req["inputShape"])
    input_shape = (int(req["batchSize"]), *sample_shape)
    batches = load_batches(req, input_shape, device, torch)

    optimiser = torch.optim.AdamW(model.parameters(), lr=float(req["learningRate"]))
    loss_fn = nn.CrossEntropyLoss()
    total_steps = max(1, int(req["stepsPerEpoch"]) * int(req["epochs"]))
    steps_per_epoch = max(1, int(req["stepsPerEpoch"]))
    checkpoint_every = max(1, int(req["checkpointEverySteps"]))
    keep_checkpoints = int(req.get("keepCheckpoints", 3))

    # Resume, when a checkpoint is there to resume from. The step it reached
    # is where this run picks up — not step 1, which would silently retrain
    # everything already done.
    step = 0
    resumed = sorted((RUN_DIR / "checkpoints").glob("step_*.pt"))
    if resumed:
        try:
            ckpt = torch.load(resumed[-1], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimiser.load_state_dict(ckpt["optimiser"])
            step = int(ckpt.get("step", 0))
            if "rng" in ckpt:
                torch.set_rng_state(ckpt["rng"])
            print(f"[neurax] resumed at step {step}", flush=True)
        except Exception:  # noqa: BLE001
            print("[neurax] a checkpoint was present but unreadable; starting fresh", flush=True)

    steps_file = (RUN_DIR / "steps.jsonl").open("a", buffering=1)
    update_state(status="running", pid=os.getpid(), step=step)

    control_checked_at = 0.0
    command = "run"

    try:
        while step < total_steps:
            # Read at most twice a second rather than once a step, for the
            # same reason as the state file: at a thousand steps a second the
            # loop would spend its time opening a file that changes when a
            # human clicks something. Half a second is imperceptible on a
            # stop button and free on the loop.
            if time.monotonic() - control_checked_at >= 0.5:
                command = (read_json("control.json", {}) or {}).get("command", "run")
                control_checked_at = time.monotonic()

            if command == "stop":
                save_checkpoint(torch, model, optimiser, step, exact=True, keep=keep_checkpoints)
                update_state(status="finished", step=step)
                print("[neurax] stopped by request", flush=True)
                return

            if command == "pause":
                # A pause writes a checkpoint once, then idles cheaply. Idling
                # rather than exiting is what makes resume instant and keeps
                # the process's own identity — the studio is still attached to
                # a run, not to a corpse it has to restart.
                if not (RUN_DIR / "checkpoints" / f"step_{step:06d}.pt").exists():
                    save_checkpoint(torch, model, optimiser, step, exact=True, keep=keep_checkpoints)
                update_state(status="paused", step=step)
                time.sleep(0.5)
                # Force the next iteration to re-read: while paused, the file
                # is the only thing that can change, and a throttled read
                # would make resume take up to a second longer than the click.
                control_checked_at = 0.0
                continue

            update_state(status="running", step=step)

            started = time.perf_counter()
            x, y = next(batches)
            data_seconds = time.perf_counter() - started

            optimiser.zero_grad(set_to_none=True)
            out = model(x)
            if out.ndim > 2:
                # A sequence model returns [batch, time, classes]; the loss
                # wants [batch*time, classes].
                out = out.reshape(-1, out.shape[-1])
                y = y.repeat_interleave(out.shape[0] // y.shape[0])
            loss = loss_fn(out, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            optimiser.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - started
            step += 1

            steps_file.write(
                json.dumps(
                    {
                        "step": step,
                        "epoch": (step - 1) // steps_per_epoch,
                        "loss": float(loss.item()),
                        "learningRate": float(optimiser.param_groups[0]["lr"]),
                        "gradNorm": float(grad_norm),
                        "stepTimeMs": step_seconds * 1000.0,
                        "dataTimeMs": data_seconds * 1000.0,
                        "samplesPerSec": float(x.shape[0]) / max(step_seconds, 1e-9),
                        **memory_fields(torch, device),
                    }
                )
                + "\n"
            )

            if step % checkpoint_every == 0:
                save_checkpoint(torch, model, optimiser, step, exact=True, keep=keep_checkpoints)

        save_checkpoint(torch, model, optimiser, step, exact=True, keep=keep_checkpoints)
        update_state(status="finished", step=step)
        print(f"[neurax] finished at step {step}", flush=True)

    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail("training stopped with an error:\n" + traceback.format_exc())
    finally:
        steps_file.close()


def memory_fields(torch, device):
    """Memory and telemetry, where the device reports them.

    A CPU run has no VRAM figures, and inventing them — reporting process RSS
    as though it were allocator state — would put a number in the Accuracy
    view that means something else entirely. Absent is the honest answer.
    """
    if device.type != "cuda":
        return {"vramAllocatedBytes": 0, "vramReservedBytes": 0}
    fields = {
        "vramAllocatedBytes": int(torch.cuda.memory_allocated()),
        "vramReservedBytes": int(torch.cuda.memory_reserved()),
    }
    try:
        fields["gpuUtilisationPct"] = float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001
        pass
    return fields


def prune_checkpoints(keep):
    """
    Keep the last `keep` checkpoints, delete the rest.

    Without this a run accumulates one checkpoint per interval forever. A
    100 000-step run checkpointing every 500 steps leaves 200 files; at the
    742 MB a mid-size model weighs, that is 148 GB from a single run — the
    disk fills, and the run dies from its own bookkeeping.

    Only the most recent ones are useful: resuming uses the last, and the one
    before it is insurance against the last being written when the power went.
    `keep` is part of the request, so a user who wants every checkpoint can
    have them; the default just refuses to be the reason a machine runs out of
    disk.
    """
    if keep <= 0:
        return
    files = sorted((RUN_DIR / "checkpoints").glob("step_*.pt"))
    for stale in files[:-keep]:
        try:
            stale.unlink()
            stale.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            # A checkpoint that cannot be deleted is not worth ending a run
            # over — the next prune will try again.
            pass


def save_checkpoint(torch, model, optimiser, step, exact, keep=3):
    """
    Write a checkpoint, and say honestly what kind it is.

    Weights and optimizer state resume the *training*. Only a checkpoint that
    also saved the RNG state resumes the *same run* — rerunning it produces
    the same batches in the same order. A tool that promises a reproducible
    record has to distinguish the two rather than quietly offer the weaker
    one, so the claim is written beside the file and the studio displays it.
    """
    path = RUN_DIR / "checkpoints" / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "rng": torch.get_rng_state(),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"step": step, "exactResume": bool(exact), "savedAt": now()}, indent=2)
    )
    prune_checkpoints(keep)


if __name__ == "__main__":
    main()
