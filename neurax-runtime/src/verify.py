"""
Does this model build, and is it the model NEURAX costed?

Run against a candidate `model.py` before anything is trained. It answers three
questions, in the order that makes each cheap:

  1. Does the file import and the class instantiate?
  2. How many parameters does PyTorch actually build?
  3. Does a forward pass run on one batch of the declared input?

The second is the one that matters. NEURAX's whole claim is that it can tell
you what a design costs before you run it, and the only way that claim survives
contact with generated code is if the generated code is held to it. A model
that trains fine and has a different parameter count from the one analysed is
not a bug in training — it is a different model, and every figure the studio
showed about it was about something else.

The third catches what the second cannot: a model can have exactly the right
weights and still fail on the first batch, because a shape does not line up.
Finding that here costs a second. Finding it at step one of a queued run costs
whatever the queue cost.

Reads a request on stdin and writes the verdict to the file the request names,
so the caller never has to parse a traceback out of a log — or out of whatever
the candidate model decided to print. Model code that prints a banner in its
`__init__` is common and harmless, and it would make a verdict read off stdout
unparseable. Stdout stays the model's; the verdict gets its own file.
"""

import json
import sys
import traceback

#: Where the verdict goes. Taken from the request before anything else can
#: fail, so even an early failure has somewhere to be written.
_VERDICT_PATH = None


def answer(verdict: dict) -> None:
    """Write the verdict to its file, and to stdout as a fallback."""
    text = json.dumps(verdict)
    if _VERDICT_PATH:
        try:
            with open(_VERDICT_PATH, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass
    print(text)


def main() -> None:
    global _VERDICT_PATH

    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        answer({"ok": False, "stage": "request", "error": f"unreadable request: {exc}"})
        return

    _VERDICT_PATH = request.get("verdictPath")
    directory = request["directory"]
    sys.path.insert(0, directory)

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        answer({"ok": False, "stage": "torch", "error": str(exc)})
        return

    # ── Does it import? ─────────────────────────────────────────────────
    try:
        import model as model_module
    except Exception:  # noqa: BLE001
        answer({"ok": False, "stage": "import", "error": traceback.format_exc()})
        return

    wanted = request.get("modelClass")
    cls = getattr(model_module, wanted, None) if wanted else None
    if cls is None:
        # The class may be named something else. One `nn.Module` in the file is
        # unambiguous; several is not, and guessing there would verify the
        # wrong class.
        candidates = [
            v
            for v in vars(model_module).values()
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
        ]
        if len(candidates) != 1:
            answer(
                {
                    "ok": False,
                    "stage": "class",
                    "error": f"no class named {wanted!r}, and {len(candidates)} candidates to choose from",
                }
            )
            return
        cls = candidates[0]

    # ── Does it instantiate? ────────────────────────────────────────────
    try:
        instance = cls()
    except Exception:  # noqa: BLE001
        answer({"ok": False, "stage": "construct", "error": traceback.format_exc()})
        return

    parameters = sum(p.numel() for p in instance.parameters())

    # ── Does one batch go through it? ───────────────────────────────────
    forward_ok, forward_error, output_shape = False, None, None
    shape = [int(d) for d in (request.get("inputShape") or [])]
    if shape:
        kind = request.get("inputKind", "features")
        vocab = max(1, int(request.get("vocabSize") or 1))
        try:
            if kind == "tokens":
                x = torch.randint(0, vocab, (2, *shape))
            else:
                x = torch.randn(2, *shape)
            with torch.no_grad():
                out = instance(x)
            output_shape = list(out.shape)
            forward_ok = True
        except Exception:  # noqa: BLE001
            forward_error = traceback.format_exc()

    answer(
        {
            "ok": True,
            "stage": "verified",
            "parameters": parameters,
            "className": cls.__name__,
            "forwardOk": forward_ok,
            "forwardError": forward_error,
            "outputShape": output_shape,
            "torchVersion": torch.__version__,
        }
    )


if __name__ == "__main__":
    main()
