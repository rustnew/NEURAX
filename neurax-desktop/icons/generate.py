"""
Render the NEURAX application icons from the brand mark.

The source of truth is `neurax-ui/public/neurax-mark.png` — the same file the
studio serves — so the desktop icon cannot drift from the product's.

## No background

The icon is the mark on transparency, with nothing behind it. That is a
deliberate choice with a cost worth stating: the mark exists in two variants —
the artwork as drawn, which is charcoal with green highlights and reads on a
light taskbar, and its lightness-inverted twin, which reads on a dark dock.
An icon is one file, so one of them has to be chosen.

The inverted variant wins, because dark docks and taskbars are the common case
across macOS, Windows and the GNOME shells NEURAX is developed on. The
alternative, a filled tile, is what most applications ship and would read on
every ground — but it puts a solid square on the user's desktop, which is a
lockup rather than a logo.

## Bigger

The mark fills 92% of the canvas rather than sitting inside generous padding.
An icon competes at 32 pixels in a taskbar; whitespace it does not need is
whitespace that makes it smaller than everything beside it.

    python3 neurax-desktop/icons/generate.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "neurax-ui" / "public" / "neurax-mark.png"
OUT = ROOT / "neurax-desktop" / "icons"

# How much of the canvas the artwork occupies.
FILL = 0.92

# Everything Tauri references, plus the Windows Store set.
SIZES = {
    "32x32.png": 32,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "master-1024.png": 1024,
    "StoreLogo.png": 50,
    "Square30x30Logo.png": 30,
    "Square44x44Logo.png": 44,
    "Square71x71Logo.png": 71,
    "Square89x89Logo.png": 89,
    "Square107x107Logo.png": 107,
    "Square142x142Logo.png": 142,
    "Square150x150Logo.png": 150,
    "Square284x284Logo.png": 284,
    "Square310x310Logo.png": 310,
}


def render(size: int) -> Image.Image:
    """The mark, centred on a transparent square of `size`."""
    mark = Image.open(SOURCE).convert("RGBA")
    inner = max(1, round(size * FILL))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(
        mark.resize((inner, inner), Image.LANCZOS),
        ((size - inner) // 2, (size - inner) // 2),
    )
    return canvas


def main() -> None:
    for name, size in SIZES.items():
        render(size).save(OUT / name)
        print(f"  {name} ({size}px)")

    # Windows wants every size in one container; macOS wants its own format.
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    render(256).save(OUT / "icon.ico", sizes=[(s, s) for s in ico_sizes])
    print(f"  icon.ico ({', '.join(str(s) for s in ico_sizes)})")

    try:
        render(1024).save(OUT / "icon.icns")
        print("  icon.icns (1024px)")
    except (OSError, ValueError) as exc:
        # Not fatal on a Linux build, which never reads it — but the file must
        # not be silently left describing the previous logo.
        print(f"  icon.icns SKIPPED — {exc}")


if __name__ == "__main__":
    main()
