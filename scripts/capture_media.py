#!/usr/bin/env python3
"""Render the README's figures by driving the real viewer in a headless browser.

The images in `docs/media/` are not hand-made screenshots: this script opens
`viewer/index.html`, picks a dataset and an ablation, scrubs the optimisation and
captures the frames, so a figure can never quietly disagree with the code that
produced it.

    python scripts/capture_media.py --dataset mug --out-dir docs/media

Payloads for licence-gated datasets are git-ignored, so regenerate them first
(`scripts/export_viewer.py --source grab:mug_drink_1:best ...`) if you want the
real-grasp figures rather than the synthetic ones.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from toporetarget.paths import REPO_ROOT

VIEWER = REPO_ROOT / "viewer" / "index.html"
# Headless Chromium needs a software rasteriser for WebGL.
BROWSER_ARGS = ["--use-gl=angle", "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader", "--disable-lcd-text"]


def pick(page, selector, label):
    ok = page.evaluate(
        """([sel, want]) => {
             const b = [...document.querySelectorAll(sel)]
                 .find(x => x.textContent.trim() === want);
             if (!b) return false;
             b.click();
             return true;
           }""", [selector, label])
    if not ok:
        have = page.evaluate(
            "sel => [...document.querySelectorAll(sel)].map(x => x.textContent.trim())",
            selector)
        raise SystemExit(f"no {selector} entry {label!r}; available: {have}")
    page.wait_for_timeout(400)


def set_frame(page, i):
    page.evaluate("""i => { const s = document.getElementById('scrub');
                            s.value = i; s.dispatchEvent(new Event('input')); }""", i)
    page.wait_for_timeout(90)


def n_frames(page):
    return int(page.evaluate("() => +document.getElementById('scrub').max"))


def toggles(page, **state):
    for name, on in state.items():
        page.evaluate("""([id, on]) => { const c = document.getElementById(id);
                          if (c && c.checked !== on) c.click(); }""", [name, on])
    page.wait_for_timeout(250)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mug", help="label shown in the object switch")
    ap.add_argument("--out-dir", default="docs/media")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=810)
    ap.add_argument("--gif-width", type=int, default=900)
    ap.add_argument("--fps", type=int, default=11)
    ap.add_argument("--ablations", default="full,no_IM",
                    help="comma list of runs for the comparison figure; pick ones "
                         "that actually differ on the chosen object")
    ap.add_argument("--hold-frames", type=int, default=9,
                    help="repeats of the converged frame before the loop restarts")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="tr_media_"))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=BROWSER_ARGS)
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=2)
        page.goto(VIEWER.as_uri())
        page.wait_for_function("() => document.querySelectorAll('#dataSeg button').length > 0",
                               timeout=30000)
        page.wait_for_timeout(1200)
        pick(page, "#dataSeg button", args.dataset)
        toggles(page, tObj=True, tAnchor=True, tKp=False, tEdges=False)

        # --- hero: the full method converging -------------------------------- #
        pick(page, "#abl button", "full")
        total = n_frames(page)
        frames = tmp / "hero"; frames.mkdir()
        for i in range(total + 1):
            set_frame(page, i)
            page.screenshot(path=str(frames / f"{i:04d}.png"))
        # hold on the converged pose, so the loop does not snap back instantly
        last = frames / f"{total:04d}.png"
        for k in range(1, args.hold_frames + 1):
            shutil.copyfile(last, frames / f"{total + k:04d}.png")
        print(f"[hero] captured {total + 1} frames (+{args.hold_frames} held)")

        # --- stills: each ablation at convergence ----------------------------- #
        # The loss panel is hidden here: at figure scale it covered the hand and
        # added nothing the caption does not say.
        page.evaluate("() => { document.getElementById('loss').style.display='none'; }")
        stills, scores = {}, {}
        wanted = [a.strip() for a in args.ablations.split(",") if a.strip()]
        for name in wanted:
            # the viewer prints ablation names with spaces, not underscores
            pick(page, "#abl button", name.replace("_", " "))
            set_frame(page, n_frames(page))
            scores[name] = page.evaluate(
                """() => [document.getElementById('mE').textContent.trim(),
                          document.getElementById('mP').textContent.trim()]""")
            p = tmp / f"still_{name}.png"
            page.screenshot(path=str(p))
            stills[name] = p
        page.evaluate("() => { document.getElementById('loss').style.display=''; }")
        print("[stills] " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in scores.items()))

        # --- one shot with the interaction mesh shown -------------------------- #
        pick(page, "#abl button", "full")
        toggles(page, tKp=True, tEdges=True)
        set_frame(page, n_frames(page))
        page.screenshot(path=str(out / "interaction-mesh.png"))
        browser.close()

    gif = out / "hero.gif"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", str(tmp / "hero" / "%04d.png"),
         "-vf", (f"scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
                 "[a]palettegen=max_colors=200[p];[b][p]paletteuse=dither=sierra2_4a"),
         "-loop", "0", str(gif)], check=True)

    # Ablation figure: crop the ROBOT pane out of each still and lay the three
    # side by side.  Stacking whole screenshots produced a figure three screens
    # tall, which is unreadable inline in a README.
    from PIL import Image, ImageDraw
    captions = {"full": "full method", "no_IM": "without E_IM",
                "no_pen": "without E_pen", "no_bone": "without E_bone",
                "no_reg": "without E_reg"}
    order = [(k, captions.get(k, k)) for k in wanted]
    # Both metrics, because contact precision alone misleads: dropping E_pen
    # *improves* it while the hand sinks into the object.
    order = [(k, cap, "contact %s   ·   penetration %s" % tuple(scores.get(k, ("?", "?"))))
             for k, cap in order]
    crops = []
    for key, *_ in order:
        im = Image.open(stills[key])
        W, H = im.size
        crops.append(im.crop((W // 2, int(H * 0.17), W, int(H * 0.90))))
    cw = args.gif_width // len(crops)
    crops = [c.resize((cw, round(c.height * cw / c.width)), Image.LANCZOS) for c in crops]
    from PIL import ImageFont
    def face(size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
        except OSError:
            return ImageFont.load_default()

    title, small, bar = face(17, bold=True), face(14), 56
    strip = Image.new("RGB", (cw * len(crops), crops[0].height + bar), "#0e1116")
    draw = ImageDraw.Draw(strip)
    for i, (c, (_, caption, metrics)) in enumerate(zip(crops, order)):
        strip.paste(c, (i * cw, bar))
        draw.text((i * cw + 14, 8), caption, fill="#e8ecf4", font=title)
        draw.text((i * cw + 14, 31), metrics, fill="#8f9bb0", font=small)
        if i:
            draw.line([(i * cw, 0), (i * cw, strip.height)], fill="#242a35", width=2)
    strip.save(out / "ablation.png", optimize=True)

    shutil.rmtree(tmp, ignore_errors=True)
    for f in sorted(out.iterdir()):
        print(f"  {f.stat().st_size / 1024:8.0f} KB  {f}")


if __name__ == "__main__":
    main()
