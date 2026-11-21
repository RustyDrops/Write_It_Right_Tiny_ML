#!/usr/bin/env python3
"""Code-drawn promo card for Write It Right, 1200x630. No AI imagery.
v2 card (2026-11-20): v8 hybrid state. Claims from release_v8 README /
hybrid-v8/results_v8_h1.json: word-mode 95.24% (v2 image model, full
official EMNIST test), hybrid 91.52% on the v8 exam (1,768 samples,
17 UJI writers), deployed size 296,637 B (<=0.5 MB)."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "write-it-right-promo-v2.png")
W, H = 1200, 630

NOTO = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
NOTOR = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
if not os.path.exists(NOTO):
    NOTO = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    NOTOR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def F(path, size):
    return ImageFont.truetype(path, size)

NAVY_TOP = (9, 32, 48)
NAVY_BOT = (13, 58, 74)
WHITE = (255, 255, 255)
SOFT = (203, 228, 238)
MUTED = (148, 183, 199)
AMBER = (245, 176, 65)
TEAL_CHIP = (20, 74, 96)
GREEN = (34, 150, 84)
SCREEN = (244, 248, 249)
INK = (30, 44, 52)
GRAYTXT = (84, 102, 112)

img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):
    t = y / (H - 1)
    c = tuple(int(NAVY_TOP[i] + (NAVY_BOT[i] - NAVY_TOP[i]) * t) for i in range(3))
    for x in range(W):
        px[x, y] = c
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([-140, -180, 420, 380], fill=(245, 176, 65, 34))
glow = glow.filter(ImageFilter.GaussianBlur(90))
img = Image.alpha_composite(img.convert("RGBA"), glow)
d = ImageDraw.Draw(img, "RGBA")

def fit_font(text, maxw, start, path=NOTO):
    size = start
    while size > 36 and d.textlength(text, font=F(path, size)) > maxw:
        size -= 4
    return F(path, size)

# ---- left column
LX = 64
# badge: amber rounded square with a navy "b"
d.rounded_rectangle([66, 58, 158, 166], radius=22, fill=AMBER)
fb = F(NOTO, 72)
tw = d.textlength("b", font=fb)
d.text((112 - tw / 2, 70), "b", font=fb, fill=NAVY_TOP)

title_font = fit_font("Write It Right", 600, 84)
d.text((178, 74), "Write It Right", font=title_font, fill=WHITE)

d.text((LX, 196), "Write a letter.", font=F(NOTOR, 37), fill=SOFT)
d.text((LX, 242), "Hear it read back.", font=F(NOTO, 37), fill=WHITE)

d.text((LX, 292), "V8 HYBRID \u2014 STROKE + IMAGE", font=F(NOTO, 20), fill=AMBER)

chips = [
    ("95.2%", " word-mode EMNIST accuracy"),
    ("91.5%", " on real pen strokes (v8 exam)"),
    ("<0.5 MB", " \u00b7 deployed \u00b7 fully offline"),
]
cy = 336
f_b = F(NOTO, 27)
f_r = F(NOTOR, 27)
for bold, rest in chips:
    wb = d.textlength(bold, font=f_b)
    wr = d.textlength(rest, font=f_r)
    cw = int(wb + wr) + 68
    d.rounded_rectangle([LX, cy, LX + cw, cy + 58], radius=29, fill=TEAL_CHIP)
    d.ellipse([LX + 18, cy + 24, LX + 28, cy + 34], fill=AMBER)
    d.text((LX + 40, cy + 13), bold, font=f_b, fill=AMBER)
    d.text((LX + 40 + wb, cy + 13), rest, font=f_r, fill=WHITE)
    cy += 74

d.text((LX, 566), "Free \u00b7 Open source \u00b7 github.com/RustyDrops/Write_It_Right_Tiny_ML",
       font=F(NOTOR, 20), fill=MUTED)

# ---- paper panel (right)
PX0, PY0, PX1, PY1 = 806, 42, 1146, 588
d.rounded_rectangle([PX0 - 8, PY0 - 8, PX1 + 8, PY1 + 8], radius=46, fill=(5, 18, 26))
d.rounded_rectangle([PX0, PY0, PX1, PY1], radius=38, fill=SCREEN)
d.rounded_rectangle([PX0 + 118, PY0 + 12, PX1 - 118, PY0 + 30], radius=9, fill=(5, 18, 26))
d.text((PX0 + 26, PY0 + 46), "Letter check", font=F(NOTO, 21), fill=INK)
d.text((PX0 + 26, PY0 + 74), "drawn by hand, read by the model", font=F(NOTOR, 16), fill=GRAYTXT)
d.line([PX0 + 26, PY0 + 104, PX1 - 26, PY0 + 104], fill=(214, 224, 228), width=2)

def letter_cell(x, y, w, h, ch, angle, picked=False):
    cell = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cell)
    cd.rounded_rectangle([2, 2, w - 3, h - 3], radius=14, fill=(255, 255, 255, 255),
                         outline=GREEN if picked else (205, 216, 221),
                         width=5 if picked else 2)
    lf = F(NOTO, 84)
    lw_ = cd.textlength(ch, font=lf)
    cd.text(((w - lw_) / 2, h / 2 - 58), ch, font=lf, fill=INK)
    if angle:
        cell = cell.rotate(angle, resample=Image.BICUBIC, center=(w / 2, h / 2))
    img.alpha_composite(cell, (x, y))

GX, GY = PX0 + 24, PY0 + 122
CW, CH, GAP = 140, 148, 12
letter_cell(GX, GY, CW, CH, "b", -4, picked=True)
letter_cell(GX + CW + GAP, GY, CW, CH, "d", 3)
letter_cell(GX, GY + CH + GAP, CW, CH, "p", 2)
letter_cell(GX + CW + GAP, GY + CH + GAP, CW, CH, "q", -3)

# verdict badge under the picked cell row
f_badge = F(NOTO, 20)
badge_text = "READ: b"
bw = int(d.textlength(badge_text, font=f_badge)) + 52
bx0, by0 = GX, GY + 2 * CH + GAP + 18
d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + 38], radius=19, fill=GREEN)
icy = by0 + 19
d.line([(bx0 + 15, icy), (bx0 + 22, icy + 7)], fill=WHITE, width=4)
d.line([(bx0 + 22, icy + 7), (bx0 + 35, icy - 8)], fill=WHITE, width=4)
d.text((bx0 + 44, by0 + 8), badge_text, font=f_badge, fill=WHITE)

cap = "Checked on this device \u2014 nothing uploaded"
f_cap = F(NOTOR, 13)
d.text((PX0 + (PX1 - PX0 - d.textlength(cap, font=f_cap)) / 2, PY1 - 44),
       cap, font=f_cap, fill=GRAYTXT)

img.convert("RGB").save(OUT, "PNG")
print("saved", OUT, os.path.getsize(OUT))
