"""One frame per second in a labelled grid, for reviewing the whole film at a glance."""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


def sheet(frames_dir, out, every=30, cols=8, w=320, offset=15):
    idx = list(range(offset, 1200, every))
    rows = (len(idx) + cols - 1) // cols
    h = w * 9 // 16
    im = Image.new('RGB', (cols * w, rows * h), (0, 0, 0))
    font = ImageFont.truetype(os.path.join(HERE, 'fonts', 'Oswald.ttf'), 16)
    for n, fi in enumerate(idx):
        p = os.path.join(frames_dir, 'f%04d.png' % fi)
        if not os.path.exists(p):
            continue
        fr = Image.open(p).convert('RGB').resize((w, h), Image.LANCZOS)
        d = ImageDraw.Draw(fr)
        d.text((6, 3), '%.1f s' % (fi / 30), fill=(255, 220, 90), font=font)
        im.paste(fr, ((n % cols) * w, (n // cols) * h))
    im.save(out, quality=90)


if __name__ == '__main__':
    sheet(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
