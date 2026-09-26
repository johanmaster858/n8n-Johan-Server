"""Render a scene's frames:  python render.py SCENE FIRST LAST  (global frame numbers, t = i / 30).
Frames land in frames/SCENE/NNNNN.png; existing frames are skipped so a run can resume."""
import os
import sys
import time

from driver import Session

HERE = os.path.dirname(os.path.abspath(__file__))

# global frame ranges per scene (inclusive); overlaps are cross-dissolved in finish.py
RANGES = {
    'forest': (0, 440),
    'hand': (414, 522),
    'boots': (504, 600),
    'campfire': (579, 690),
    'aerial': (657, 1080),
}


def main():
    scene = sys.argv[1]
    a, b = RANGES[scene]
    if len(sys.argv) > 3:
        a, b = int(sys.argv[2]), int(sys.argv[3])
    out = os.path.join(HERE, 'frames', scene)
    os.makedirs(out, exist_ok=True)
    todo = [i for i in range(a, b + 1) if not os.path.exists(os.path.join(out, '%05d.png' % i))]
    if not todo:
        print(scene, 'complete')
        return
    s = Session('index.html', '?scene=%s' % scene)
    t0 = time.time()
    for n, i in enumerate(todo):
        path = os.path.join(out, '%05d.png' % i)
        s.frame(i, path + '.tmp')
        os.replace(path + '.tmp', path)
        if n % 10 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            print('%s %d (%d/%d) %.1f s/frame, eta %.0f min' % (scene, i, n + 1, len(todo), el / (n + 1), el / (n + 1) * (len(todo) - n - 1) / 60), flush=True)
    s.close()


if __name__ == '__main__':
    main()
