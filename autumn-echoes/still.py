"""Render test stills of one scene at chosen times:  python still.py forest 0 3 7.5 [--w 540 --h 960] [--out dir]"""
import argparse
import base64
import os
import time

from driver import Session

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene')
    ap.add_argument('times', nargs='+', type=float)
    ap.add_argument('--w', type=int, default=1080)
    ap.add_argument('--h', type=int, default=1920)
    ap.add_argument('--out', default=os.path.join(HERE, 'stills'))
    ap.add_argument('--extra', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    s = Session('index.html', '?scene=%s&w=%d&h=%d%s' % (a.scene, a.w, a.h, a.extra), width=a.w, height=a.h)
    print('init %.1f s' % (time.time() - t0))
    for line in s.logs:
        print('  |', line[:300])
    for t in sorted(a.times):
        t1 = time.time()
        r = s.page.evaluate('(t) => window.renderAt(t)', t)
        data = base64.b64decode(r['url'].split(',', 1)[1])
        path = os.path.join(a.out, '%s_%06.2f.png' % (a.scene, t))
        with open(path, 'wb') as f:
            f.write(data)
        print('t=%6.2f  render %5.0f ms  encode %4.0f ms  wall %.1f s  -> %s' % (t, r['render'], r['encode'], time.time() - t1, path))
    n = len(s.logs)
    for line in s.logs[n - 20:]:
        print('  |', line[:300])
    s.close()


if __name__ == '__main__':
    main()
