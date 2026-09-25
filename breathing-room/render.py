"""Render "Breathing Room" frame by frame (exact 1/30 s steps) and encode it.

    python render.py assets            # build the stained glass + textures
    python render.py frames A B        # render frames [A, B) as PNG
    python render.py still T out.png   # render a single moment (debugging)
    python render.py encode            # PNG sequence (+ soundtrack) -> MP4
"""
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scene as S          # noqa: E402
import timeline as TL      # noqa: E402
import post                # noqa: E402

W = int(os.environ.get('BR_W', 1920))
H = int(os.environ.get('BR_H', 1080))
TS = 16                    # screen tile size for cube binning
BUILD = os.environ.get('BR_BUILD', os.path.join(HERE, 'build'))
FRAMES = os.environ.get('BR_FRAMES', os.path.join(BUILD, 'frames'))


class Renderer:
    def __init__(self, w=W, h=H):
        import assets
        self.w, self.h = w, h
        npz = os.path.join(BUILD, 'window_design.npz')
        if not os.path.exists(npz):
            import stained_glass
            stained_glass.save_assets(BUILD)
        A = assets.build_all(npz)
        self.G = A['G']
        self.F = A['F']
        self.VT = A['VT']
        self.NT = A['NT']
        self.BN = A['BN']
        self.N3 = A['N3']
        self.WM, self.WS, self.WM1, self.WS1 = A['WM'], A['WS'], A['WM1'], A['WS1']
        # 40 px/m versions for light-source (VPL) estimation
        self.F40 = cv2.resize(np.ascontiguousarray(self.F[..., :4]), (400, 600), interpolation=cv2.INTER_AREA)
        self.F8 = cv2.resize(self.F40, (80, 120), interpolation=cv2.INTER_AREA)
        u8 = -5.0 + (np.arange(80) + 0.5) / 8.0
        v8 = 15.0 - (np.arange(120) + 0.5) / 8.0
        U8, V8 = np.meshgrid(u8, v8)
        self.g8_col = np.clip(np.floor((U8 + 5.0) / S.B).astype(int), 0, S.NCOLS - 1)
        self.g8_row = np.clip(np.floor(V8 / S.B).astype(int), 0, S.NROWS - 1)
        g2 = self.G[2]
        self.G40 = g2
        self.ty = (h + TS - 1) // TS
        self.tx = (w + TS - 1) // TS
        self.DD = np.zeros((570, 360), np.float32)
        self.DA = np.zeros((570, 360), np.float32)
        self.DI = np.zeros((570, 360), np.int32)
        # grid for evaluating cover on the 40 px/m window textures (v in [0, 15])
        u = -5.0 + (np.arange(400) + 0.5) / 40.0
        v = 15.0 - (np.arange(600) + 0.5) / 40.0
        U, V = np.meshgrid(u, v)
        self.g_col = np.clip(np.floor((U + 5.0) / S.B).astype(int), 0, S.NCOLS - 1)
        self.g_row = np.clip(np.floor(V / S.B).astype(int), 0, S.NROWS - 1)

    # ---------------------------------------------------------------- lights --
    def lights(self, P, cellcov):
        rv = P[S.P_REVEAL]
        tint = np.array([P[S.P_SUNR], P[S.P_SUNG], P[S.P_SUNB]], np.float32)
        open_ = 1.0 - cellcov[self.g_row, self.g_col]
        Tf = self.F40[..., :3] * rv + self.F40[..., 3:4] * tint * (1 - rv)
        Tf = Tf * open_[..., None]
        vpl = []
        nu, nv = 4, 8
        for j in range(nv):
            for i in range(nu):
                blk = Tf[j * 600 // nv:(j + 1) * 600 // nv, i * 400 // nu:(i + 1) * 400 // nu]
                m = blk.reshape(-1, 3).mean(0)
                uc = -5.0 + (i + 0.5) * 10.0 / nu
                vc = 15.0 - (j + 0.5) * 15.0 / nv
                area = (10.0 / nu) * (15.0 / nv)
                flux = 0.44 * P[S.P_LSUN] * S.S2 * m * area / math.pi
                vpl.append([uc, 0.02, vc + S.HS - S.D, *flux])
        Tg = self.G40[..., :3] * rv + self.G40[..., 3:4] * tint * (1 - rv)
        Tg = Tg * open_[..., None]
        wvpl = []
        wu, wv = 3, 5
        for j in range(wv):
            for i in range(wu):
                blk = Tg[j * 600 // wv:(j + 1) * 600 // wv, i * 400 // wu:(i + 1) * 400 // wu]
                m = blk.reshape(-1, 3).mean(0)
                uc = -5.0 + (i + 0.5) * 10.0 / wu
                vc = 15.0 - (j + 0.5) * 15.0 / wv
                area = (10.0 / wu) * (15.0 / wv)
                flux = P[S.P_LGLASS] * P[S.P_WSKY] * m * area / math.pi
                wvpl.append([uc, S.HS + vc, -S.D + 0.05, *flux])
        # spill: floor irradiance blurred over metres (air + walls scatter it back)
        open8 = 1.0 - cellcov[self.g8_row, self.g8_col]
        T8 = self.F8[..., :3] * rv + self.F8[..., 3:4] * tint * (1 - rv)
        E8 = T8 * open8[..., None] * (P[S.P_LSUN] * S.S2)
        SP = np.zeros((280, 192, 3), np.float32)
        SP[80:200, 56:136] = E8
        SP = cv2.GaussianBlur(SP, (0, 0), 13.0)
        tot = float(np.array(vpl)[:, 3:6].sum()) / 3.0
        P[S.P_AMB] = 0.0022 + 0.00035 * tot
        P[S.P_FILL] = 0.0042 * tot
        return np.array(vpl, np.float64), np.array(wvpl, np.float64), SP

    # ------------------------------------------------------------- binning --
    def bin_blocks(self, C, BLK, visible):
        tile_n = np.zeros((self.ty, self.tx), np.int32)
        tile_i = np.zeros((self.ty, self.tx, max(1, len(visible))), np.int32)
        pos = C[0:3]
        r, u, f = C[3:6], C[6:9], C[9:12]
        for k in visible:
            c = BLK[k, 0:3]
            R = BLK[k, 3:12].reshape(3, 3)
            h = BLK[k, 12] * 1.02
            corners = []
            for a in (-1, 1):
                for b in (-1, 1):
                    for cc in (-1, 1):
                        corners.append(c + R @ np.array([a * h, b * h, cc * h]))
            corners = np.array(corners) - pos
            zc = corners @ f
            if (zc < 0.05).any():
                x0, x1, y0, y1 = 0, self.tx - 1, 0, self.ty - 1
            else:
                sx = (corners @ r / zc / C[12] + 1) * 0.5 * self.w
                sy = (1 - corners @ u / zc / C[13]) * 0.5 * self.h
                if sx.max() < -4 or sx.min() > self.w + 4 or sy.max() < -4 or sy.min() > self.h + 4:
                    continue
                x0 = int(max(0, (sx.min() - 3) // TS))
                x1 = int(min(self.tx - 1, (sx.max() + 3) // TS))
                y0 = int(max(0, (sy.min() - 3) // TS))
                y1 = int(min(self.ty - 1, (sy.max() + 3) // TS))
            for ty in range(y0, y1 + 1):
                for tx in range(x0, x1 + 1):
                    tile_i[ty, tx, tile_n[ty, tx]] = k
                    tile_n[ty, tx] += 1
        return tile_n, tile_i

    # ------------------------------------------------------------ one image --
    def render_hdr(self, t, subframes=1, shutter=1.0 / 60.0):
        w, h = self.w, self.h
        acc = np.zeros((h, w, 3), np.float32)
        depth_c = None
        for si in range(subframes):
            ts = t if subframes == 1 else t + shutter * ((si + 0.5) / subframes - 0.5)
            C = TL.camera(ts)
            P = TL.params(ts)
            P[S.P_PIXANG] = 2.0 * C[13] / h
            BLK, visible, moving, cellcov, leak = TL.blocks(ts)
            CV = TL.cover_texture(cellcov, leak)
            S.build_dyn_map(BLK, moving, self.DD, self.DA, self.DI)
            VPL, WVPL, SP = self.lights(P, cellcov)
            tile_n, tile_i = self.bin_blocks(C, BLK, visible)
            out = np.zeros((h, w, 3), np.float32)
            depth = np.zeros((h, w), np.float32)
            objid = np.zeros((h, w), np.int32)
            G0, G1, G2, G3 = self.G
            args = (C, P, w, h, BLK, tile_n, tile_i, TS, G0, G1, G2, G3, self.F, CV, self.DD, self.DA, self.DI,
                    VPL, WVPL, SP, self.NT, self.WM, self.WS, self.WM1, self.WS1, self.BN)
            S.render_primary(*args, out, depth, objid)
            S.refine_edges(*args, out, depth, objid)
            acc += out
            if si == subframes // 2 or depth_c is None:
                depth_c = depth
                state_c = (C, P, BLK, moving, CV, ts)
        hdr = acc / subframes
        C, P, BLK, moving, CV, ts = state_c
        S.build_dyn_map(BLK, moving, self.DD, self.DA, self.DI)
        # volumetric light columns at half resolution
        hw, hh = w // 2, h // 2
        vol = np.zeros((hh, hw, 3), np.float32)
        vdepth = np.zeros((hh, hw), np.float32)
        S.render_volume(C, P, hw, hh, self.VT, CV, self.DD, self.DA, self.DI, depth_c, 2.0, self.N3, vol, vdepth)
        S.upsample_add(vol, vdepth, depth_c, 2.0, hdr)
        # dust sparkling in the beams
        dp, dtw, dsz = TL.dust(ts)
        dcol = np.zeros((dp.shape[0], 3), np.float64)
        S.light_points(P, dp, self.F, CV, self.DD, self.DA, self.DI, dcol)
        rel = dp - C[0:3]
        dist = np.sqrt((rel ** 2).sum(1))
        cosang = (rel @ np.array([S.LX, S.LY, S.LZ])) / np.maximum(dist, 1e-6)
        g = 0.55
        ph = (1 - g * g) / (1 + g * g - 2 * g * cosang) ** 1.5
        k = 0.05 * ph * dtw / np.maximum(dist / 8.0, 1.0) ** 0.8
        dcol *= k[:, None]
        S.splat_points(C, w, h, dp, dcol, dsz, depth_c, hdr)
        # light particles from dissolving cubes
        pp, pc, pr = TL.particles(ts)
        if len(pp):
            lcol = np.zeros((pp.shape[0], 3), np.float64)
            S.light_points(P, pp, self.F, CV, self.DD, self.DA, self.DI, lcol)
            dist = np.sqrt(((pp - C[0:3]) ** 2).sum(1))
            glow = pc * (0.9 + 0.02 * lcol.sum(1, keepdims=True)) + 0.01 * lcol
            glow *= (3.2 / np.maximum(dist, 2.0) ** 0.5)[:, None]
            S.splat_points(C, w, h, pp, glow, pr, depth_c, hdr)
        return hdr

    def subframes(self, t, shutter=1.0 / 60.0):
        """Enough temporal samples that fast cubes smear instead of strobing."""
        Ca, Cb = TL.camera(t - shutter / 2), TL.camera(t + shutter / 2)
        Ba, _, _, _, _ = TL.blocks(t - shutter / 2)
        Bb, vis, _, _, _ = TL.blocks(t + shutter / 2)
        worst = 0.0
        for k in vis:
            pa = self._project(Ca, Ba[k, 0:3])
            pb = self._project(Cb, Bb[k, 0:3])
            if pa is None or pb is None:
                continue
            ra = np.linalg.norm(Ba[k, 3:12] - Bb[k, 3:12]) * pa[2]
            worst = max(worst, math.hypot(pa[0] - pb[0], pa[1] - pb[1]) + ra)
        return int(min(10, max(1, math.ceil(worst / 4.0))))

    def _project(self, C, p):
        rel = p - C[0:3]
        zc = rel @ C[9:12]
        if zc < 0.2:
            return None
        sx = (rel @ C[3:6] / zc / C[12] + 1) * 0.5 * self.w
        sy = (1 - rel @ C[6:9] / zc / C[13]) * 0.5 * self.h
        size = S.B / zc / C[13] * 0.5 * self.h
        return sx, sy, size

    def frame(self, fi):
        t = fi / TL.FPS
        if t >= 36.0:
            import logo
            return logo.render_frame(t, self.w, self.h, fi)
        sub = self.subframes(t)
        hdr = self.render_hdr(t, subframes=sub)
        img = post.grade_image(hdr, TL.grade(t))
        img = post.draw_titles(img, t)
        img = post.grain(img, fi)
        return post.to_uint8(img, fi)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if cmd == 'assets':
        import stained_glass
        stained_glass.save_assets(BUILD)
        return
    if cmd == 'still':
        t = float(sys.argv[2])
        out = sys.argv[3]
        R = Renderer()
        fi = int(round(t * TL.FPS))
        t0 = time.time()
        img = R.frame(fi)
        print('frame %d rendered in %.2fs' % (fi, time.time() - t0))
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return
    if cmd == 'frames':
        a, b = int(sys.argv[2]), int(sys.argv[3])
        step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        os.makedirs(FRAMES, exist_ok=True)
        R = Renderer()
        for fi in range(a, b, step):
            path = os.path.join(FRAMES, 'f%04d.png' % fi)
            if os.path.exists(path):
                continue
            t0 = time.time()
            img = R.frame(fi)
            cv2.imwrite(path + '.tmp.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            os.replace(path + '.tmp.png', path)
            print('frame %4d  t=%6.3f  %.2fs' % (fi, fi / TL.FPS, time.time() - t0), flush=True)
        return
    if cmd == 'encode':
        import encode
        encode.main(sys.argv[2:])
        return
    print(__doc__)


if __name__ == '__main__':
    main()
