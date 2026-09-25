"""Encode the PNG frame sequence (+ soundtrack) into H.264 MP4 files."""
import os
import subprocess
import sys

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))


def ffmpeg(args):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, '-y', '-hide_banner', '-loglevel', 'error'] + args
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def encode(frames, out, audio=None, crf=20, fps=30):
    vf = 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p'
    args = ['-framerate', str(fps), '-i', os.path.join(frames, 'f%04d.png')]
    if audio:
        args += ['-i', audio]
    args += ['-vf', vf, '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-profile:v', 'high',
             '-tune', 'film', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
             '-movflags', '+faststart']
    if audio:
        args += ['-c:a', 'aac', '-b:a', '256k', '-shortest']
    args += [out]
    ffmpeg(args)


def main(argv):
    frames = argv[0] if argv else os.environ.get('BR_FRAMES', os.path.join(HERE, 'build', 'frames'))
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, 'output')
    audio = os.path.join(HERE, 'build', 'soundtrack.wav')
    if not os.path.exists(audio):
        import audio as A
        A.main(audio)
    os.makedirs(out_dir, exist_ok=True)
    encode(frames, os.path.join(out_dir, 'breathing_room.mp4'), audio)
    encode(frames, os.path.join(out_dir, 'breathing_room_silent.mp4'), None)


if __name__ == '__main__':
    main(sys.argv[1:])
