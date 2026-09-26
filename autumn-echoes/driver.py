"""Headless-Chromium frame driver: serves web/, calls window.renderFrame(i), saves PNGs."""
import base64
import functools
import http.server
import os
import socketserver
import sys
import threading
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, 'web')
CHROME = os.environ.get('CHROME', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
FLAGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
         '--enable-webgl', '--disable-web-security', '--disable-dev-shm-usage', '--js-flags=--max-old-space-size=8192']


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def serve():
    handler = functools.partial(Quiet, directory=WEB)
    httpd = socketserver.ThreadingTCPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


class Session:
    def __init__(self, page_name, query='', width=1080, height=1920, timeout=600000):
        self.httpd, port = serve()
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(executable_path=CHROME, headless=True, args=FLAGS)
        self.page = self.browser.new_page(viewport={'width': width, 'height': height})
        self.page.set_default_timeout(timeout)
        self.logs = []
        self.page.on('console', lambda m: self.logs.append(m.text))
        self.page.on('pageerror', lambda e: self.logs.append('PAGEERROR ' + str(e)))
        self.page.goto('http://127.0.0.1:%d/%s%s' % (port, page_name, query))
        self.page.wait_for_function('window.READY === true || window.FAILED', timeout=timeout)
        if self.page.evaluate('window.FAILED || null'):
            raise RuntimeError('page failed: %s\n%s' % (self.page.evaluate('window.FAILED'), '\n'.join(self.logs)))

    def frame(self, i, path=None):
        r = self.page.evaluate('(i) => window.renderFrame(i)', i)
        url = r['url'] if isinstance(r, dict) else r
        data = base64.b64decode(url.split(',', 1)[1])
        if path:
            with open(path, 'wb') as f:
                f.write(data)
        return r, data

    def close(self):
        self.browser.close()
        self.pw.stop()
        self.httpd.shutdown()


if __name__ == '__main__':
    s = Session(sys.argv[1] if len(sys.argv) > 1 else 'bench.html')
    print('\n'.join(s.logs[-5:]))
    info = s.page.evaluate("(() => { const gl = document.createElement('canvas').getContext('webgl2'); "
                           "const d = gl.getExtension('WEBGL_debug_renderer_info'); "
                           "return [gl.getParameter(d ? d.UNMASKED_RENDERER_WEBGL : gl.RENDERER), gl.getParameter(gl.MAX_TEXTURE_SIZE), "
                           "!!gl.getExtension('EXT_color_buffer_float'), gl.getParameter(gl.MAX_SAMPLES)]; })()")
    print('renderer', info)
    for i in range(4):
        t0 = time.time()
        r, data = s.frame(i, os.path.join(HERE, 'bench_%d.png' % i) if i == 3 else None)
        print('frame %d: render %.0f ms, encode %.0f ms, total %.2f s, %d KB' % (i, r['render'], r['encode'], time.time() - t0, len(data) // 1024))
    s.close()
