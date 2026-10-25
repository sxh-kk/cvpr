"""Read-only desktop camera viewer; shares the collector's existing frames.

No SDK, simulator, controller or recorder is initialized here. The collector
keeps its single PICO subscription and continues serving the control panel.
"""
from __future__ import annotations

import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import urllib.error
import urllib.request


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>4090 · SIMPLE 相机</title><style>
*{box-sizing:border-box}body{margin:0;background:#080e15;color:#edf2f8;font:16px system-ui}
header{height:56px;display:flex;align-items:center;gap:22px;padding:0 24px;background:#142130}
h1{font-size:18px;margin:0}#state{color:#a5d7b9}a{color:#b2d5ff;margin-left:auto}
main{height:calc(100vh - 104px);display:grid;place-items:center}img{width:100%;height:100%;object-fit:contain}
footer{height:48px;padding:12px 24px;color:#aebdcc;background:#142130}#error{color:#ffb9ac}
button{font:inherit;padding:5px 12px;color:#eaf3ff;background:#294260;border:0;border-radius:5px}
</style><header><h1>4090 · SIMPLE 实时相机</h1><span id="state">连接中</span>
<button id="full">全屏画面</button><a href="http://127.0.0.1:43100/" target="_blank">打开采集控制面板</a></header>
<main><img src="/stream.mjpg" alt="G1真实仿真相机画面"></main>
<footer>X 标定　Y 跟随／暂停　A 录制／保存　左右扳机抓握　<span id="error"></span></footer>
<script>
document.querySelector('#full').onclick=()=>document.documentElement.requestFullscreen();
async function refresh(){try{const response=await fetch('/status.json',{cache:'no-store'});if(!response.ok)throw Error('采集服务未连接');const v=await response.json();const s=v.status;
const labels={paused:'已暂停',following:'正在跟随',saving:'已暂停，正在保存',stopped:'采集已结束'};
const phase=s.recording==='recording'?'正在录制':s.recording==='warmup'?'保持姿势一秒':labels[s.status]??'等待采集状态';
document.querySelector('#state').textContent=phase+' · '+(s.tracking?.valid?'追踪正常':'等待有效追踪')+' · '+(s.calibrated?'已标定':'需要标定');
document.querySelector('#error').textContent=s.error??'';}catch(e){document.querySelector('#state').textContent='采集服务未连接';}}
refresh();setInterval(refresh,500);
</script></html>'''


def serve(port, collector_port):
    origin=f'http://127.0.0.1:{collector_port}'
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def log_message(self,*args):
            pass

        def do_GET(self):
            route=self.path.split('?',1)[0]
            if route=='/':
                payload=PAGE.replace('127.0.0.1:43100/',f'127.0.0.1:{collector_port}/').encode()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            elif route=='/status.json':
                try:
                    with urllib.request.urlopen(origin+'/status.json',timeout=2) as response:
                        payload=response.read(1024*1024)
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
                    self.end_headers();self.wfile.write(payload)
                except OSError:
                    self.send_error(503,'Collector unavailable')
            elif route=='/stream.mjpg':
                self.send_response(200)
                self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control','no-store');self.end_headers()
                try:
                    while True:
                        started=time.monotonic()
                        try:
                            with urllib.request.urlopen(origin+'/latest.jpg',timeout=2) as response:
                                frame=response.read(8*1024*1024)
                        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
                            time.sleep(.2)
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+
                            str(len(frame)).encode()+b'\r\n\r\n'+frame+b'\r\n')
                        self.wfile.flush()
                        time.sleep(max(0.,.05-(time.monotonic()-started)))
                except (BrokenPipeError,ConnectionResetError,TimeoutError):
                    pass
            else:
                self.send_error(404)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    print(f'Desktop preview: http://127.0.0.1:{port}/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=43101)
    parser.add_argument('--collector-port',type=int,default=43100)
    args=parser.parse_args()
    if not all(1<=port<=65535 for port in (args.port,args.collector_port)):
        parser.error('Invalid local TCP port')
    serve(args.port,args.collector_port)


if __name__=='__main__':
    main()
