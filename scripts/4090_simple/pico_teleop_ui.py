"""Local-only PICO teleoperation display and queued operator intentions.

HTTP threads never access an SDK, model, or simulator. ``update`` copies image
pixels without JPEG work; request threads encode the latest immutable snapshot.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
import math
from pathlib import Path
import queue
import select
import socket
import struct
import threading
import time
import uuid
from urllib.parse import urlsplit

import numpy as np


COMMANDS = frozenset({"calibrate", "follow", "pause", "record", "save", "discard", "reset", "quit", "toggle_navigation"})
BUSY_STATES = frozenset({"saving", "auditing", "resetting"})
MOBILE_COLLECTION_LOCKED_STATES = frozenset({"awaiting_review", "complete", "error"})
MOBILE_COLLECTION_LOCKED_COMMANDS = frozenset({"calibrate", "follow", "record", "reset", "toggle_navigation"})


def _mobile_collection_locked(status):
    progress = status.get("mobile_collection_progress")
    return isinstance(progress, dict) and progress.get("status") in MOBILE_COLLECTION_LOCKED_STATES


_PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PICO 遥操作采集</title><style>
:root{font-family:system-ui,sans-serif;color:#e8eef7;background:#101722;color-scheme:dark}
body{max-width:1240px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 8px}.muted{color:#acbacd}
.notice{background:#263749;color:#e7d69c;padding:12px 16px;border-radius:10px;margin:18px 0}
.view{background:#070b11;border:1px solid #344255;border-radius:14px;min-height:260px;display:grid;place-items:center;overflow:hidden}
img{display:block;width:100%;max-height:65vh;object-fit:contain}img[hidden]{display:none}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:18px 0}
button{font:inherit;background:#26384d;border:1px solid #52677f;border-radius:9px;color:white;padding:11px 16px;cursor:pointer}
button:hover{background:#3b516b}button:disabled{opacity:.5;cursor:wait}.primary{background:#23568f}.danger{border-color:#b26363}
#error{color:#ffb3ac;white-space:pre-wrap}#message{min-height:24px;color:#afcdef}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;color:#b9c9dc}details{margin-top:20px}#phase{font-weight:600}
</style><h1>PICO 遥操作采集</h1><div class="muted">当前状态：<span id="phase">等待电脑状态</span></div>
<div class="notice">原始示例 · 待独立验收。退出采集后补图并独立验收，合格后才能用于训练。</div>
<p id="instructions" class="muted">X 标定 · Y 跟随／暂停 · A 录制／保存 · 左右前扳机或侧握键抓握。开始录制后先保持姿势 1 秒。B 未绑定相机开关。</p>
<p id="mobile-status" role="status" hidden></p>
<p id="mobile-collection-progress" role="status" hidden></p>
<p id="tracking-status" role="status">等待PICO追踪状态</p>
<p id="grasp" role="status">等待抓握反馈</p>
<p id="preview-feedback" role="status">等待相机</p>
<p id="last-episode" role="status">上一条：暂无已完成录制</p>
<div id="review-panel"><p id="review-status" role="status">后台复核：准备中</p><p id="review-result" role="status">最近完成：暂无</p><details><summary>本会话复核记录</summary><ul id="review-results"></ul></details><details id="review-history" hidden><summary>历史记录（其它会话，原件保留）</summary><ul id="review-history-results"></ul></details></div>
<div class="view"><span id="placeholder" class="muted">等待仿真画面</span><img id="camera" src="/stream.mjpg" alt="当前仿真相机画面" hidden></div>
<div class="bar">
<button data-command="calibrate">标定</button><button data-command="follow">开始跟随</button><button data-command="pause">暂停跟随</button>
<button data-command="toggle_navigation" hidden>B 切换移动／手臂操作</button>
<button class="primary" data-command="record">开始录制</button><button class="primary" data-command="save">保存本条</button>
<button data-command="discard">丢弃上一条</button><button data-command="reset">重置场景</button><button class="danger" data-command="quit">退出</button>
</div><div id="message" role="status"></div><div id="error" role="alert"></div>
<details><summary>查看运行详情</summary><pre id="details"></pre></details>
<script>
const camera=document.querySelector('#camera');let streamRetry=null,statusConnected=false,lastOperatorMessage=null,imageAvailable=false,cameraReady=false,lastImageEpoch=null,lastStatus=null,lastOutput=null,lastUIInstance=null,sessionGeneration=0,refreshSequence=0,refreshInFlight=false;const pendingButtons=new Map();
function clearCamera(){imageAvailable=false;cameraReady=false;lastImageEpoch=null;if(streamRetry!==null){clearTimeout(streamRetry);streamRetry=null;}camera.removeAttribute('src');showCamera();}
function resetSessionView(){sessionGeneration++;clearCamera();lastStatus=null;lastOperatorMessage=null;pendingButtons.clear();document.querySelector('#message').textContent='';}
function absolutePath(value){if(typeof value!=='string'||!value.startsWith('/')||value.includes(String.fromCharCode(0)))return null;const parts=value.split('/').filter(Boolean);return parts.some(part=>part==='.'||part==='..')?null:'/'+parts.join('/');}
function belongsToSession(record,output){if(!output)return true;const root=absolutePath(output),source=absolutePath(record?.source_path);return !!(root&&source&&source!==root&&source.startsWith(root==='/'?'/':root+'/'));}
function sessionOutput(s){return typeof s.output==='string'?s.output:(lastOutput??(lastUIInstance!==null?'awaiting-session-output':null));}
function sessionLastEpisode(s){return belongsToSession(s.last_episode,sessionOutput(s))?s.last_episode:null;}
function showCamera(){camera.hidden=!(imageAvailable&&cameraReady);document.querySelector('#placeholder').hidden=!camera.hidden;}
function reconnectStream(){if(streamRetry!==null||!imageAvailable)return;cameraReady=false;showCamera();streamRetry=setTimeout(()=>{streamRetry=null;if(imageAvailable)camera.src='/stream.mjpg?retry='+Date.now();},500);}
function updateCameraAvailability(available,version,epoch){if(Number.isInteger(epoch)){if(lastImageEpoch!==null&&epoch<lastImageEpoch)return;if(lastImageEpoch!==null&&epoch!==lastImageEpoch){cameraReady=false;camera.removeAttribute('src');if(streamRetry!==null){clearTimeout(streamRetry);streamRetry=null;}}lastImageEpoch=epoch;}const next=available===true;if(!next){imageAvailable=false;cameraReady=false;if(streamRetry!==null){clearTimeout(streamRetry);streamRetry=null;}camera.removeAttribute('src');}else{const restart=!imageAvailable||!camera.getAttribute('src');imageAvailable=true;if(restart){cameraReady=false;camera.src='/stream.mjpg?version='+version;}}showCamera();}
function previewFeedbackText(feedback,available){if(feedback?.error||feedback?.status==='error')return '相机异常';if(!feedback)return available?'相机预览':'等待相机';const limit=Number.isFinite(feedback.max_age_s)&&feedback.max_age_s>0?Math.min(.1,feedback.max_age_s):.1;return available&&['ready','fresh','ok'].includes(feedback.status)&&Number.isFinite(feedback.age_s)&&feedback.age_s>=0&&feedback.age_s<=limit?'相机预览':'等待相机';}
function trackingFeedbackText(t){if(t?.valid)return '全身追踪有效 · 手柄输入已连接';const labels={no_tracking_packet:'未收到数据，请保持XRoboToolkit前台并开启Send',controller_input_required:'请在头显切换到手柄输入模式',app_not_focused:'请保持XRoboToolkit在前台',head_tracking_invalid:'头部追踪无效，请戴好头显',body_missing_shape_or_nonfinite:'缺少完整24点全身追踪，请检查全身标定',stale:'数据已过期，请检查追踪发送连接',input_mode_changed:'输入模式刚切换，请等稳定后重新标定',device_timestamp_gap:'追踪数据中断，请检查发送连接'};return '追踪不可用：'+((t?.reasons??[]).map(reason=>labels[reason]??reason).join('；')||'等待数据');}
function mobileCollectionLocked(p){return ['awaiting_review','complete','error'].includes(p?.status);}
function mobileCollectionProgressText(p){const total=Number.isInteger(p.total)&&p.total>0?p.total:'待确认';const accepted=Number.isInteger(p.accepted_count)&&p.accepted_count>=0?p.accepted_count:'待确认';if(p.status==='complete')return '已完成'+accepted+'/'+total+'份官方TRAIN初始化采集';const position=Number.isInteger(p.current_position)&&p.current_position>0&&Number.isInteger(p.current_official_episode_index)&&p.current_official_episode_index>=0?'第'+p.current_position+'/'+total+'份，官方TRAIN episode'+String(p.current_official_episode_index).padStart(3,'0')+'；':'';const progress=position+'已完成'+accepted+'/'+total;if(p.status==='awaiting_review')return progress+' · 等待复核，复核完成后继续下一份';if(p.status==='error')return progress+' · 已暂停：技术错误'+(p.error?' · '+(typeof p.error==='string'?p.error:JSON.stringify(p.error)):'');return progress;}
function updateCommandButtons(s){s=s??{};const busy=!statusConnected||['saving','auditing','resetting'].includes(s.status);for(const b of document.querySelectorAll('button[data-command]'))b.disabled=busy||pendingButtons.has(b);const discard=document.querySelector('button[data-command="discard"]'),last=sessionLastEpisode(s);if(!busy&&!s.recording)discard.disabled=pendingButtons.has(discard)||!(last?.available||(last?.pending&&last?.status!=='pending_discard'));if(mobileCollectionLocked(s.mobile_collection_progress))for(const b of document.querySelectorAll('button[data-command]'))if(['calibrate','follow','record','reset','toggle_navigation'].includes(b.dataset.command))b.disabled=true;}

function episodeText(v){if(!v?.uuid)return '暂无';const id=v.number!=null?String(v.number).padStart(3,'0'):'待编号（'+v.uuid.slice(0,8)+'）';const state=v.status==='pending_discard'?'已申请丢弃，后台处理中':v.status==='technical_blocked'||v.selection_status==='technical_evidence_incomplete'?'技术异常，原件保留':v.discarded||v.disposition==='discarded'?(v.selection_status==='official_task_not_successful'?'未达到官方任务成功条件':'已丢弃'):v.disposition==='kept'?(v.selection_policy==='native_only'?'官方任务成功，已保留':'物理验收通过，已保留'):v.status==='running'?'复核中':v.status==='error'?'复核异常':v.pending?'待复核':v.collection_enabled===false?'诊断已完成（不计真人数据）':'待审核';return id+' · '+state+((v.reasons??[]).length?' · 原因：'+v.reasons.join('；'):'')+(v.error?' · '+v.error:'');}
function renderReview(q,output,last){const labels={starting:'正在准备',idle:'空闲',working:'正在复核',recovering:'正在恢复任务',error:'复核异常'};document.querySelector('#review-status').textContent=(output?'后台复核（全部会话）：':'后台复核：')+(q?(labels[q.status]??q.status)+' · 未完成 '+(q.pending_count??0)+' 条'+(q.error?' · '+q.error:''):'准备中');const recent=(q?.latest_results??[]).filter(v=>belongsToSession(v,output));const completed=belongsToSession(q?.latest_completed,output)?q?.latest_completed:[...recent].reverse().find(v=>v.pending===false);document.querySelector('#review-result').textContent=(output?'本会话最近完成：':'最近完成：')+episodeText(completed);const list=document.querySelector('#review-results');if(list){list.replaceChildren();for(const v of recent){const row=document.createElement('li');row.textContent=episodeText(v);list.appendChild(row);}}const history=document.querySelector('#review-history-results');if(history){history.replaceChildren();const seen=new Set();for(const v of [q?.latest_episode,q?.latest_completed,...(q?.latest_results??[]),last])if(v?.uuid&&!belongsToSession(v,output)&&!seen.has(v.uuid)){seen.add(v.uuid);const row=document.createElement('li');row.textContent='历史 · '+episodeText(v);history.appendChild(row);}document.querySelector('#review-history').hidden=seen.size===0;}}
function disconnected(error){statusConnected=false;clearCamera();updateCommandButtons(lastStatus);document.querySelector('#phase').textContent='电脑界面连接中断';document.querySelector('#tracking-status').textContent='服务未连接，追踪状态非实时';document.querySelector('#preview-feedback').textContent='服务未连接，画面非实时';document.querySelector('#grasp').textContent='服务未连接，抓握反馈非实时';document.querySelector('#error').textContent=error.message;}

camera.onerror=reconnectStream;
camera.onload=()=>{cameraReady=imageAvailable;showCamera();};
async function refresh(){if(refreshInFlight)return;refreshInFlight=true;const sequence=++refreshSequence,controller=new AbortController();let timeout=null;try{
 const v=await Promise.race([(async()=>{const r=await fetch('/status.json',{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error('读取状态失败');return r.json();})(),new Promise((resolve,reject)=>{timeout=setTimeout(()=>{controller.abort();reject(Error('读取电脑状态超时，当前画面非实时'));},1500);})]);if(sequence!==refreshSequence)return;const s=v.status;
 const instance=v.ui_session_id??null,output=typeof s.output==='string'?s.output:null,newInstance=instance!==null&&instance!==lastUIInstance;if(newInstance)lastOutput=null;if(newInstance||(output!==null&&output!==lastOutput))resetSessionView();if(instance!==null)lastUIInstance=instance;if(output!==null)lastOutput=output;
 lastStatus=s;const collection=s.mobile_collection_progress;const progress=document.querySelector('#mobile-collection-progress');progress.hidden=!collection;progress.textContent=collection?mobileCollectionProgressText(collection):'';
 document.querySelector('#tracking-status').textContent=trackingFeedbackText(s.tracking);
 const mobile=s.collection_task_profile==='mobile_pp';const continuous=mobile&&s.mobile_control_policy==='continuous_vr_mode1';document.querySelector('button[data-command="toggle_navigation"]').hidden=!mobile||continuous;
 const mobileStatus=document.querySelector('#mobile-status');mobileStatus.hidden=!mobile;
 if(continuous){document.querySelector('h1').textContent='Mobile P&P 双桌搬运数采';document.querySelector('#instructions').textContent='X 标定 · Y 跟随／暂停 · A 录制／保存。双臂持续跟随，无需B切换。左杆向前、右杆转向，松杆停步；扳机控制抓握。开始前松开摇杆，起录后先保持姿势1秒。';mobileStatus.textContent='控制模式：'+({continuous_paused:'连续控制已暂停',preparing_continuous:'正在准备连续控制，仿真暂停',continuous:'双臂实时跟随＋摇杆移动'}[s.mobile_mode]??'等待标定');}
 else if(mobile){document.querySelector('h1').textContent='Mobile P&P 双桌搬运数采';document.querySelector('#instructions').textContent='X 标定 · Y 跟随／暂停 · A 录制／保存 · B 切换移动／手臂操作。移动时左杆向前、右杆转向，上身保持进入时姿态，扳机仍控制抓握。松杆站稳并恢复入场人体姿态后，按B恢复手臂操作。起录后先保持姿势1秒。';mobileStatus.textContent='控制模式：'+({pose:'手臂操作',preparing_navigation:'正在准备移动，请松开摇杆',navigation:'移动：上身保持、手指可控'}[s.mobile_mode]??'等待准备');}
 if(!statusConnected)reconnectStream();statusConnected=true;
 let phase=s.phase??s.state??s.status??'等待运行';if(typeof phase==='object')phase=phase.status??JSON.stringify(phase);
 const labels={paused:'已暂停',following:'正在跟随',saving:'正在封存本条',auditing:'正在物理验收',resetting:'正在重置场景',stopped:'采集已结束'};
 if(s.recording==='warmup')phase='请保持姿势，正在记录起始状态';
 else if(s.recording==='recording')phase='正在录制';
 if(collection?.status==='awaiting_review')phase='等待复核';else if(collection?.status==='complete')phase='初始化采集已完成';else if(collection?.status==='error')phase='技术错误，已暂停';
 document.querySelector('#phase').textContent=labels[phase]??phase;
 const h=s.hand_control;const pct=x=>Number.isFinite(x)?Math.round(x*100)+'%':'未下发';
 document.querySelector('#grasp').textContent=h?['left','right'].map((side,i)=>{const v=h[side];return (i?'右':'左')+'：扳机 '+pct(v.trigger)+' · 侧握 '+pct(v.grip)+' · 下发 '+pct(v.applied_closed_fraction)+' · 实际闭合 '+pct(v.actual_closed_fraction);}).join('　／　')+'（闭合不代表抓住）':'当前接收端暂无抓握反馈';
 const last=sessionLastEpisode(s),outputScope=sessionOutput(s);document.querySelector('#last-episode').textContent=(outputScope?'本会话上一条：':'上一条：')+(last?.uuid?episodeText(last):'暂无已完成录制');renderReview(s.background_review,outputScope,s.last_episode);
 const discard=document.querySelector('button[data-command="discard"]');discard.textContent=s.recording?'丢弃当前录制':'丢弃上一条';
 updateCommandButtons(s);
 if(s.operator_message&&(s.operator_message!==lastOperatorMessage||document.querySelector('#message').textContent!==s.operator_message)){document.querySelector('#message').textContent=s.operator_message;lastOperatorMessage=s.operator_message;}
 document.querySelector('#details').textContent=JSON.stringify(s,null,2);
 const err=(collection?.status==='error'?collection.error:null)??s.error??s.last_error??s.reason??'';document.querySelector('#error').textContent=typeof err==='string'?err:JSON.stringify(err);
 updateCameraAvailability(v.image_available,v.image_version,v.image_epoch);
 document.querySelector('#preview-feedback').textContent=previewFeedbackText(s.preview_feedback,v.image_available);
 }catch(e){if(sequence===refreshSequence)disconnected(e);}finally{if(timeout!==null)clearTimeout(timeout);if(sequence===refreshSequence)refreshInFlight=false;}
}
for(const b of document.querySelectorAll('button[data-command]'))b.addEventListener('click',async()=>{const intention={generation:sessionGeneration,sequence:refreshSequence};pendingButtons.set(b,intention);b.disabled=true;try{
 const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:b.dataset.command})});
 const v=await r.json();if(!r.ok)throw Error(v.error??'提交失败');if(intention.generation===sessionGeneration&&intention.sequence===refreshSequence)document.querySelector('#message').textContent='已提交，等待电脑执行：'+b.textContent;
 }catch(e){if(intention.generation===sessionGeneration&&intention.sequence===refreshSequence)document.querySelector('#error').textContent=e.message;}finally{if(pendingButtons.get(b)===intention)pendingButtons.delete(b);updateCommandButtons(lastStatus);}});
updateCommandButtons(null);refresh();setInterval(refresh,500);
</script></html>"""


def _copy_status(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("UI status keys must be strings")
        return {key: _copy_status(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_status(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("UI status cannot contain object arrays")
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"UI status cannot contain {type(value).__name__}")


def _json_status(value):
    if isinstance(value, np.ndarray):
        return _json_status(value.tolist())
    if isinstance(value, dict):
        return {key: _json_status(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_status(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # Show invalid telemetry as missing, never valid JSON NaN.
    return value


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True


class _Server6(_Server):
    address_family = socket.AF_INET6


class LocalTeleopUI:
    """An HTTP display that only queues intentions for the owning main loop."""

    def __init__(self, host="127.0.0.1", port=43100):
        if host == "localhost":
            host = "127.0.0.1"
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("The teleoperation UI must bind a literal loopback address") from error
        if not address.is_loopback:
            raise ValueError("The teleoperation UI must remain on loopback")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid local UI port")
        self._host, self._requested_port = str(address), port
        self._ipv6 = address.version == 6
        self._lock = threading.Lock()
        self._frame_changed = threading.Condition(self._lock)
        self._encode_lock = threading.Lock()
        self._stream_connections = set()
        self._commands = queue.Queue(maxsize=64)
        self._ui_session_id = uuid.uuid4().hex
        self._status = {"status": "等待仿真", "accepted_new_dataset": False}
        self._rgb = None
        self._image_version = 0
        self._image_epoch = 0
        self._jpeg_version = -1
        self._jpeg = None
        self._server = self._thread = None
        self._closed = False

    @property
    def url(self):
        if self._server is None:
            raise RuntimeError("Start the local UI before requesting its URL")
        host = f"[{self._host}]" if self._ipv6 else self._host
        return f"http://{host}:{self._server.server_address[1]}/"

    def start(self):
        if self._closed:
            raise RuntimeError("A closed UI cannot be restarted")
        if self._server is not None:
            return self
        ui = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def setup(self):
                super().setup()
                self.connection.settimeout(1.)

            def log_message(self, *args):
                pass

            def _send(self, code, body, content_type="application/json; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

            def _json(self, code, value):
                self._send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))

            def _valid_host(self):
                values = self.headers.get_all("Host", [])
                if len(values) != 1 or values[0] not in ui._authorities():
                    self._json(403, {"error": "Only the local UI host is allowed"})
                    return False
                return True

            def do_GET(self):
                if not self._valid_host():
                    return
                path = urlsplit(self.path).path
                if path == "/":
                    self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/status.json":
                    self._json(200, ui._status_snapshot())
                elif path == "/stream.mjpg":
                    self._stream_images()
                elif path == "/latest.jpg":
                    try:
                        encoded = ui._latest_jpeg()
                    except Exception as error:
                        self._json(500, {"error": f"Preview encoding failed: {type(error).__name__}"})
                        return
                    if encoded is None:
                        self._json(404, {"error": "No real camera frame has arrived"})
                    else:
                        self._send(200, encoded, "image/jpeg")
                else:
                    self._json(404, {"error": "Unknown local UI endpoint"})

            def _stream_peer_connected(self):
                # A stream with no new images must still notice browser close,
                # rather than retain a non-daemon request thread indefinitely.
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0.)
                    return not readable or bool(self.connection.recv(1, socket.MSG_PEEK))
                except OSError:
                    return False

            def _stream_images(self):
                with ui._lock:
                    if ui._closed:
                        return
                    ui._stream_connections.add(self.connection)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    previous, next_send = -1, 0.
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    while True:
                        with ui._frame_changed:
                            if ui._closed:
                                return
                            delay = next_send - time.monotonic()
                            ready = ui._rgb is not None and ui._image_version != previous
                            if not ready or delay > 0:
                                ui._frame_changed.wait(timeout=min(.1, delay) if ready and delay > 0 else .1)
                                ready = False
                        if not self._stream_peer_connected():
                            return
                        if not ready:
                            continue
                        encoded, version = ui._latest_jpeg_snapshot()
                        if encoded is None or version == previous:
                            continue
                        with ui._lock:
                            if ui._closed:
                                return
                        # A paused camera may have only one frame. Finish its
                        # MIME part now so browsers display it without waiting
                        # for a second simulator frame to start the next part.
                        self.wfile.write(encoded + b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.flush()
                        previous, next_send = version, time.monotonic() + 1. / 30
                except (OSError, ValueError):
                    # Disconnect, socket timeout and server shutdown are normal
                    # stream endings. The browser reconnects without trace spam.
                    return
                except Exception:
                    # An encoder failure ends this response; /latest.jpg still
                    # provides a bounded diagnostic and the image may be retried.
                    return
                finally:
                    with ui._lock:
                        ui._stream_connections.discard(self.connection)

            def do_POST(self):
                if not self._valid_host():
                    return
                origins = self.headers.get_all("Origin", [])
                if origins and (len(origins) != 1 or origins[0] not in ui._origins()):
                    self._json(403, {"error": "Cross-origin control requests are forbidden"})
                    return
                if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
                    self._json(403, {"error": "Cross-site control requests are forbidden"})
                    return
                if urlsplit(self.path).path not in ("/command", "/commands"):
                    self._json(404, {"error": "Unknown control endpoint"})
                    return
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    self._json(415, {"error": "Control requests require application/json"})
                    return
                lengths = self.headers.get_all("Content-Length", [])
                try:
                    if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                        raise ValueError("An explicit body length is required")
                    length = int(lengths[0])
                    if not 0 < length <= 1024:
                        raise ValueError("Control request body is too large or empty")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Incomplete control request body")
                    value = json.loads(body)
                    if (not isinstance(value, dict) or set(value) != {"command"}
                            or not isinstance(value["command"], str) or value["command"] not in COMMANDS):
                        raise ValueError("Unknown operator command")
                except (ValueError, UnicodeDecodeError) as error:
                    self._json(400, {"error": str(error)})
                    return
                except (TimeoutError, OSError):
                    self._json(408, {"error": "Control request body timed out"})
                    return
                try:
                    ui._queue_command(value["command"])
                except queue.Full:
                    self._json(503, {"error": "Control queue is full"})
                    return
                except RuntimeError as error:
                    self._json(503, {"error": str(error)})
                    return
                self._json(202, {"queued": value["command"], "executed": False, "accepted_new_dataset": False})

            def do_OPTIONS(self):
                self._json(403, {"error": "Cross-origin control is not enabled"})

        server_cls = _Server6 if self._ipv6 else _Server
        self._server = server_cls((self._host, self._requested_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
            kwargs={"poll_interval": .1}, name="PicoLocalTeleopUI", daemon=False)
        self._thread.start()
        return self

    def _authorities(self):
        port = self._server.server_address[1]
        host = f"[{self._host}]" if self._ipv6 else self._host
        return {f"{host}:{port}", f"localhost:{port}"}

    def _origins(self):
        return {"http://" + authority for authority in self._authorities()}

    def _queue_command(self, command):
        with self._lock:
            if self._closed:
                raise RuntimeError("The UI is closing")
            if self._status.get("status") in BUSY_STATES and command != "quit":
                raise RuntimeError("正在保存、验收或重置，请等待操作完成后再点击")
            if _mobile_collection_locked(self._status) and command in MOBILE_COLLECTION_LOCKED_COMMANDS:
                raise RuntimeError("初始化采集正在等待复核、已完成或因技术错误暂停，请查看采集进度")
            self._commands.put_nowait(command)

    def commands(self):
        result = []
        while True:
            try:
                result.append(self._commands.get_nowait())
                self._commands.task_done()
            except queue.Empty:
                return result

    def update(self, status, rgb=None, *, clear_image=False):
        if not isinstance(status, dict):
            raise TypeError("UI status must be a dictionary")
        if type(clear_image) is not bool:
            raise TypeError("clear_image must be a boolean")
        copied_status = _copy_status(status)
        copied_status["accepted_new_dataset"] = False
        pixels = None
        if rgb is not None:
            pixels = np.asarray(rgb)
            if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                    or min(pixels.shape[:2]) <= 0 or max(pixels.shape[:2]) > 4096):
                raise ValueError("Preview requires actual uint8 RGB HxWx3 pixels")
            pixels = pixels.copy(order="C")
            pixels.flags.writeable = False
        with self._lock:
            if self._closed:
                raise RuntimeError("The UI is closed")
            if copied_status.get("status") in BUSY_STATES and self._status.get("status") not in BUSY_STATES:
                # Commands issued against the previous episode must not act on
                # the newly archived episode after its automatic reset.
                pending = self.commands()
                if "quit" in pending:
                    self._commands.put_nowait("quit")
            elif _mobile_collection_locked(copied_status) and not _mobile_collection_locked(self._status):
                # Do not replay a pre-review start/reset intention against the
                # next official initialization. Keep stop/discard/save intents.
                for command in self.commands():
                    if command not in MOBILE_COLLECTION_LOCKED_COMMANDS:
                        self._commands.put_nowait(command)
            self._status = copied_status
            if clear_image:
                # Never wait for JPEG encoding or a streaming socket to retire
                # the earlier scene. In-flight encoders check the generation.
                self._rgb = self._jpeg = None
                self._jpeg_version = -1
                self._image_epoch += 1
            if pixels is not None:
                self._rgb = pixels
            if clear_image or pixels is not None:
                self._image_version += 1
                self._frame_changed.notify_all()

    def _status_snapshot(self):
        with self._lock:
            status, version, epoch, available = self._status, self._image_version, self._image_epoch, self._rgb is not None
        return {"status": _json_status(status), "ui_session_id": self._ui_session_id,
                "image_version": version, "image_epoch": epoch,
                "image_available": available, "accepted_new_dataset": False,
                "data_label": "原始数据：待独立验收", "queued_commands": self._commands.qsize()}

    @staticmethod
    def _encode_jpeg(pixels):
        from PIL import Image
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=85)
        return stream.getvalue()

    def _latest_jpeg(self):
        return self._latest_jpeg_snapshot()[0]

    def _latest_jpeg_snapshot(self):
        with self._encode_lock:
            with self._lock:
                pixels, version = self._rgb, self._image_version
                if pixels is None:
                    return None, version
                if self._jpeg_version == version:
                    return self._jpeg, version
            encoded = self._encode_jpeg(pixels)
            with self._lock:
                if self._closed or self._image_version != version or self._rgb is not pixels:
                    # Clear/reset or a newer scene wins over an old encoding.
                    # HTTP callers may retry; never republish an obsolete JPEG.
                    return None, self._image_version
                self._jpeg, self._jpeg_version = encoded, version
            return encoded, version

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._frame_changed.notify_all()
            streams = tuple(self._stream_connections)
        # Interrupt a client that stopped reading, rather than waiting for its
        # socket send timeout while server_close joins non-daemon workers.
        for connection in streams:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()


class HeadsetVideoBridge:
    """Optional real stereo-camera return path, independent of SDK/physics.

    The PICO app sends OPEN_CAMERA to ``host:port`` using its existing binary
    protocol. Frames come only from ``update``; this class never renders or
    fabricates them. The expected headset IP is explicit for both incoming
    control and outgoing video. H.264 is supported; unverified MV-HEVC requests
    are rejected rather than silently changing their codec.
    """

    def __init__(self, *, expected_peer_ip, host="0.0.0.0", port=13579, fps=30,
                 connect_timeout=1., send_timeout=.5):
        peer = ipaddress.ip_address(expected_peer_ip)
        bind = ipaddress.ip_address(host)
        if peer.version != 4 or bind.version != 4 or peer.is_unspecified or peer.is_multicast:
            raise ValueError("Camera transport requires explicit IPv4 headset/bind addresses")
        if type(port) is not int or not 0 <= port <= 65535 or type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("Invalid camera port or frame rate")
        if not 0 < connect_timeout <= 5 or not 0 < send_timeout <= 5:
            raise ValueError("Camera connect/send timeouts must be finite and <=5 seconds")
        self.expected_peer_ip, self.host, self.port, self.fps = str(peer), str(bind), port, fps
        self.connect_timeout, self.send_timeout = float(connect_timeout), float(send_timeout)
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._server = self._streaming = self._frame_buffer = None
        self._components = None
        self._closed = False
        self._status = {"listening": False, "streaming": False, "frames_submitted": 0,
                        "frames_encoded": 0, "video_bytes_sent": 0, "error": None,
                        "codec": "h264", "expected_peer_ip": self.expected_peer_ip,
                        "headset_display_verified": False}

    @property
    def status(self):
        with self._lock:
            return _copy_status(self._status)

    def _set_status(self, **values):
        with self._lock:
            self._status.update(values)

    def start(self):
        if self._closed:
            raise RuntimeError("A closed headset bridge cannot be restarted")
        if self._server is not None:
            return self
        # Lazy imports: importing this module and using LocalTeleopUI does not
        # import SIMPLE, PyAV, Unitree, or any tracking SDK.
        from simple.teleop.pico.tcp_server import TCPControlServer
        from simple.teleop.pico.tcp_video_sender import TCPVideoSender
        from simple.teleop.pico.streaming import FrameBuffer, StreamingThread

        bridge = self

        class PeerControlServer(TCPControlServer):
            def _run(self):
                try:
                    while not self._stop_event.is_set():
                        try:
                            conn, address = self.listener.accept()
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if address[0] != bridge.expected_peer_ip:
                            conn.close()
                            bridge._set_status(error="Camera control peer IP rejected")
                            continue
                        try:
                            self._handle_client(conn)
                        except Exception as error:
                            bridge._set_status(error=f"Camera control failed: {type(error).__name__}: {error}")
                finally:
                    self.listener.close()

            def _read_framed(self, conn):
                header = self._recv_exactly(conn, 4)
                if header is None:
                    return None
                length, = struct.unpack(">I", header)
                if not 8 <= length <= 8192:
                    bridge._set_status(error="Camera control frame length rejected")
                    return None
                return self._recv_exactly(conn, length, deadline=time.monotonic() + 2.)

            def _recv_exactly(self, conn, length, deadline=None):
                chunks = bytearray()
                while len(chunks) < length and not self._stop_event.is_set():
                    if deadline is not None and time.monotonic() > deadline:
                        bridge._set_status(error="Partial camera control frame timed out")
                        return None
                    try:
                        chunk = conn.recv(length - len(chunks))
                    except socket.timeout:
                        continue
                    except OSError:
                        return None
                    if not chunk:
                        return None
                    if deadline is None:
                        deadline = time.monotonic() + 2.
                    chunks.extend(chunk)
                return bytes(chunks) if len(chunks) == length else None

            def stop(self):
                worker = self._thread
                self._stop_event.set()
                self.listener.close()
                conn = self._current_conn
                if conn is not None:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                super().stop()
                if worker is not None and worker.is_alive():
                    worker.join()

        class DeadlineVideoSender(TCPVideoSender):
            def __init__(self, **parameters):
                try:
                    super().__init__(**parameters)
                except BaseException:
                    sock = getattr(self, "_sock", None)
                    if sock is not None:
                        sock.close()
                        self._sock = None
                    raise

            def _connect(self):
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock.settimeout(bridge.connect_timeout)
                try:
                    self._sock.connect((self._ip, self._port))
                    self._sock.settimeout(bridge.send_timeout)
                except BaseException:
                    self._sock.close()
                    self._sock = None
                    raise

            def publish_raw_frame(self, frame):
                if self._closed:
                    return
                import av
                # Resize each eye independently and preserve its aspect ratio.
                # All resize/encode work runs in StreamingThread, not update.
                if frame.shape[:2] != (self._height, self._width):
                    source_h, source_w = frame.shape[0], frame.shape[1] // 2
                    target_w = self._width // 2
                    scale = min(target_w / source_w, self._height / source_h)
                    width, height = max(1, int(source_w * scale)), max(1, int(source_h * scale))
                    resized = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                    for eye in range(2):
                        original = frame[:, eye * source_w:(eye + 1) * source_w]
                        pixels = av.VideoFrame.from_ndarray(original, format="bgr24").reformat(
                            width=width, height=height, format="bgr24").to_ndarray()
                        y, x = (self._height - height) // 2, eye * target_w + (target_w - width) // 2
                        resized[y:y + height, x:x + width] = pixels
                    frame = resized
                try:
                    super().publish_raw_frame(frame)
                    with bridge._lock:
                        bridge._status["frames_encoded"] += 1
                except Exception as error:
                    bridge._set_status(streaming=False, error=f"Camera encode/send failed: {type(error).__name__}: {error}")
                    raise

            def _send_packets(self, packets):
                for packet in packets:
                    data = bytes(packet)
                    if data and self._sock is not None:
                        self._sock.sendall(struct.pack(">I", len(data)) + data)
                        with bridge._lock:
                            bridge._status["video_bytes_sent"] += len(data) + 4

            def close(self):
                if self._closed and self._sock is None:
                    return
                self._closed = True
                try:
                    if self._encoder is not None:
                        self._send_packets(self._encoder.encode(None))
                except Exception:
                    pass
                finally:
                    if self._sock is not None:
                        try:
                            self._sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        self._sock.close()
                        self._sock = None

        class PacedStreamingThread(StreamingThread):
            def _run(self):
                # Upstream StreamingThread uses fps only as a get() timeout;
                # enforce the negotiated publication cap for a 50 Hz producer.
                period, deadline = 1. / self._fps, time.monotonic()
                try:
                    while not self._stop_event.is_set():
                        if self._stop_event.wait(max(0., deadline - time.monotonic())):
                            break
                        frame = self._frame_buffer.get(timeout=min(period, .1))
                        if frame is None:
                            deadline = time.monotonic()
                            continue
                        for publisher in self._publishers:
                            publisher.publish_raw_frame(frame)
                        deadline = max(deadline + period, time.monotonic())
                except Exception as error:
                    bridge._set_status(streaming=False, error=f"Camera streaming stopped: {type(error).__name__}: {error}")
                finally:
                    for publisher in self._publishers:
                        publisher.close()
                    if self._on_ended:
                        self._on_ended()

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(1)
            listener.settimeout(.2)
            self.port = listener.getsockname()[1]
            control = PeerControlServer(f"{self.host}:{self.port}")
            control.listener = listener
            control.on_open_camera = self._open_camera
            control.on_close_camera = self._close_camera
            self._components = (DeadlineVideoSender, PacedStreamingThread)
            self._frame_buffer = FrameBuffer()
            self._server = control
            control.start()
            self._set_status(listening=True, control_address=f"{self.host}:{self.port}")
        except BaseException:
            listener.close()
            raise
        return self

    def _open_camera(self, request):
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._streaming is not None and self._streaming.is_running():
                return
            try:
                if str(ipaddress.ip_address(request.get("ip", ""))) != self.expected_peer_ip:
                    raise ValueError("Camera return IP does not match the connected headset")
                values = {key: request[key] for key in ("width", "height", "fps", "bitrate", "port")}
                if any(type(value) is not int for value in values.values()):
                    raise ValueError("Camera dimensions/rate/port must be integers")
                if (not 16 <= values["width"] <= 2560 or values["width"] % 2
                        or not 16 <= values["height"] <= 1440 or values["height"] % 2
                        or not 1 <= values["fps"] <= 60 or not 128000 <= values["bitrate"] <= 16000000
                        or not 1 <= values["port"] <= 65535):
                    raise ValueError("Camera request is outside bounded video limits")
                if request.get("enableMvHevc"):
                    raise ValueError("MV-HEVC is not verified; select H.264 for camera return")
                sender_cls, streaming_cls = self._components
                sender = sender_cls(ip=self.expected_peer_ip, port=values["port"],
                    width=values["width"], height=values["height"], fps=min(values["fps"], self.fps),
                    bitrate=values["bitrate"], hevc=False)
                self._streaming = streaming_cls(self._frame_buffer, fps=min(values["fps"], self.fps),
                                               publishers=[sender], on_ended=self._stream_ended)
                self._set_status(streaming=True, error=None, requested_camera=_copy_status(request),
                    output_fps=min(values["fps"], self.fps), output_shape=[values["height"], values["width"], 3])
                self._streaming.start()
            except Exception as error:
                self._set_status(streaming=False, error=f"Camera open failed: {type(error).__name__}: {error}")

    def _stream_ended(self):
        self._set_status(streaming=False)
        if self._server is not None:
            self._server.close_client()

    def _close_camera(self):
        with self._lifecycle_lock:
            if self._streaming is not None:
                worker = self._streaming._thread
                self._streaming.stop()
                if worker is not None and worker.is_alive():
                    worker.join()
                self._streaming = None
            self._set_status(streaming=False)

    def update(self, left_rgb, right_rgb=None):
        if self._closed or self._frame_buffer is None:
            return False
        if isinstance(left_rgb, dict) and right_rgb is None:
            observation = left_rgb
            left_rgb, right_rgb = observation.get("head_stereo_left"), observation.get("head_stereo_right")
        left, right = np.asarray(left_rgb), np.asarray(right_rgb)
        if (left.dtype != np.uint8 or right.dtype != np.uint8 or left.ndim != 3
                or right.shape != left.shape or left.shape[2] != 3 or min(left.shape[:2]) <= 0
                or max(left.shape[:2]) > 4096):
            raise ValueError("Headset return requires the actual matching left/right uint8 RGB frames")
        # Concatenation owns its pixels; the simulation can reuse both inputs.
        stereo_bgr = np.concatenate([left[..., ::-1], right[..., ::-1]], axis=1)
        self._frame_buffer.put(stereo_bgr)
        with self._lock:
            self._status["frames_submitted"] += 1
            self._status["last_frame_monotonic_s"] = time.monotonic()
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.stop()
        self._close_camera()
        self._set_status(listening=False, streaming=False)
