#!/usr/bin/env python3
"""Drive the STAT web interface in headless Chrome over the DevTools protocol.

    python scripts/ui_probe.py --base http://127.0.0.1:8899 --dataset /path/to/ds --out /tmp/ui \
        --say "annotate cell types" --say "what can you do for me?" --reload

Needs Google Chrome and the ``websockets`` package (no Playwright). Prints console errors,
uncaught exceptions and layout metrics after every step and saves a screenshot per step.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME_CANDIDATES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]


def find_chrome() -> str:
    env = os.environ.get("STAT_CHROME")
    if env:
        return env
    for c in CHROME_CANDIDATES:
        if os.path.exists(c) or shutil.which(c):
            return c
    raise SystemExit("no Chrome/Chromium found; set STAT_CHROME")


class CDP:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.next_id = 0
        self.events: list[dict] = []

    async def send(self, method: str, **params):
        self.next_id += 1
        mid = self.next_id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    async def drain(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            try:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=max(0.05, end - time.time())))
            except TimeoutError:
                break
            if "method" in msg:
                self.events.append(msg)

    async def eval(self, expr: str, await_promise: bool = False):
        r = await self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=await_promise)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(f"eval failed: {d.get('text')} {json.dumps(d.get('exception', {}))[:300]}")
        return r.get("result", {}).get("value")

    async def wait_for(self, expr: str, timeout: float, poll: float = 0.4):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = await self.eval(expr)
            if v:
                return v
            await self.drain(poll)
        raise TimeoutError(f"timeout waiting for: {expr[:120]}")

    async def screenshot(self, path: Path) -> None:
        r = await self.send("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))

    def sse_summary(self) -> dict:
        """Events the browser's EventSource actually received (from the network layer)."""
        counts: dict[str, int] = {}
        for e in self.events:
            if e["method"] == "Network.eventSourceMessageReceived":
                name = e["params"].get("eventName") or "message"
                counts[name] = counts.get(name, 0) + 1
        statuses = [(e["params"]["response"].get("url", "")[-40:], e["params"]["response"].get("status")) for e in self.events if e["method"] == "Network.responseReceived" and "/api/" in e["params"]["response"].get("url", "")]
        failed = [(e["params"].get("errorText"), e["params"].get("requestId")) for e in self.events if e["method"] == "Network.loadingFailed"]
        return {"sse": counts, "api_responses": statuses[-12:], "failed": failed[-5:]}

    def problems(self) -> list[str]:
        out = []
        for e in self.events:
            if e["method"] == "Runtime.exceptionThrown":
                d = e["params"]["exceptionDetails"]
                exc = d.get("exception") or {}
                out.append(f"EXCEPTION {d.get('text')} {exc.get('description', '')[:400]}")
            elif e["method"] == "Runtime.consoleAPICalled" and e["params"]["type"] in ("error", "warning"):
                args = " ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])
                out.append(f"CONSOLE.{e['params']['type']} {args[:400]}")
        self.events = [e for e in self.events if e["method"] not in ("Runtime.exceptionThrown", "Runtime.consoleAPICalled")]
        return out


METRICS_JS = r"""
(() => {
  const chat = document.getElementById('chat-messages');
  const panel = chat && chat.closest('.chat-panel');
  const msgs = chat ? chat.querySelectorAll('.chat-msg') : [];
  const last = msgs.length ? msgs[msgs.length - 1] : null;
  if (chat) chat.scrollTop = chat.scrollHeight;
  const cr = chat ? chat.getBoundingClientRect() : null;
  const lr = last ? last.getBoundingClientRect() : null;
  const pr = panel ? panel.getBoundingClientRect() : null;
  const input = document.querySelector('.chat-input-area');
  const ir = input ? input.getBoundingClientRect() : null;
  const bubble = chat ? chat.querySelector('.chat-msg.assistant:last-of-type .chat-msg-body') : null;
  const text = bubble ? bubble.innerText : '';
  return {
    app_hidden: document.getElementById('app').classList.contains('hidden'),
    overlay_hidden: document.getElementById('init-overlay').classList.contains('hidden'),
    n_msgs: msgs.length,
    chat_scroll: chat ? {scrollHeight: chat.scrollHeight, clientHeight: chat.clientHeight, scrollTop: chat.scrollTop, bottom: Math.round(cr.bottom), panel_bottom: pr ? Math.round(pr.bottom) : null, input_top: ir ? Math.round(ir.top) : null, input_bottom: ir ? Math.round(ir.bottom) : null, window_h: window.innerHeight} : null,
    last_msg_bottom: lr ? Math.round(lr.bottom) : null,
    last_msg_visible: lr && cr ? (lr.bottom <= cr.bottom + 1 && lr.top >= cr.top - 1) : null,
    stream_leftovers: chat ? chat.querySelectorAll('.stream-text').length : 0,
    code_details: chat ? chat.querySelectorAll('details.code-details').length : 0,
    raw_fences_in_text: (text.match(/```/g) || []).length,
    pending_prompt_cards: chat ? chat.querySelectorAll('.clarification-ui').length : 0,
    turn_status: (document.getElementById('turn-status') || {}).className,
    send_hidden: document.getElementById('send-btn').classList.contains('hidden'),
    send_disabled: document.getElementById('send-btn').disabled,
    abort_hidden: document.getElementById('abort-btn').classList.contains('hidden'),
    reconnect_banner: !(document.getElementById('reconnect-banner') || {classList: {contains: () => true}}).classList.contains('hidden'),
    stale_info: (document.getElementById('stale-info') || {}).innerText,
    last_text: text.slice(0, 400),
  };
})()
"""

ANSWER_JS = r"""
(() => {
  const card = [...document.querySelectorAll('.clarification-ui')].reverse().find(c => c.querySelector('button:not([disabled]), input:not([disabled])'));
  if (!card) return null;
  const opt = card.querySelector('.clarification-option-btn:not([disabled])');
  if (opt) { opt.click(); return 'option'; }
  const pre = card.querySelectorAll('.prereq-input');
  if (pre.length) { pre.forEach(i => { i.value = %ANSWER%; }); card.querySelector('.prereq-submit').click(); return 'prerequisites'; }
  const ti = card.querySelector('.clarification-text-input');
  if (ti) { ti.value = %ANSWER%; card.querySelector('.clarification-submit').click(); return 'answer'; }
  return null;
})()
"""


async def run(a) -> int:
    chrome = find_chrome()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="stat-ui-chrome-")
    port = a.debug_port
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", f"--remote-debugging-port={port}", f"--user-data-dir={prof}", f"--window-size={a.width},{a.height}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report: list[dict] = []
    try:
        ws_url = None
        for _ in range(60):
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read())
                page = next((t for t in targets if t.get("type") == "page"), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        if not ws_url:
            print("chrome did not expose a page target")
            return 2
        async with websockets.connect(ws_url, max_size=None) as ws:
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Network.enable")
            await cdp.send("Emulation.setDeviceMetricsOverride", width=a.width, height=a.height, deviceScaleFactor=1, mobile=False)

            async def step(name: str, extra: dict | None = None):
                m = await cdp.eval(METRICS_JS)
                probs = cdp.problems()
                shot = out / f"{len(report) + 1:02d}_{name}.png"
                await cdp.screenshot(shot)
                rec = {"step": name, "metrics": m, "problems": probs, "screenshot": str(shot), **(extra or {})}
                report.append(rec)
                flags = []
                if probs:
                    flags.append(f"{len(probs)} console problem(s)")
                if m and m.get("last_msg_visible") is False:
                    flags.append("LAST MESSAGE NOT FULLY VISIBLE after scroll-to-bottom")
                if m and m.get("stream_leftovers"):
                    flags.append("stream preview left behind")
                if m and m.get("raw_fences_in_text"):
                    flags.append("raw ``` in bubble text")
                cs = (m or {}).get("chat_scroll") or {}
                if cs and cs.get("input_bottom") and cs["input_bottom"] > cs["window_h"] + 1:
                    flags.append("chat input below the viewport")
                print(f"[{name}] msgs={m and m.get('n_msgs')} status={m and m.get('turn_status')} scroll={cs} last_visible={m and m.get('last_msg_visible')} code_blocks={m and m.get('code_details')} sse={cdp.sse_summary()['sse']} " + ("; ".join(flags) if flags else "ok"))
                for p in probs:
                    print("   ", p)
                if m and m.get("last_text"):
                    print("    last bubble:", m["last_text"].replace("\n", " | ")[:300])
                return m

            await cdp.send("Page.navigate", url=a.base + "/")
            await cdp.drain(2.5)
            await cdp.wait_for("document.readyState === 'complete'", 30)
            await cdp.drain(2.0)
            m = await step("loaded")
            if not m["overlay_hidden"]:
                if not a.dataset:
                    print("init overlay shown and no --dataset given")
                    return 3
                await cdp.eval(f"""(() => {{
                    const v = (id, val) => {{ const el = document.getElementById(id); if (el) {{ el.value = val; el.dispatchEvent(new Event('input', {{bubbles: true}})); el.dispatchEvent(new Event('change', {{bubbles: true}})); }} }};
                    v('dataset-dir', {json.dumps(str(Path(a.dataset).expanduser()))});
                    v('session-name', 'ui_probe');
                    v('provider', {json.dumps(a.provider)});
                    v('model', {json.dumps(a.model)});
                    v('api-key', {json.dumps(a.api_key)});
                    document.getElementById('init-form').requestSubmit();
                    return true;
                }})()""")
                await cdp.wait_for("!document.getElementById('app').classList.contains('hidden')", a.load_timeout)
                await cdp.drain(3.0)
                await step("initialised")

            async def turn(text: str):
                n0 = await cdp.eval("document.querySelectorAll('.chat-msg.assistant').length")
                await cdp.eval(f"""(() => {{ const i = document.getElementById('chat-input'); i.value = {json.dumps(text)}; i.dispatchEvent(new Event('input', {{bubbles: true}})); document.getElementById('send-btn').click(); return true; }})()""")
                try:
                    await cdp.wait_for(f"document.querySelectorAll('.chat-msg.assistant').length > {n0}", 45)
                except TimeoutError:
                    html = await cdp.eval("document.getElementById('chat-messages').innerHTML.slice(0, 1500)")
                    print(f"[turn] no assistant bubble after 45 s for {text!r}")
                    print("    network:", json.dumps(cdp.sse_summary()))
                    for pr in cdp.problems():
                        print("   ", pr)
                    print("    chat html:", html.replace("\n", " ")[:1200])
                    return await step("turn_failed_" + "".join(ch if ch.isalnum() else "_" for ch in text[:20]).strip("_"), {"message": text, "failed": True})
                t0 = time.time()
                answered = []
                while time.time() - t0 < a.turn_timeout:
                    await cdp.drain(1.0)
                    cls = await cdp.eval("(document.getElementById('turn-status') || {}).className || ''")
                    if "idle" in cls and not await cdp.eval("document.getElementById('send-btn').classList.contains('hidden')"):
                        break
                    if "waiting" in cls or await cdp.eval("!!document.querySelector('.clarification-ui button:not([disabled])')"):
                        kind = await cdp.eval(ANSWER_JS.replace("%ANSWER%", json.dumps(a.answer)))
                        if kind:
                            answered.append(kind)
                            await cdp.drain(1.0)
                await cdp.drain(1.0)
                return await step("turn_" + "".join(ch if ch.isalnum() else "_" for ch in text[:24]).strip("_"), {"message": text, "answered": answered})

            for text in a.say:
                await turn(text)
            if a.reload:
                await cdp.send("Page.reload")
                await cdp.drain(3.0)
                await cdp.wait_for("document.readyState === 'complete' && !document.getElementById('app').classList.contains('hidden')", a.load_timeout)
                await cdp.drain(4.0)
                await step("after_reload")
            if a.show_banner:
                await cdp.eval("(() => { const b = document.getElementById('reconnect-banner'); b.textContent = 'Reconnecting (attempt 3)...'; b.classList.remove('hidden'); return true; })()")
                await cdp.drain(1.0)
                await step("with_reconnect_banner")
                await cdp.eval("document.getElementById('reconnect-banner').classList.add('hidden')")
            if a.narrow:
                await cdp.send("Emulation.setDeviceMetricsOverride", width=1100, height=700, deviceScaleFactor=1, mobile=False)
                await cdp.drain(1.5)
                await step("narrow_1100x700")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    bad = [r for r in report if r["problems"] or (r["metrics"] or {}).get("last_msg_visible") is False or (r["metrics"] or {}).get("stream_leftovers")]
    print(f"\n{len(report)} steps, {len(bad)} with problems; screenshots and report.json in {out}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8899")
    ap.add_argument("--dataset")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="scripted/model")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--out", default="ui_probe_out")
    ap.add_argument("--say", action="append", default=[])
    ap.add_argument("--answer", default="cortex", help="answer typed into any clarification card")
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--narrow", action="store_true", help="also check a 1100x700 viewport")
    ap.add_argument("--show-banner", action="store_true", help="force the reconnect banner visible and re-measure the layout")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--load-timeout", type=float, default=180)
    ap.add_argument("--turn-timeout", type=float, default=600)
    ap.add_argument("--debug-port", type=int, default=9333)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
