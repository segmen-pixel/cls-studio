# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Shared startup progress state and the boot loading page.

Extracted from main.py during the pre-OSS refactor so middleware,
endpoints and startup tasks can share the state without importing
app.main.
"""
from __future__ import annotations

from typing import Any

startup_state: dict[str, Any] = {"ready": False, "steps": [], "current": "", "warnings": []}

# Light palette matching the UI's light theme (the old navy card made the
# very first screen of the product dark regardless of the app theme), with
# the colourblind-safe accents used across the app (teal / Okabe-Ito green).
# The inline SVG favicon shows an "A" so the tab is branded before the dist
# favicon is even reachable.
LOADING_HTML = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cls-studio - Starting...</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%2322C7DB'/%3E%3Ctext x='32' y='46' font-family='Arial,Helvetica,sans-serif' font-size='40' font-weight='bold' fill='%23fff' text-anchor='middle'%3EA%3C/text%3E%3C/svg%3E">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#f2f2f7;color:#1d2129;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#ffffff;border:1px solid #e3e6ea;border-radius:12px;padding:40px 48px;max-width:420px;width:90%;
  box-shadow:0 8px 28px rgba(30,40,60,.12);text-align:center}
h1{font-size:20px;font-weight:600;margin-bottom:4px;color:#1a1e28}
.sub{font-size:12px;color:#6b7684;margin-bottom:24px}
.spinner{width:36px;height:36px;border:3px solid #dde3ea;border-top-color:#22C7DB;
  border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 20px}
@keyframes spin{to{transform:rotate(360deg)}}
.current{font-size:13px;color:#0f98ac;min-height:20px;margin-bottom:16px}
.steps{text-align:left;font-size:12px;color:#6b7684;line-height:1.8}
.steps .done{color:#009E73}
.steps .done::before{content:"\\2714 ";color:#009E73}
.error{color:#d33;font-size:13px;margin-top:12px}
</style>
</head>
<body>
<div class="card">
  <h1>cls-studio</h1>
  <div class="sub">See Every Pixel, From Pixel</div>
  <div class="spinner" id="sp"></div>
  <div class="current" id="cur">Connecting...</div>
  <div class="steps" id="steps"></div>
  <div class="error" id="err"></div>
</div>
<script>
(function(){
  var iv=setInterval(function(){
    fetch("/startup-status").then(function(r){return r.json()}).then(function(d){
      var el=document.getElementById("steps");
      el.innerHTML="";d.steps.forEach(function(s){var div=document.createElement("div");div.className="done";div.textContent=s;el.appendChild(div)});
      document.getElementById("cur").textContent=d.current||"";
      if(d.error) document.getElementById("err").textContent=d.error;
      if(d.ready){
        clearInterval(iv);
        document.getElementById("sp").style.borderTopColor="#009E73";
        document.getElementById("cur").textContent="Startup complete";
        setTimeout(function(){location.href="/ui/"},600);
      }
    }).catch(function(){
      document.getElementById("cur").textContent="Connecting to server...";
    });
  },800);
})();
</script>
</body>
</html>
"""
