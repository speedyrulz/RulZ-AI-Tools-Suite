@echo off
rem Start the WebUI and serve it to your Tailscale devices: https://<this machine>.<tailnet>.ts.net:7870
call "%~dp0run_webui.bat" 7870 tailscale
