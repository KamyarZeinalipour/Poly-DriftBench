#!/bin/bash
# Start annotation server + cloudflare tunnel
cd /home4/kamyar/long_contex/annotation_ui

echo "Starting gunicorn on port 5555..."
/home4/kamyar/long_contex/venv/bin/gunicorn -b 0.0.0.0:5555 -w 2 --timeout 120 app:app &
GUNICORN_PID=$!
sleep 3

echo "Starting Cloudflare tunnel..."
/tmp/cloudflared tunnel --url http://localhost:5555 2>&1 &
CF_PID=$!

wait $GUNICORN_PID $CF_PID
