#!/usr/bin/env bash
# Real Federated Learning launcher (Linux/macOS)
# 1 server + 3 clients as separate processes
cd "$(dirname "$0")"
export FL_ROUNDS=5
export FL_CLIENTS=3

python server.py &
SERVER_PID=$!
sleep 5
python client.py 0 &
python client.py 1 &
python client.py 2 &
wait $SERVER_PID
echo "Federated run finished."
