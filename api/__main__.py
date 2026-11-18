"""``python -m api`` — start the API on all interfaces and print the address for laptop 2.

    python -m api                 # port 8000
    python -m api --port 9000
    python -m api --reload        # development only
"""
from __future__ import annotations

import argparse
import os
import socket

import uvicorn


def lan_addresses() -> list[str]:
    """Best-effort IPv4 addresses other devices on the Wi-Fi/hotspot can use."""
    found = []
    try:  # the OS picks the outbound interface; UDP connect sends no packet
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    return [ip for ip in dict.fromkeys(found) if not ip.startswith("127.")]


def main():
    ap = argparse.ArgumentParser(description="PathSnap API server")
    ap.add_argument("--host", default=os.environ.get("PATHSNAP_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PATHSNAP_PORT", "8000")))
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    print("\n  PathSnap API")
    print(f"  local:    http://localhost:{a.port}")
    for ip in lan_addresses():
        print(f"  network:  http://{ip}:{a.port}    <- enter this in the UI on laptop 2")
    print(f"  docs:     http://localhost:{a.port}/docs\n")

    # access_log off: the app writes its own one-line-per-request log (api/logs.py)
    uvicorn.run("api.main:app", host=a.host, port=a.port, reload=a.reload, access_log=False)


if __name__ == "__main__":
    main()
