"""Fast tests for mcp_server.py's own plumbing - no model, no subprocess.
tests/test_mcp_server.py covers the tools end to end against the real model
(marked slow)."""

import threading
import time

from supervisor import mcp_server


def test_concurrent_first_calls_load_the_judge_once(monkeypatch):
    # The MCP SDK runs sync tools on worker threads, so the first few tool
    # calls can all find no judge loaded yet. Each load is several GB.
    loads = []

    def fake_calibrated():
        loads.append(object())
        time.sleep(0.02)  # a load is slow; widen the race window
        return loads[-1]

    monkeypatch.setattr(mcp_server, "_judge", None)
    monkeypatch.setattr(mcp_server.Judge, "calibrated", staticmethod(fake_calibrated))

    barrier = threading.Barrier(8)
    got = []

    def call():
        barrier.wait()
        got.append(mcp_server._get_judge())

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loads) == 1
    assert all(judge is loads[0] for judge in got)
