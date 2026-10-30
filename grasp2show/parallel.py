"""Process pools for long runs."""
from __future__ import annotations

import time
from multiprocessing import Pool


class Workers:
    """A process pool that survives a worker dying. With multiprocessing.Pool, a worker that
    crashes (on Windows, silently) is replaced but its job never returns, and imap waits forever
    (the first run stopped at 1337/1340 teacher jobs this way). Here results come back as they
    finish; if nothing finishes for `stall` seconds, the jobs still out are given up (returned
    with result None) and the pool is restarted."""

    def __init__(self, n: int, stall: float = 600.0):
        self.n, self.stall = n, stall
        self.pool = Pool(n)

    def run(self, fn, jobs: list):
        """Yields (job, result or None), in the order they finish."""
        pending = {i: self.pool.apply_async(fn, (j,)) for i, j in enumerate(jobs)}
        last = time.time()
        while pending:
            done = [i for i, a in pending.items() if a.ready()]
            for i in done:
                a = pending.pop(i)
                try:
                    res = a.get()
                except Exception:  # noqa: BLE001 (an error inside the job: give it up, keep going)
                    res = None
                yield jobs[i], res
                last = time.time()
            if not done:
                if time.time() - last > self.stall:
                    for i in sorted(pending):
                        yield jobs[i], None
                    self.restart()
                    return
                time.sleep(0.2)

    def restart(self) -> None:
        self.pool.terminate()
        self.pool.join()
        self.pool = Pool(self.n)

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()
