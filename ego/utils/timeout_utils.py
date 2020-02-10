#!/usr/bin/env python
"""Provides scikit interface."""

import signal


def assign_timeout(func, timeout):
    """assign_timeout."""
    def handler(signum, frame):
        raise Exception("end of timeout")

    def timed_func(*args, **kargs):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            r = func(*args, **kargs)
            return r
        except Exception:
            return None
    return timed_func
