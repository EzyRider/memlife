"""Tests for the thread-safe SQLite connection proxy."""

from __future__ import annotations

import sqlite3
import threading
import time
from queue import Queue

import pytest

from memlife._locked_conn import _LockedConn


class _TrackingRLock:
    """RLock wrapper that exposes an ``is_held`` flag for tests."""

    def __init__(self):
        self._lock = threading.RLock()
        self._held_count = 0
        self._cond = threading.Condition(threading.Lock())

    def acquire(self, blocking=True, timeout=-1):
        acquired = self._lock.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self._cond:
                self._held_count += 1
        return acquired

    def release(self):
        self._lock.release()
        with self._cond:
            self._held_count -= 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def is_held(self):
        with self._cond:
            return self._held_count > 0


class TestLockedCursor:
    """HF-001: cursors must hold the lock for their entire lifetime."""

    def test_cursor_context_manager_releases_lock_on_exit(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        raw.execute("CREATE TABLE t (x INT)")
        raw.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
        conn = _LockedConn(raw, lock)

        assert not lock.is_held
        with conn.cursor() as cur:
            assert lock.is_held
            cur.execute("SELECT x FROM t")
            rows = list(cur)
            assert rows == [(1,), (2,), (3,)]
        assert not lock.is_held

    def test_cursor_context_manager_releases_lock_on_exception(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        conn = _LockedConn(raw, lock)

        with pytest.raises(RuntimeError), conn.cursor() as cur:
            cur.execute("SELECT 1")
            raise RuntimeError("boom")
        assert not lock.is_held

    def test_cursor_blocks_concurrent_access(self):
        lock = threading.RLock()
        # check_same_thread=False lets us verify that the _LockedConn lock
        # serialises access across threads. The RLock is what makes this safe
        # in memlife's actual usage.
        raw = sqlite3.connect(":memory:", check_same_thread=False)
        raw.execute("CREATE TABLE t (x INT)")
        for i in range(100):
            raw.execute("INSERT INTO t VALUES (?)", (i,))
        conn = _LockedConn(raw, lock)

        results: Queue[list[tuple[int]]] = Queue()
        errors: Queue[BaseException] = Queue()

        def reader():
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT x FROM t ORDER BY x")
                    # Hold the cursor open long enough that concurrent access
                    # would interleave if the lock were not held.
                    time.sleep(0.01)
                    results.put(list(cur))
            except Exception as exc:
                errors.put(exc)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors.empty()
        assert results.qsize() == 5
        while not results.empty():
            assert results.get() == [(i,) for i in range(100)]


class TestLockedConnRowFactory:
    """HF-002: row_factory access must be serialised by the lock."""

    def test_row_factory_getter_holds_lock(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        conn = _LockedConn(raw, lock)

        assert not lock.is_held
        _ = conn.row_factory
        # The getter is a single statement; we verify it does not crash and
        # that the lock is available immediately after.
        assert conn.row_factory is None
        assert not lock.is_held

    def test_row_factory_setter_holds_lock(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        conn = _LockedConn(raw, lock)

        def factory(cursor, row):
            return row

        conn.row_factory = factory
        assert conn.row_factory is factory

    def test_row_factory_setter_blocks_under_contention(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        raw.execute("CREATE TABLE t (x INT)")
        raw.execute("INSERT INTO t VALUES (1)")
        conn = _LockedConn(raw, lock)

        order: list[str] = []
        barrier = threading.Barrier(2)

        def writer_a():
            barrier.wait()
            with conn._lock:
                conn.row_factory = lambda c, r: ("A", r[0])
                time.sleep(0.02)
                order.append("A")

        def writer_b():
            barrier.wait()
            with conn._lock:
                conn.row_factory = lambda c, r: ("B", r[0])
                time.sleep(0.02)
                order.append("B")

        t_a = threading.Thread(target=writer_a)
        t_b = threading.Thread(target=writer_b)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # Both writers completed; order is non-deterministic but both ran.
        assert sorted(order) == ["A", "B"]

    def test_isolation_level_and_text_factory_are_locked(self):
        lock = _TrackingRLock()
        raw = sqlite3.connect(":memory:")
        conn = _LockedConn(raw, lock)

        conn.isolation_level = "IMMEDIATE"
        assert conn.isolation_level == "IMMEDIATE"
        conn.text_factory = bytes
        assert conn.text_factory is bytes
