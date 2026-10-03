"""The tensor-copy pool preserves defaults and enforces per-rank limits."""

import argparse
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.validation_hook import (
    validate_weight_loader_copy_num_threads,
)
from sglang.srt.model_loader.utils import create_weight_loader_executor
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWeightLoaderExecutor(CustomTestCase):
    def _config(self, workers):
        return patch(
            "sglang.srt.model_loader.utils.get_model",
            return_value=SimpleNamespace(weight_loader_copy_num_threads=workers),
        )

    def test_default_preserves_python_sizing(self):
        with self._config(None), create_weight_loader_executor() as executor:
            with ThreadPoolExecutor() as reference:
                self.assertEqual(executor._max_workers, reference._max_workers)
            self.assertEqual(executor.submit(lambda: 42).result(timeout=10), 42)

    def test_copy_concurrency_and_completion(self):
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                self._check_copy_concurrency(workers)

    def _check_copy_concurrency(self, workers):
        release = threading.Event()
        ready = threading.Event()
        lock = threading.Lock()
        active = peak = 0
        caller = threading.get_ident()
        worker_ids = set()

        def copy(value):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                worker_ids.add(threading.get_ident())
                if active == workers:
                    ready.set()
            try:
                if not release.wait(timeout=10):
                    raise TimeoutError("copy workers were not released")
                return value * 2
            finally:
                with lock:
                    active -= 1

        with self._config(workers), create_weight_loader_executor() as executor:
            futures = [executor.submit(copy, i) for i in range(12)]
            try:
                self.assertTrue(ready.wait(timeout=10))
            finally:
                release.set()

        # Exiting the context waits for every queued copy.
        self.assertTrue(all(f.done() for f in futures))
        self.assertEqual([f.result() for f in futures], list(range(0, 24, 2)))
        self.assertEqual(peak, workers)
        self.assertNotIn(caller, worker_ids)

    def test_worker_exception_is_not_swallowed(self):
        def failing_copy():
            raise RuntimeError("copy failed")

        with self._config(2), create_weight_loader_executor() as executor:
            future = executor.submit(failing_copy)
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                future.result(timeout=10)

    def test_validate_worker_count(self):
        for value in (None, 1, 2, 32):
            with self.subTest(value=value):
                validate_weight_loader_copy_num_threads(value)
        for value in (0, -1, True, False, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_weight_loader_copy_num_threads(value)

    def test_cli_round_trip_and_validation(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for value in (None, 1, 2):
            argv = ["--model-path", "dummy"]
            if value is not None:
                argv += ["--weight-loader-copy-num-threads", str(value)]
            args = ServerArgs.from_cli_args(parser.parse_args(argv))
            self.assertEqual(args.weight_loader_copy_num_threads, value)
        for value in (0, -1):
            args = ServerArgs(model_path="dummy", weight_loader_copy_num_threads=value)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                args.check_server_args()


if __name__ == "__main__":
    unittest.main()
