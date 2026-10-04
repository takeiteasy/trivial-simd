import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

WRAPPER = pathlib.Path(__file__).with_name("bounded-run.py")


class BoundedRunTests(unittest.TestCase):
    def run_command(self, directory, code, seconds=5):
        log = pathlib.Path(directory) / "run.log"
        result = subprocess.run(
            [sys.executable, str(WRAPPER), "--seconds", str(seconds), "--log", str(log),
             "--", sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        return result, log.read_text()

    def test_exit_status_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result, log = self.run_command(directory, 'print("result", flush=True); raise SystemExit(7)')
            self.assertEqual(result.returncode, 7)
            self.assertIn("result", log)

    def test_timeout_keeps_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result, log = self.run_command(directory, 'import time; print("started", flush=True); time.sleep(5)', 0.4)
            self.assertEqual(result.returncode, 124)
            self.assertIn("started", log)
            self.assertIn("exit=124", log)

    def test_timeout_stops_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = pathlib.Path(directory) / "survived"
            child = f'import time, pathlib, signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(1); pathlib.Path({str(marker)!r}).touch()'
            code = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {child!r}]); time.sleep(5)'
            result, _ = self.run_command(directory, code, 0.4)
            self.assertEqual(result.returncode, 124)
            time.sleep(1)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
