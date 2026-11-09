"""The external observer binds actual exit, report and immutable driver bytes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pick_rtx as driver


class ObservedExitTests(unittest.TestCase):
    def test_success_and_failure_keep_actual_exit_and_driver_snapshot(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report, log = root/'report.json', root/'render.log'
                report_value = {'status': 'render_completed_pending_process_exit'}

                class Child:
                    pid = 123456
                    def wait(self):
                        report.write_text(json.dumps(report_value))
                        return exit_code

                argv = ['run_pick_rtx.py', '--python', '/environment/python',
                    '--manifest', str(root/'frozen.json'), '--data-root', str(root/'data'),
                    '--report', str(report), '--log', str(log), '--profile', 'bend_pick',
                    '--offset', '2', '--limit', '3']
                original_open = Path.open
                def redirected_open(path, *args, **kwargs):
                    if str(path) == '/home/zxc/CVPR2027/log.md':
                        return original_open(root/'project.log', *args, **kwargs)
                    return original_open(path, *args, **kwargs)

                with patch.object(sys, 'argv', argv), patch.object(driver, 'require_idle'), \
                        patch.object(driver.subprocess, 'Popen', return_value=Child()), \
                        patch.object(Path, 'open', redirected_open):
                    if exit_code:
                        with self.assertRaises(SystemExit) as error:
                            driver.main()
                        self.assertEqual(error.exception.code, exit_code)
                    else:
                        driver.main()
                    receipt = json.loads((root/'report_EXIT.json').read_text())
                    self.assertEqual(receipt['exit_code'], exit_code)
                    self.assertEqual(receipt['report_sha256'], driver.sha(report))
                    snapshot = Path(receipt['parentdriver_path'])
                    self.assertEqual(snapshot.read_bytes(), Path(driver.__file__).read_bytes())
                    self.assertEqual(driver.sha(snapshot), receipt['parentdriver_sha256'])
                    # A second invocation cannot replace a previous report or log.
                    with self.assertRaisesRegex(ValueError, 'Preserve previous'):
                        driver.main()


if __name__ == '__main__':
    unittest.main()
