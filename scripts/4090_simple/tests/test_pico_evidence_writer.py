"""Real CPU spawn tests; no SDK, simulation, renderer, GPU, or live paths."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_evidence_writer import EvidenceWriter, EvidenceWriterError, _select_cpus


class FileHandler:
    def __init__(self, directory, fail_init=False, fail_close=False):
        if fail_init:
            raise ValueError('test init failed')
        self.directory = Path(directory)
        self.fail_close = fail_close
        self.rows = []
        self.stream = (self.directory / 'events.jsonl').open('x')

    def handle(self, op, payload):
        if op == 'append':
            plain = {'array': payload['array'].tolist(), 'nested': payload['nested']}
            self.rows.append(plain)
            self.stream.write(json.dumps(plain) + '\n')
            return None
        if op == 'rows':
            return self.rows
        if op == 'noop':
            return {'discarded_async_result': payload}
        if op == 'runtime':
            import threadpoolctl
            return {'pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0)),
                    'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'omp': os.environ.get('OMP_NUM_THREADS'),
                    'native_pools': threadpoolctl.threadpool_info()}
        if op == 'block':
            (self.directory / 'entered').write_text('entered')
            while not (self.directory / 'release').exists():
                time.sleep(.002)
            return 'released'
        if op == 'sleep':
            time.sleep(payload)
            return 'slept'
        if op == 'fail':
            raise ValueError('test handler failed')
        if op == 'bad_result':
            return lambda: None
        if op == 'die':
            os._exit(17)
        raise ValueError('unknown test op')

    def close(self):
        with (self.directory / 'close_calls').open('a') as stream:
            stream.write('close\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        (self.directory / 'closed').write_text(str(len(self.rows)))
        if self.fail_close:
            raise ValueError('test close failed')
        return {'written': len(self.rows)}


def handler_factory(**kwargs):
    return FileHandler(**kwargs)


class EvidenceWriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.cpus = [min(os.sched_getaffinity(0))]
        self.writers = []

    def tearDown(self):
        for writer in self.writers:
            writer.close(timeout_s=.5)
        self.temporary.cleanup()

    def make(self, **kwargs):
        writer = EvidenceWriter(handler_factory,
            init_kwargs={'directory': str(self.directory)}, cpus=self.cpus, **kwargs)
        self.writers.append(writer)
        return writer

    def wait_until(self, predicate, timeout=3.):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('bounded test wait timed out')
            time.sleep(.003)

    @staticmethod
    def payload(number):
        return {'array': np.array([number, number + 1], dtype=np.float64),
                'nested': {'values': [number]}}

    def test_spawn_ready_and_child_cpu_policy_leave_parent_unchanged(self):
        before_cpus = os.sched_getaffinity(0)
        before_env = dict(os.environ)
        before_nice = os.getpriority(os.PRIO_PROCESS, 0)
        writer = self.make()
        runtime = writer.call('runtime')
        self.assertNotEqual(runtime['pid'], os.getpid())
        self.assertEqual(runtime['cpus'], self.cpus)
        self.assertEqual(runtime['cuda'], '')
        self.assertEqual(runtime['omp'], '1')
        self.assertTrue(all(p['num_threads'] == 1 for p in runtime['native_pools']))
        self.assertEqual(os.sched_getaffinity(0), before_cpus)
        self.assertEqual(dict(os.environ), before_env)
        self.assertEqual(os.getpriority(os.PRIO_PROCESS, 0), before_nice)
        self.assertEqual(writer.check()['worker']['pid'], runtime['pid'])
        self.assertEqual(writer.check()['worker']['nice'], 19)

    def test_parent_freezes_nested_arrays_before_feeder_and_preserves_fifo(self):
        writer = self.make(max_pending=16)
        writer.submit('block')
        self.wait_until(lambda: (self.directory / 'entered').exists())
        for number in range(8):
            payload = self.payload(number)
            writer.submit('append', payload)
            payload['array'][:] = -999
            payload['nested']['values'].append(-999)
        (self.directory / 'release').write_text('release')
        rows = writer.call('rows')
        self.assertEqual(rows, [{'array': [i, i + 1], 'nested': {'values': [i]}} for i in range(8)])
        receipt = writer.close()
        self.assertTrue(receipt['completed'])
        self.assertEqual(receipt['accepted'], receipt['acknowledged'])
        self.assertEqual(receipt['handler_close_result'], {'written': 8})
        self.assertEqual((self.directory / 'closed').read_text(), '8')

    def test_queue_full_is_immediate_sticky_and_accepted_work_still_drains(self):
        writer = self.make(max_pending=1)
        writer.submit('block')
        self.wait_until(lambda: (self.directory / 'entered').exists())
        writer.submit('append', self.payload(7))
        started = time.monotonic()
        with self.assertRaisesRegex(EvidenceWriterError, 'queue_full'):
            writer.submit('append', self.payload(8))
        self.assertLess(time.monotonic() - started, .2)
        self.assertTrue(writer.failed)
        first_error = writer.error
        (self.directory / 'release').write_text('release')
        receipt = writer.close()
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['pending'], 0)
        self.assertEqual(receipt['accepted'], 2)
        self.assertEqual(receipt['error'], first_error)
        self.assertEqual(len((self.directory / 'events.jsonl').read_text().splitlines()), 1)
        with self.assertRaises(EvidenceWriterError):
            writer.check()
        self.assertEqual(writer.close(), receipt)

    def test_serialization_error_is_sticky_without_accepting_request(self):
        writer = self.make()
        with self.assertRaisesRegex(EvidenceWriterError, 'serialization_error'):
            writer.submit('append', {'bad': lambda: None})
        self.assertTrue(writer.failed)
        with self.assertRaises(EvidenceWriterError):
            writer.call('rows')
        receipt = writer.close()
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['accepted'], 0)

    def test_handler_exception_visible_in_all_error_interfaces(self):
        writer = self.make()
        writer.submit('fail')
        self.wait_until(lambda: writer.failed)
        self.assertIn('test handler failed', writer.error)
        with self.assertRaises(EvidenceWriterError):
            writer.check()
        receipt = writer.close()
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['pending'], 1)

    def test_unpickleable_handler_result_is_not_acknowledged(self):
        writer = self.make()
        with self.assertRaises(EvidenceWriterError):
            writer.call('bad_result')
        self.assertFalse(writer.close()['completed'])
        self.assertEqual(writer.close()['acknowledged'], 0)

    def test_unexpected_child_exit_is_sticky_and_close_does_not_hang(self):
        writer = self.make()
        writer.submit('die')
        self.wait_until(lambda: writer.failed)
        self.assertIn('worker_died', writer.error)
        started = time.monotonic()
        receipt = writer.close(timeout_s=.2)
        self.assertLess(time.monotonic() - started, 1.)
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['exitcode'], 17)

    def test_close_waits_for_fifo_and_handler_flush_before_reporting_complete(self):
        writer = self.make(max_pending=8)
        for number in range(6):
            writer.submit('append', self.payload(number))
        receipt = writer.close()
        self.assertTrue(receipt['completed'])
        self.assertTrue(receipt['handler_closed'])
        self.assertEqual(receipt['pending'], 0)
        self.assertEqual(receipt['exitcode'], 0)
        self.assertEqual(len((self.directory / 'events.jsonl').read_text().splitlines()), 6)
        with self.assertRaisesRegex(EvidenceWriterError, 'closed'):
            writer.submit('append', self.payload(9))

    def test_close_timeout_terminates_only_owned_child_and_cannot_claim_completion(self):
        writer = self.make()
        writer.submit('block')
        self.wait_until(lambda: (self.directory / 'entered').exists())
        started = time.monotonic()
        receipt = writer.close(timeout_s=.05)
        self.assertLess(time.monotonic() - started, 1.)
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['pending'], 1)
        self.assertIn('close_timeout', receipt['error'])
        self.assertIsNotNone(receipt['exitcode'])

    def test_call_timeout_stays_failed_even_if_handler_later_finishes(self):
        writer = self.make()
        with self.assertRaisesRegex(EvidenceWriterError, 'request_timeout'):
            writer.call('sleep', .1, timeout_s=.01)
        receipt = writer.close(timeout_s=2.)
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['pending'], 0)
        self.assertIn('request_timeout', receipt['error'])

    def test_handler_close_failure_is_not_a_complete_archive(self):
        writer = EvidenceWriter(handler_factory,
            init_kwargs={'directory': str(self.directory), 'fail_close': True}, cpus=self.cpus)
        self.writers.append(writer)
        receipt = writer.close()
        self.assertFalse(receipt['completed'])
        self.assertIn('test close failed', receipt['error'])
        self.assertEqual((self.directory / 'close_calls').read_text(), 'close\n')

    def test_startup_failure_and_timeout_reap_spawned_worker(self):
        with self.assertRaisesRegex(EvidenceWriterError, 'test init failed'):
            EvidenceWriter(handler_factory,
                init_kwargs={'directory': str(self.directory), 'fail_init': True}, cpus=self.cpus)
        with self.assertRaisesRegex(EvidenceWriterError, 'startup_timeout'):
            EvidenceWriter(handler_factory, init_kwargs={'directory': str(self.directory)},
                cpus=self.cpus, startup_timeout_s=.00001)

    def test_dotted_factory_and_explicit_result(self):
        writer = EvidenceWriter(__name__ + ':handler_factory',
            init_kwargs={'directory': str(self.directory)}, cpus=self.cpus)
        self.writers.append(writer)
        request = writer.submit('rows', return_result=True)
        self.assertEqual(writer.result(request), [])
        with self.assertRaises(ValueError):
            writer.result(request)

    def test_cpu_validation_and_default_role(self):
        from unittest.mock import patch
        with patch('pico_evidence_writer.os.sched_getaffinity', return_value={20, 21, 22, 23, 24}):
            self.assertEqual(_select_cpus(None)[0], (21,))
            for cpus in ([], [True], [21, 21], [99], '21'):
                with self.subTest(cpus=cpus), self.assertRaises(ValueError):
                    _select_cpus(cpus)
        with patch('pico_evidence_writer.os.sched_getaffinity', return_value={24}):
            with self.assertRaises(ValueError):
                _select_cpus(None)
            self.assertEqual(_select_cpus([24])[0], (24,))

    def test_repeated_close_keeps_fds_and_helper_threads_bounded(self):
        import threading
        warm = self.directory / 'warm'
        warm.mkdir()
        writer = EvidenceWriter(handler_factory, init_kwargs={'directory': str(warm)}, cpus=self.cpus)
        self.assertTrue(writer.close()['completed'])
        baseline_fds = len(list(Path('/proc/self/fd').iterdir()))
        baseline_threads = len(threading.enumerate())
        retained = [writer]
        for i in range(6):
            path = self.directory / str(i)
            path.mkdir()
            writer = EvidenceWriter(handler_factory, init_kwargs={'directory': str(path)}, cpus=self.cpus)
            retained.append(writer)
            writer.submit('append', self.payload(i))
            self.assertTrue(writer.close()['completed'])
        self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())), baseline_fds + 1)
        self.assertLessEqual(len(threading.enumerate()), baseline_threads + 1)

    def test_deadline_survives_queued_large_requests_and_reaps_feeder(self):
        writer = self.make(max_pending=4)
        writer.submit('block')
        self.wait_until(lambda: (self.directory / 'entered').exists())
        for _ in range(3):
            writer.submit('append', {'array': np.zeros(200000), 'nested': {}})
        started = time.monotonic()
        receipt = writer.close(timeout_s=.05)
        self.assertLess(time.monotonic() - started, 1.)
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['pending'], 4)
        self.assertFalse(writer._requests._thread.is_alive())
        self.assertFalse(writer._reply_thread.is_alive())

    def test_oversized_request_is_sticky_rejected_without_clipping_or_acceptance(self):
        writer = self.make(max_request_bytes=512)
        with self.assertRaisesRegex(EvidenceWriterError, 'payload_too_large'):
            writer.submit('append', {'array': np.zeros(1000), 'nested': {}})
        receipt = writer.close()
        self.assertFalse(receipt['completed'])
        self.assertEqual(receipt['accepted'], 0)
        self.assertGreater(receipt['fault']['encoded_bytes'], 512)
        self.assertEqual((self.directory / 'events.jsonl').read_bytes(), b'')

    def test_long_async_stream_does_not_retain_return_values(self):
        writer = self.make(max_pending=64)
        for batch in range(20):
            for item in range(25):
                writer.submit('noop', {'batch': batch, 'item': item})
            self.assertEqual(writer.call('rows'), [])
            self.assertEqual(writer._results, {})
            self.assertEqual(writer._result_expected, set())
        receipt = writer.close()
        self.assertTrue(receipt['completed'])
        self.assertEqual(receipt['accepted'], 520)
        self.assertEqual(receipt['pending'], 0)


if __name__ == '__main__':
    unittest.main()
