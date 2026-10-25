"""Archived-input mapping integration, using explicitly synthetic fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_pico_locomotion_intent import replay
from test_pico_locomotion_intent import packet


class IntentReplayTests(unittest.TestCase):
    def test_ledger_to_planar_walk_stays_diagnostic_without_network_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'commands.jsonl'
            rows = [{'event_type': 'metadata', 'metadata': {'synthetic_test': True}}]
            for i in range(61):
                frame = packet(i+1, 1.+i/50, raw_translation=[0., 0., -.3*i/50])
                rows.append({'source_sequence': i+100, 'source_raw_packet': frame.raw_payload})
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            receipt = replay(source, root/'output')
            self.assertEqual(receipt['input_packets'], 61)
            self.assertGreater(receipt['slow_walk_rows'], 30)
            self.assertEqual(receipt['invalid_rows'], 0)
            self.assertFalse(receipt['SDK_started'])
            self.assertFalse(receipt['models_loaded'])
            self.assertFalse(receipt['original_network_freshness_revalidated'])
            self.assertFalse(receipt['training_eligible'])
            outputs = [json.loads(x) for x in (root/'output/intents.jsonl').read_text().splitlines()]
            command = outputs[-1]['planner']
            self.assertEqual(command['mode'], 1)
            self.assertAlmostEqual(command['movement_direction'][0], 1.)
            self.assertAlmostEqual(command['speed'], .3, places=4)
            self.assertEqual(outputs[-1]['source_sequence'], 160)

    def test_duplicate_source_time_cannot_pad_new_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'packets.json'
            frame = packet(1, 1.)
            source.write_text(json.dumps([{'data': frame.raw_payload}, {'data': frame.raw_payload}]))
            with self.assertRaisesRegex(ValueError, 'strictly increase'):
                replay(source, root/'output')
            self.assertFalse((root/'output/receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
