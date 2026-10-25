"""Isolation and asset provenance guards, without inference or physics."""
from pathlib import Path
import hashlib
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_pico_locomotion as diagnostic


class DiagnosticGuards(unittest.TestCase):
    def test_model_binds_included_robot_and_its_relative_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'robots/meshes').mkdir(parents=True)
            scene = root/'scene.xml'
            robot = root/'robots/g1.xml'
            mesh = root/'robots/meshes/foot.stl'
            scene.write_text('<mujoco><include file="robots/g1.xml"/></mujoco>')
            robot.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="foot.stl"/></asset></mujoco>')
            mesh.write_bytes(b'mesh')
            self.assertEqual(set(diagnostic.model_input_files(scene)), {scene, robot, mesh})
            mesh.unlink()
            with self.assertRaises(FileNotFoundError):
                diagnostic.model_input_files(scene)

    def test_walk_has_explicit_idle_before_and_after(self):
        for tick in (0, 99, 400, 499):
            command = diagnostic.fixed_command(tick, (0., -1., 0.), .3)
            self.assertEqual(command['mode'], 0)
            self.assertEqual(command['speed'], -1.)
            self.assertEqual(command['movement_direction'], [0., 0., 0.])
        for tick in (100, 399):
            command = diagnostic.fixed_command(tick, (0., -1., 0.), .3)
            self.assertEqual(command['mode'], 1)
            self.assertEqual(command['movement_direction'], [0., -1., 0.])

    def test_archived_intent_is_causal_and_a_gap_cannot_continue_old_walk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'intents.jsonl'
            moving = diagnostic.fixed_command(100, (1., 0., 0.), .3)
            rows = [{'elapsed_device_s': t, 'source_sequence': i+1,
                     'device_timestamp_ns': int((1+t)*1e9), 'planner': moving,
                     'diagnostic_only': True, 'training_eligible': False}
                    for i, t in enumerate((.02, .04, .4))]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            receipt = {'contract': 'cvpr-pico-root-planner-offline-replay-v1',
                       'intents_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'training_eligible': False}
            (root/'receipt.json').write_text(json.dumps(receipt))
            source = diagnostic.RecordedIntent(path)
            actual = [0., 0., .78, math.cos(math.pi/4), 0., 0., math.sin(math.pi/4)]
            self.assertEqual(source.at(.01, actual)[0]['mode'], 0)
            self.assertEqual(source.at(.03, actual)[1]['source_sequence'], 1)
            self.assertEqual(source.at(.1, actual)[0]['mode'], 1)
            self.assertEqual(source.at(.3, actual)[0]['mode'], 0)
            gap = source.at(.3, actual)[0]
            self.assertAlmostEqual(gap['facing_direction'][0], 0.)
            self.assertAlmostEqual(gap['facing_direction'][1], 1.)
            self.assertEqual(source.at(.41, actual)[1]['source_sequence'], 3)
            self.assertEqual(source.at(.7, actual)[0]['mode'], 0)
            path.write_text(path.read_text()+'{}\n')
            with self.assertRaisesRegex(ValueError, 'Bound diagnostic'):
                diagnostic.RecordedIntent(path)


if __name__ == '__main__':
    unittest.main()
