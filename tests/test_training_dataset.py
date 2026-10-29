import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_smolvla_dataset import prepare, select_action_key


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        for folder in ['meta/episodes/chunk-000', 'data/chunk-000', 'videos/camera/chunk-000']:
            (self.root / folder).mkdir(parents=True)
        features = {k: {'dtype': 'float32', 'shape': [7], 'names': [str(i) for i in range(7)]}
                    for k in ['observation.state', 'action_tcp', 'action_tcp_base', 'action_joints']}
        (self.root / 'meta/info.json').write_text(json.dumps({'features': features}))
        stats = {k: {'mean': [n] * 7} for n, k in enumerate(features)}
        (self.root / 'meta/stats.json').write_text(json.dumps(stats))
        table = pa.table({k: pa.array([[n] * 7, [n+1] * 7], type=pa.list_(pa.float32(), 7))
                          for n, k in enumerate(features)})
        pq.write_table(table, self.root / 'data/chunk-000/episode_000000.parquet')
        pq.write_table(pa.table({'stats/action_tcp/mean': [[1.] * 7], 'stats/action_tcp_base/mean': [[2.] * 7]}),
                       self.root / 'meta/episodes/chunk-000/file-000.parquet')
        (self.root / 'videos/camera/chunk-000/file-000.mp4').write_bytes(b'video-fixture')

    def test_alias_and_stats_preserve_source_and_state(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        dest = prepare(self.root)
        table = pq.read_table(dest / 'data/chunk-000/episode_000000.parquet')
        self.assertEqual(table['action'], pq.read_table(self.root / 'data/chunk-000/episode_000000.parquet')['action_tcp'])
        self.assertEqual([k for k in table.column_names if k.startswith('action')], ['action'])
        self.assertEqual(table['observation.state'], pq.read_table(self.root / 'data/chunk-000/episode_000000.parquet')['observation.state'])
        stats = json.loads((dest / 'meta/stats.json').read_text())
        self.assertEqual(stats['action'], json.loads((self.root / 'meta/stats.json').read_text())['action_tcp'])
        ep = pq.read_table(dest / 'meta/episodes/chunk-000/file-000.parquet')
        self.assertEqual(ep['stats/action/mean'].to_pylist(), [[1.] * 7])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        video = Path('videos/camera/chunk-000/file-000.mp4')
        self.assertEqual((dest / video).stat().st_ino, (self.root / video).stat().st_ino)
        self.assertEqual(prepare(self.root), dest)

    def test_tcp_state_data_schema_and_stats_are_mapped_together(self):
        info_path = self.root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        info['features']['observation.tcp_pose'] = {'dtype': 'float32', 'shape': [7], 'names': ['x','y','z','rx','ry','rz','gripper']}
        info_path.write_text(json.dumps(info))
        data_path = self.root / 'data/chunk-000/episode_000000.parquet'
        table = pq.read_table(data_path)
        tcp = pa.array([[10.] * 7, [11.] * 7], type=pa.list_(pa.float32(), 7))
        pq.write_table(table.append_column('observation.tcp_pose', tcp), data_path)
        stats_path = self.root / 'meta/stats.json'
        stats = json.loads(stats_path.read_text())
        stats['observation.tcp_pose'] = {'mean': [10.5] * 7}
        stats_path.write_text(json.dumps(stats))
        ep_path = self.root / 'meta/episodes/chunk-000/file-000.parquet'
        ep = pq.read_table(ep_path).append_column('stats/observation.state/mean', pa.array([[0.] * 7]))
        ep = ep.append_column('stats/observation.tcp_pose/mean', pa.array([[10.5] * 7]))
        pq.write_table(ep, ep_path)
        dest = prepare(self.root)
        result = pq.read_table(dest / data_path.relative_to(self.root))
        self.assertEqual(result['observation.state'].to_pylist(), tcp.to_pylist())
        self.assertEqual(json.loads((dest / 'meta/info.json').read_text())['features']['observation.state'], info['features']['observation.tcp_pose'])
        self.assertEqual(json.loads((dest / 'meta/stats.json').read_text())['observation.state'], stats['observation.tcp_pose'])
        self.assertEqual(pq.read_table(dest / ep_path.relative_to(self.root))['stats/observation.state/mean'].to_pylist(), [[10.5] * 7])
        self.assertEqual(json.loads((dest / 'meta/training_view.json').read_text())['state_source'], 'observation.tcp_pose')
        joint_dest = prepare(self.root, 'action_joints')
        self.assertEqual(pq.read_table(joint_dest / data_path.relative_to(self.root))['observation.state'], table['observation.state'])

    def test_explicit_base_action_and_stale_source_rejected(self):
        dest = prepare(self.root, 'action_tcp_base')
        t = pq.read_table(dest / 'data/chunk-000/episode_000000.parquet')
        self.assertEqual(t['action'], pq.read_table(self.root / 'data/chunk-000/episode_000000.parquet')['action_tcp_base'])
        (self.root / 'meta/stats.json').write_text('{}')
        with self.assertRaises(ValueError):
            prepare(self.root, 'action_tcp_base')

    def test_train_wrapper_uses_same_view_for_preflight_and_training(self):
        project = Path(self.tmp.name) / 'project'
        project.mkdir()
        repository = Path(__file__).resolve().parents[1]
        (project / 'train.sh').write_text((repository / 'train.sh').read_text())
        (project / 'prepare_smolvla_dataset.py').symlink_to(repository / 'prepare_smolvla_dataset.py')
        bindir = project / '.venv/bin'
        bindir.mkdir(parents=True)
        python = bindir / 'python'
        python.write_text('#!/usr/bin/env bash\n' +
                          'if [[ "$1" == */check_dataset.py ]]; then\n' +
                          '  printf "%s\\n" "$@" > "$PREFLIGHT_ARGS"\n' +
                          'else\n' + f'  exec "{repository}/.venv/bin/python" "$@"\n' +
                          'fi\n')
        trainer = bindir / 'lerobot-train'
        trainer.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TRAINER_ARGS"\n')
        python.chmod(0o755)
        trainer.chmod(0o755)
        env = dict(os.environ, DATASET_ROOT=str(self.root), ACTION_KEY='action_tcp_base',
                   PREFLIGHT_ARGS=str(project / 'preflight_args'), TRAINER_ARGS=str(project / 'trainer_args'))
        result = subprocess.run(['bash', str(project / 'train.sh'), 'local/test'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = f'--dataset.root={self.root}_smolvla_action_tcp_base'
        for file in ['preflight_args', 'trainer_args']:
            roots = [line for line in (project / file).read_text().splitlines() if line.startswith('--dataset.root=')]
            self.assertEqual(roots[-1], expected)

    def test_legacy_action_is_preferred_and_missing_selection_fails(self):
        self.assertEqual(select_action_key({'action': {}, 'action_tcp': {}}), 'action')
        with self.assertRaises(ValueError):
            select_action_key({'action_tcp': {}}, 'action_joints')
        with self.assertRaises(ValueError):
            select_action_key({})


if __name__ == '__main__':
    unittest.main()
