"""Create a separate SmolVLA dataset with a selected action exposed as `action`."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pyarrow.parquet as pq

ACTION_KEYS = ('action', 'action_tcp', 'action_tcp_base', 'action_joints')


def select_action_key(features, requested=None):
    key = requested or ('action' if 'action' in features else 'action_tcp')
    if key not in ACTION_KEYS or key not in features:
        raise ValueError(f'找不到 action 欄位 {key}；可選欄位：{[k for k in ACTION_KEYS if k in features]}')
    return key


def prepare(root, action_key=None):
    root = Path(root).expanduser().resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    key = select_action_key(info['features'], action_key)
    if key == 'action':
        return root
    state_key = ('observation.tcp_pose' if key in ('action_tcp', 'action_tcp_base')
                 and 'observation.tcp_pose' in info['features'] else 'observation.state')
    files = sorted((root / 'meta').rglob('*')) + sorted((root / 'data').rglob('*.parquet'))
    signature = hashlib.sha256(json.dumps([
        (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in files if p.is_file()
    ]).encode()).hexdigest()
    marker = {'version': 3, 'source_root': str(root), 'source_signature': signature,
              'action_source': key, 'state_source': state_key}
    output = root.with_name(f'{root.name}_smolvla_{key}')
    if output.exists():
        if not (output / 'meta/training_view.json').is_file() or json.loads((output / 'meta/training_view.json').read_text()) != marker:
            raise ValueError(f'訓練資料目錄已存在且來源不一致：{output}；請移走該目錄後重新產生。')
        return output
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=root.parent))
    try:
        shutil.copytree(root / 'meta', staging / 'meta')
        for source in sorted((root / 'data').rglob('*.parquet')):
            table = pq.read_table(source)
            if state_key != 'observation.state':
                table = table.set_column(table.schema.get_field_index('observation.state'),
                                         'observation.state', table[state_key])
            selected = table[key]
            table = table.drop([column for column in table.column_names if column.startswith('action')])
            table = table.append_column('action', selected)
            # Old HuggingFace schema metadata does not describe the added alias.
            table = table.replace_schema_metadata(None)
            dest = staging / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
        # Hard links share video bytes without another encode or copy.
        for source in sorted((root / 'videos').rglob('*')):
            if source.is_file():
                dest = staging / source.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, dest)
                except OSError:
                    shutil.copy2(source, dest)
        info['features']['observation.state'] = info['features'][state_key].copy()
        selected_feature = info['features'][key].copy()
        info['features'] = {k: v for k, v in info['features'].items() if not k.startswith('action')}
        info['features']['action'] = selected_feature
        (staging / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
        stats_path = staging / 'meta/stats.json'
        stats = json.loads(stats_path.read_text())
        stats['observation.state'] = stats[state_key]
        selected_stats = stats[key]
        stats = {k: v for k, v in stats.items() if not k.startswith('action')}
        stats['action'] = selected_stats
        stats_path.write_text(json.dumps(stats, indent=2) + '\n')
        for dest in (staging / 'meta/episodes').rglob('*.parquet'):
            table = pq.read_table(dest)
            if state_key != 'observation.state':
                for column in list(table.column_names):
                    if column.startswith(f'stats/{state_key}/'):
                        alias = column.replace(f'stats/{state_key}/', 'stats/observation.state/', 1)
                        if alias in table.column_names:
                            table = table.drop([alias])
                        table = table.append_column(alias, table[column])
            for column in list(table.column_names):
                if column.startswith(f'stats/{key}/'):
                    alias = column.replace(f'stats/{key}/', 'stats/action/', 1)
                    if alias in table.column_names:
                        table = table.drop([alias])
                    table = table.append_column(alias, table[column])
            table = table.drop([column for column in table.column_names
                                if column.startswith('stats/action') and not column.startswith('stats/action/')])
            pq.write_table(table.replace_schema_metadata(None), dest)
        modality = staging / 'meta/modality.json'
        if modality.exists():
            payload = json.loads(modality.read_text())
            if state_key == 'observation.tcp_pose':
                payload['state'] = {
                    'eef_position': {'original_key': 'observation.state', 'start': 0, 'end': 3},
                    'eef_rotation': {'original_key': 'observation.state', 'start': 3, 'end': 6, 'rotation_type': 'axis_angle'},
                    'gripper_position': {'original_key': 'observation.state', 'start': 6, 'end': 7},
                }
            payload['action'] = {
                'selected_action': {'original_key': 'action', 'source_key': key,
                                    'start': 0, 'end': 7, 'action_space': key},
            }
            modality.write_text(json.dumps(payload, indent=2) + '\n')
        # Source quality/provenance are retained; source verification is not a view audit.
        (staging / 'meta/verification.json').unlink(missing_ok=True)
        (staging / 'meta/training_view.json').write_text(json.dumps(marker, indent=2) + '\n')
        if output.exists():
            raise ValueError(f'訓練資料目錄在產生期間已出現：{output}')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--action-key', choices=ACTION_KEYS)
    args = parser.parse_args()
    print(prepare(args.root, args.action_key))
