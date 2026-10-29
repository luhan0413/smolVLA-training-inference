"""Merge LeRobot v3 TCP datasets and shuffle whole episodes without video re-encoding."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA = 'observation.images.arm'
POSE_NAMES = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
ACTION_NAMES = [f'tcp_local_delta_{k}' for k in POSE_NAMES[:6]] + ['gripper']


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def stats(values):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    values = values.astype(np.float64)
    result = {name: getattr(values, name)(axis=0).tolist() for name in ['min', 'max', 'mean', 'std']}
    result['count'] = [len(values)]
    for q in [1, 10, 50, 90, 99]:
        result[f'q{q:02d}'] = np.quantile(values, q / 100, axis=0).tolist()
    return result


def read_episodes(root):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))]).to_pylist()


def merge(sources, output, seed=42):
    sources = [Path(p).resolve() for p in sources]
    output = Path(output).resolve()
    if output.exists() or any(output == p or p in output.parents for p in sources):
        raise ValueError('Output must be new and outside all source datasets')
    records, datasets, task_ids = [], [], {}
    for source_id, root in enumerate(sources):
        info = json.loads((root / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v3.0' or info['splits'] != {'train': f"0:{info['total_episodes']}"}:
            raise ValueError('Require v3.0 and a full train split')
        features = info['features']
        cameras = [k for k, f in features.items() if f['dtype'] == 'video']
        if len(cameras) != 1:
            raise ValueError('Require exactly one video camera per source')
        camera = cameras[0]
        if features['observation.tcp_pose']['names'] != POSE_NAMES or features['action']['names'] != ACTION_NAMES:
            raise ValueError('Require explicit TCP pose and local TCP action names')
        canonical = copy.deepcopy(features)
        canonical['observation.state'] = copy.deepcopy(features['observation.tcp_pose'])
        canonical[CAMERA] = canonical.pop(camera)
        if datasets:
            first = datasets[0]['info']
            if info['fps'] != first['fps'] or set(canonical) != set(datasets[0]['features']):
                raise ValueError('FPS or feature keys differ')
            for key, feature in canonical.items():
                other = datasets[0]['features'][key]
                if any(feature.get(k) != other.get(k) for k in ['dtype', 'shape', 'names']):
                    raise ValueError(f'Incompatible feature: {key}')
            for key in ['video.codec', 'video.pix_fmt', 'video.fps']:
                if canonical[CAMERA]['info'][key] != datasets[0]['features'][CAMERA]['info'][key]:
                    raise ValueError(f'Incompatible video encoding: {key}')
        tasks = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
        mapping = {}
        for text, row in tasks.iterrows():
            mapping[int(row.task_index)] = task_ids.setdefault(str(text), len(task_ids))
        episodes = sorted(read_episodes(root), key=lambda e: e['episode_index'])
        if [e['episode_index'] for e in episodes] != list(range(info['total_episodes'])):
            raise ValueError('Invalid episode indices')
        datasets.append(dict(root=root, info=info, camera=camera, features=canonical, tasks=mapping))
        records.extend((source_id, e) for e in episodes)
    robot_type = '_'.join(sorted({ds['info']['robot_type'] for ds in datasets}))
    random.Random(seed).shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        (stage / 'meta').mkdir()
        info = copy.deepcopy(datasets[0]['info'])
        info.update(features=datasets[0]['features'], robot_type=robot_type, total_episodes=len(records),
                    total_frames=sum(e['length'] for _, e in records), total_tasks=len(task_ids),
                    splits={'train': f'0:{len(records)}'},
                    data_path='data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet',
                    video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        video_files, cached_data, rows, tables, provenance = {}, {}, [], [], []
        offset = 0
        for episode_id, (source_id, source_ep) in enumerate(records):
            ds = datasets[source_id]
            root, original, camera = ds['root'], ds['info'], ds['camera']
            path = root / original['data_path'].format(chunk_index=source_ep['data/chunk_index'], file_index=source_ep['data/file_index'])
            if path not in cached_data:
                cached_data[path] = pq.read_table(path)
            table = cached_data[path]
            mask = pa.compute.equal(table['episode_index'], source_ep['episode_index'])
            table = table.filter(mask)
            if table.num_rows != source_ep['length'] or table['frame_index'].to_pylist() != list(range(table.num_rows)):
                raise ValueError('Invalid source episode length or frame order')
            table = table.set_column(table.schema.get_field_index('observation.state'), 'observation.state', table['observation.tcp_pose'])
            values = {'episode_index': [episode_id] * len(table), 'index': list(range(offset, offset + len(table))),
                      'task_index': [ds['tasks'][v] for v in table['task_index'].to_pylist()]}
            for key, value in values.items():
                i = table.schema.get_field_index(key)
                table = table.set_column(i, table.schema.field(i), pa.array(value, type=pa.int64()))
            numeric = [k for k, f in info['features'].items() if f['dtype'] != 'video']
            table = table.select(numeric).replace_schema_metadata(None)
            chunk, file = divmod(episode_id, info['chunks_size'])
            dest = stage / info['data_path'].format(chunk_index=chunk, file_index=file)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
            row = {'episode_index': episode_id, 'length': len(table),
                   'tasks': source_ep['tasks'], 'data/chunk_index': chunk, 'data/file_index': file,
                   'dataset_from_index': offset, 'dataset_to_index': offset + len(table),
                   'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0}
            for key in numeric:
                array = np.asarray(table[key].to_pylist())
                for name, value in stats(array).items():
                    row[f'stats/{key}/{name}'] = value
            old_prefix, new_prefix = f'videos/{camera}', f'videos/{CAMERA}'
            video = root / original['video_path'].format(video_key=camera, chunk_index=source_ep[old_prefix+'/chunk_index'], file_index=source_ep[old_prefix+'/file_index'])
            if video not in video_files:
                video_id = len(video_files)
                video_chunk, video_file = divmod(video_id, info['chunks_size'])
                dest = stage / info['video_path'].format(video_key=CAMERA, chunk_index=video_chunk, file_index=video_file)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(video, dest)
                except OSError:
                    shutil.copy2(video, dest)
                video_files[video] = (video_chunk, video_file)
            vc, vf = video_files[video]
            row.update({new_prefix+'/chunk_index': vc, new_prefix+'/file_index': vf,
                        new_prefix+'/from_timestamp': source_ep[old_prefix+'/from_timestamp'],
                        new_prefix+'/to_timestamp': source_ep[old_prefix+'/to_timestamp']})
            for key, value in source_ep.items():
                if key.startswith(f'stats/{camera}/'):
                    row[key.replace(f'stats/{camera}/', f'stats/{CAMERA}/', 1)] = value
            rows.append(row)
            tables.append(table)
            provenance.append({'episode_index': episode_id, 'source_root': str(root),
                               'source_episode_index': source_ep['episode_index'], 'robot_type': original['robot_type'],
                               'source_camera': camera, 'state_source': 'observation.tcp_pose', 'action_source': 'action'})
            offset += len(table)
        if offset != info['total_frames']:
            raise ValueError('Frame count mismatch')
        merged = pa.concat_tables(tables)
        global_stats = {key: stats(np.asarray(merged[key].to_pylist())) for key in merged.column_names}
        from lerobot.datasets.compute_stats import aggregate_stats
        camera_stats = []
        for ds in datasets:
            original_stats = json.loads((ds['root'] / 'meta/stats.json').read_text())
            camera_stats.append({CAMERA: {k: np.asarray(v) for k,v in original_stats[ds['camera']].items()}})
        global_stats[CAMERA] = {k: v.tolist() for k,v in aggregate_stats(camera_stats)[CAMERA].items()}
        dest = stage / 'meta/episodes/chunk-000/file-000.parquet'
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), dest)
        import pandas as pd
        pd.DataFrame({'task_index': list(task_ids.values())}, index=pd.Index(list(task_ids), name='task')).to_parquet(stage / 'meta/tasks.parquet')
        write_json(stage/'meta/info.json', info)
        write_json(stage/'meta/stats.json', global_stats)
        write_json(stage/'merge_provenance.json', {'seed': seed, 'unit': 'episode', 'robot_type': robot_type,
                   'camera_mapping': {str(ds['root']): {ds['camera']: CAMERA} for ds in datasets},
                   'sources': [{'root': str(ds['root']), 'robot_type': ds['info']['robot_type'],
                                'episodes': ds['info']['total_episodes'], 'frames': ds['info']['total_frames']} for ds in datasets],
                   'episodes': provenance})
        write_json(stage/'shuffle_mapping.json', {'seed': seed, 'unit': 'episode', 'episodes': provenance})
        write_json(stage/'meta/modality.json', {'state': {'source': 'observation.state','names':POSE_NAMES},
                   'action': {'source': 'action','translation_frame':'current_tcp',
                              'rotation_delta':'rotvec(R_current.T @ R_next)','gripper':'next_absolute_command'},
                   'video': {'image': {'original_key': CAMERA}}})
        if output.exists():
            raise ValueError('Output appeared during merge')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'Created {output}: {len(records)} episodes, {offset} frames, seed={seed}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    merge(args.sources, args.output, args.seed)
