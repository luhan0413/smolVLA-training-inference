#!/usr/bin/env python3
"""Shuffle complete episodes in a local LeRobot v3 dataset (one data file/episode).

Run with .venv/bin/python shuffle_dataset.py --seed 42
The source is never modified. Videos are copied without re-encoding.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def statistics(values, keys):
    a = np.asarray(values, dtype=np.float64)
    result = {}
    for key in keys:
        if key == 'count':
            value = len(a)
        elif key in ('min', 'max', 'mean', 'std'):
            value = float(getattr(a, key)())
        elif key.startswith('q'):
            value = float(np.quantile(a, int(key[1:]) / 100))
        else:
            raise ValueError(f'Unsupported statistic: {key}')
        result[key] = [value]
    return result


def shuffle(source, output, seed, robot_sources=None):
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError('Output must be a new directory outside the source dataset.')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('Only LeRobot v3.0 is supported.')
    n = info['total_episodes']
    if info.get('splits') != {'train': f'0:{n}'}:
        raise ValueError('Only a full train split is supported; refusing to mix dataset splits.')
    metadata = pa.concat_tables([pq.read_table(p) for p in sorted((source / 'meta/episodes').rglob('*.parquet'))])
    rows = sorted(metadata.to_pylist(), key=lambda r: r['episode_index'])
    if [r['episode_index'] for r in rows] != list(range(n)):
        raise ValueError('Episode indices must be contiguous.')
    paths = [source / info['data_path'].format(chunk_index=r['data/chunk_index'], file_index=r['data/file_index']) for r in rows]
    if len(set(paths)) != n:
        raise ValueError('This script requires one parquet data file per episode.')
    order = list(range(n))
    rng = random.Random(seed)
    balance = None
    if robot_sources is not None:
        provenance = json.loads((source / 'merge_provenance.json').read_text())
        lookup = {r['episode_index']: r for r in provenance['episodes']}
        groups = {'ur5': [], 'ur7e': []}
        for old in order:
            robot = robot_sources[lookup[old]['source']]
            groups[robot].append(old)
        count = min(map(len, groups.values()))
        if not count:
            raise ValueError('Both UR5 and UR7e must have episodes.')
        order = [old for group in groups.values() for old in rng.sample(group, count)]
        balance = {'unit': 'episode', 'method': 'downsample_without_replacement',
                   'source_robot_mapping': robot_sources,
                   'original_counts': {k: len(v) for k, v in groups.items()},
                   'selected_counts': {k: count for k in groups}}
    rng.shuffle(order)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.tmp-', dir=output.parent))
    try:
        shutil.copytree(source / 'meta', stage / 'meta')
        shutil.rmtree(stage / 'meta/episodes')
        if balance is None:
            shutil.copytree(source / 'videos', stage / 'videos')
        else:
            for old in order:
                for key, feature in info['features'].items():
                    if feature['dtype'] != 'video':
                        continue
                    prefix = f'videos/{key}'
                    rel = info['video_path'].format(video_key=key,
                        chunk_index=rows[old][prefix + '/chunk_index'],
                        file_index=rows[old][prefix + '/file_index'])
                    dest = stage / rel
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source / rel, dest)
        offset, new_rows, episode_ids = 0, [], []
        for new, old in enumerate(order):
            row = dict(rows[old])
            table = pq.read_table(paths[old])
            length = row['length']
            if table.num_rows != length or set(table['episode_index'].to_pylist()) != {old}:
                raise ValueError(f'Invalid episode data: {old}')
            if table['index'].to_pylist() != list(range(row['dataset_from_index'], row['dataset_to_index'])):
                raise ValueError(f'Invalid frame indices: {old}')
            chunk, file = divmod(new, info['chunks_size'])
            values = {'episode_index': [new] * length, 'index': list(range(offset, offset + length))}
            for key, data in values.items():
                i = table.schema.get_field_index(key)
                table = table.set_column(i, table.schema.field(i), pa.array(data, type=table.schema.field(i).type))
                prefix = f'stats/{key}/'
                keys = [k[len(prefix):] for k in row if k.startswith(prefix)]
                row.update({prefix + k: v for k, v in statistics(data, keys).items()})
            dest = stage / info['data_path'].format(chunk_index=chunk, file_index=file)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
            row.update({'episode_index': new, 'data/chunk_index': chunk, 'data/file_index': file,
                        'dataset_from_index': offset, 'dataset_to_index': offset + length,
                        'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
            new_rows.append(row)
            episode_ids.extend(values['episode_index'])
            offset += length
        if offset != sum(rows[old]['length'] for old in order):
            raise ValueError('Total frame count mismatch.')
        dest = stage / 'meta/episodes/chunk-000/file-000.parquet'
        dest.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(new_rows, schema=metadata.schema), dest)
        stats = json.loads((stage / 'meta/stats.json').read_text())
        if balance is not None:
            from lerobot.datasets.compute_stats import aggregate_stats
            per_episode = []
            for row in new_rows:
                episode_stats = {}
                for key, value in row.items():
                    if key.startswith('stats/'):
                        feature, stat = key[6:].rsplit('/', 1)
                        episode_stats.setdefault(feature, {})[stat] = np.atleast_1d(value)
                per_episode.append(episode_stats)
            stats = {k: {s: v.tolist() for s, v in values.items()}
                     for k, values in aggregate_stats(per_episode).items()}
            info.update(total_episodes=len(order), total_frames=offset,
                        splits={'train': f'0:{len(order)}'}, robot_type='ur5_ur7e')
            write_json(stage / 'meta/info.json', info)
            balance['frames_by_robot'] = {
                robot: sum(rows[old]['length'] for old in order if old in group)
                for robot, group in groups.items()}
            write_json(stage / 'balance_report.json', balance)
        for key, data in [('episode_index', episode_ids), ('index', range(offset))]:
            if key in stats:
                stats[key] = statistics(data, stats[key])
        write_json(stage / 'meta/stats.json', stats)
        provenance = source / 'merge_provenance.json'
        if provenance.exists():
            value = json.loads(provenance.read_text())
            lookup = {r['episode_index']: r for r in value['episodes']}
            value['episodes'] = [dict(lookup[old], episode_index=new) for new, old in enumerate(order)]
            if balance is not None:
                for entry in value['episodes']:
                    entry['robot_type'] = robot_sources[entry['source']]
                value['robot_type'] = 'Mixed UR5 and UR7e; see per-episode robot_type'
            write_json(stage / provenance.name, value)
        write_json(stage / 'shuffle_mapping.json', {'source': str(source), 'seed': seed,
                   'episodes': [{'episode_index': new, 'original_episode_index': old} for new, old in enumerate(order)]})
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f'Created {output}: {len(order)} episodes, {offset} frames, seed={seed}')
    print(f'First 10 original episode indices: {order[:10]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent / 'datasets/training-data_6')
    parser.add_argument('--output', type=Path, help='Default: SOURCE_shuffled')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--robot-sources', type=Path,
                        help='JSON mapping every provenance source to ur5 or ur7e; enables equal-episode downsampling')
    args = parser.parse_args()
    shuffle(args.source, args.output or args.source.with_name(args.source.name + '_shuffled'), args.seed,
            json.loads(args.robot_sources.read_text()) if args.robot_sources else None)


if __name__ == '__main__':
    main()
