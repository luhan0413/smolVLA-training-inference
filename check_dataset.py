"""Read-only LeRobot v3 / SmolVLA preflight. No model weights or GPU required."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class InvalidDataset(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidDataset(message)


def train_options(args):
    """Read both --key=value and --key value, respecting the final occurrence."""
    result = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            key, sep, value = arg[2:].partition('=')
            if not sep:
                i += 1
                require(i < len(args), f'Missing value for --{key}')
                value = args[i]
            result[key] = value
        i += 1
    return result


def validate_tasks(tasks):
    # LeRobot resolves task_index positionally via tasks.iloc[task_index].name.
    require('task_index' in tasks, 'tasks.parquet 缺少 task_index。')
    require(all(isinstance(t, str) and t.strip() for t in tasks.index),
            'tasks.parquet 必須以任務文字作為 DataFrame index；'
            '只有 task 欄位不足以供 LeRobot loader 讀取，請先 set_index("task") 並保存 index。')
    require(tasks.index.is_unique, 'tasks.parquet 任務文字重複。')
    require(np.array_equal(tasks.task_index.to_numpy(), np.arange(len(tasks))),
            'tasks.parquet task_index 必須依列順序從 0 連續編號，以符合 loader 的 iloc 查找。')
    return list(tasks.index)


def audit(root, repo_id, report, sample_video=False, options=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    options = options or {}
    info = json.loads((root / 'meta/info.json').read_text())
    require(info.get('codebase_version') == 'v3.0', '需要 LeRobotDataset v3.0；請先轉換舊版資料。')
    fps = info['fps']
    require(np.isfinite(fps) and fps > 0, 'fps 必須大於 0。')
    features = info['features']
    cameras = [k for k, v in features.items() if v['dtype'] in ('video', 'image')]
    require(cameras, '缺少 RGB camera feature。')
    for key in ('observation.state', 'action'):
        require(key in features, f'缺少 {key}')
        shape = features[key]['shape']
        require(len(shape) == 1 and 0 < shape[0] <= 32, f'{key}: 此工具支援 SmolVLA base 的 1–32 維向量。')
        require(features[key]['dtype'] == 'float32', f'{key}: 必須為 float32。')
        names = features[key].get('names')
        require(names is None or len(names) == shape[0], f'{key}: names 長度與 shape 不符。')
    for key in cameras:
        require(features[key]['shape'][-1] == 3, f'{key}: 需要 HWC 三通道 RGB schema。')

    files = sorted((root / 'data').glob('chunk-*/*.parquet'))
    ep_files = sorted((root / 'meta/episodes').glob('chunk-*/*.parquet'))
    require(files and ep_files, '缺少 data 或 episodes parquet。')
    frames = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    tasks = pd.read_parquet(root / 'meta/tasks.parquet')
    stats = json.loads((root / 'meta/stats.json').read_text())
    require(len(frames) == info['total_frames'] > 0, 'total_frames 與 parquet 不符或資料為空。')
    require(len(episodes) == info['total_episodes'] > 0, 'total_episodes 與 metadata 不符。')
    require(len(tasks) == info['total_tasks'] > 0, 'total_tasks 與 tasks parquet 不符。')
    require(set(features) - set(cameras) <= set(frames.columns), 'Parquet 缺少 schema 宣告的數值欄位。')
    require(set(frames.task_index) <= set(tasks.task_index), '存在未知 task_index。')
    require(not tasks.task_index.duplicated().any(), 'task_index 重複。')
    task_texts = validate_tasks(tasks)
    require(np.array_equal(frames['index'], np.arange(len(frames))), '全域 index 必須連續且從 0 起算。')
    require(np.array_equal(episodes.episode_index, np.arange(len(episodes))), 'episode_index 必須連續且從 0 起算。')
    report['summary'] = dict(episodes=len(episodes), frames=len(frames), fps=fps,
                             state_dim=features['observation.state']['shape'][0],
                             action_dim=features['action']['shape'][0], cameras=cameras,
                             robot_type=info.get('robot_type'), tasks=list(task_texts))

    for key in features:
        if key in cameras:
            continue
        values = np.stack(frames[key].to_numpy())
        require(np.isfinite(values).all(), f'{key}: 包含 NaN/Inf。')
        expected = tuple(features[key]['shape'])
        actual = values.shape[1:] or (1,)
        require(actual == expected, f'{key}: shape {actual} != schema {expected}')
        if key in ('observation.state', 'action'):
            require(values.dtype == np.float32, f'{key}: 實際 dtype 不是 float32。')
            require(key in stats, f'{key}: 缺少 normalization 統計。')
            for name in ('mean', 'std', 'min', 'max'):
                stored = np.asarray(stats[key][name])
                require(stored.shape == expected and np.isfinite(stored).all(), f'{key}/{name}: 統計維度不符或非有限值。')
                require(np.allclose(stored, getattr(values.astype(np.float64), name)(axis=0), rtol=1e-3, atol=1e-5),
                        f'{key}/{name}: 統計與資料不符，請重新計算 stats。')
            if np.any(np.asarray(stats[key]['std']) < 1e-8):
                report['warnings'].append(f'{key}: 有近乎不變的維度，請確認是預期行為。')
    report['checks'].append('所有數值欄位的 shape/finite 與 state/action normalization 統計通過')

    offset = 0
    sampled = set()
    for ep in episodes.to_dict('records'):
        start, end = int(ep['dataset_from_index']), int(ep['dataset_to_index'])
        require(start == offset and end > start and end - start == ep['length'], f"episode {ep['episode_index']}: 邊界或長度錯誤。")
        seg = frames.iloc[start:end]
        require(len(seg) == end - start and (seg.episode_index == ep['episode_index']).all(), 'episode row 對齊錯誤。')
        require(np.array_equal(seg.frame_index, np.arange(len(seg))), 'frame_index 不連續。')
        require(np.allclose(seg.timestamp, np.arange(len(seg)) / fps, atol=1e-4, rtol=0), 'timestamp 與 FPS 不符。')
        data_path = root / info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
        require(data_path.is_file(), f'缺少 data file: {data_path}')
        for camera in cameras:
            if features[camera]['dtype'] != 'video':
                continue
            prefix = f'videos/{camera}'
            path = root / info['video_path'].format(video_key=camera, chunk_index=int(ep[f'{prefix}/chunk_index']), file_index=int(ep[f'{prefix}/file_index']))
            require(path.is_file(), f'缺少影片: {path}')
            a, b = ep[f'{prefix}/from_timestamp'], ep[f'{prefix}/to_timestamp']
            require(np.isfinite([a,b]).all() and 0 <= a < b and b-a >= (len(seg)-1)/fps-1e-4, f'{camera}: 影片時間範圍不足。')
        sampled.update((start, (start+end-1)//2, end-1))
        offset = end
    require(offset == len(frames), 'episode metadata 未涵蓋所有 frames。')
    report['checks'].append('所有 episode 邊界、索引、FPS 時間戳及引用檔案通過')

    if options:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        require(options.get('policy.type', 'smolvla') == 'smolvla', '此 preflight 僅支援 SmolVLA。')
        if 'policy.input_features' in options:
            expected = json.loads(options['policy.input_features'])
        else:
            cfg = SmolVLAConfig.from_pretrained(options.get('policy.path', 'lerobot/smolvla_base'))
            expected = {k: {'shape': list(v.shape), 'type': v.type.value} for k,v in cfg.input_features.items()}
        for key, option in [('observation.state', 'policy.max_state_dim'), ('action', 'policy.max_action_dim')]:
            require(features[key]['shape'][0] <= int(options.get(option, 32)), f'{key}: 超過模型 padding 維度。')
        rename = json.loads(options.get('rename_map', '{}'))
        available = {rename.get(k,k): v for k,v in features.items()}
        for key, spec in expected.items():
            require(key in available, f'模型需要 {key}，資料未提供；請設定 input_features 或 rename_map。')
            if spec['type'] == 'STATE':
                require(spec['shape'] == available[key]['shape'], f'{key}: 模型與資料 state 維度不符。')
        report['checks'].append('訓練參數中的模型 state 維度與 camera keys 通過')
    chunk = int(options.get('policy.chunk_size', 50))
    require(chunk > 0, 'chunk_size 必須大於 0。')
    # Local dataset validation must not silently fetch or repair missing data.
    from huggingface_hub import constants as hub_constants
    hub_constants.HF_HUB_OFFLINE = True
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps={'action': [i/fps for i in range(chunk)]},
                        video_backend=options.get('dataset.video_backend', 'torchcodec'))
    indices = sorted(sampled) if sample_video else range(len(ds))
    for count, index in enumerate(indices, 1):
        item = ds[index]
        require(tuple(item['action'].shape) == (chunk, features['action']['shape'][0]), 'action chunk 維度錯誤。')
        require(isinstance(item['task'], str) and item['task'].strip(), 'loader 未提供有效 task。')
        for camera in cameras:
            img = item[camera]
            h,w,c = features[camera]['shape']
            require(tuple(img.shape) == (c,h,w), f'{camera}: 解碼後 shape 不符。')
            require(bool(img.isfinite().all()) and img.min() >= 0 and img.max() <= 1, f'{camera}: 影像值非有限值或不在 [0,1]。')
        if count % 1000 == 0:
            print(f'影像/action chunk 已檢查 {count}/{len(indices)}', flush=True)
    report['checks'].append(f'LeRobot loader 解码、task、action chunk: {len(indices)} frames 通過')
    report['video_coverage'] = 'sampled' if sample_video else 'all_frames'
    if sample_video:
        report['warnings'].append('影像僅抽樣每段起中末；未檢查其餘影格。正式訓練預檢使用全量。')
    report['warnings'].append('通過表示資料可被此 pipeline 讀取；不保證任務成功率、示範品質或 TCP 座標/旋轉/增量控制語意正確。')
    if 'observation.joints' in features:
        report['warnings'].append('額外 observation.joints 不會自動拼入 observation.state。')


def main():
    parser = argparse.ArgumentParser(description='SmolVLA 訓練前資料檢查（預設全量影像；不修改資料）')
    parser.add_argument('root', nargs='?', help='資料集根目錄（直接檢查時必填）')
    parser.add_argument('--repo-id', default='local/preflight')
    parser.add_argument('--report', default=str(Path(__file__).parent/'reports/dataset_check.json'))
    parser.add_argument('--sample-video', action='store_true')
    parser.add_argument('--training-args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.root and not args.training_args:
        parser.error('請指定資料集根目錄，例如 python check_dataset.py /path/to/dataset')
    report = dict(status='FAIL', checked_at=datetime.now(timezone.utc).isoformat(), checks=[], warnings=[], errors=[])
    try:
        options = train_options(args.training_args or [])
        root_arg = options.get('dataset.root', args.root)
        repo_id = options.get('dataset.repo_id', args.repo_id)
        if args.training_args and 'dataset.root' not in options:
            from huggingface_hub import snapshot_download
            root_arg = snapshot_download(repo_id, repo_type='dataset', revision=options.get('dataset.revision'))
        root = Path(root_arg).expanduser().resolve()
        report['dataset_root'] = str(root)
        audit(root, repo_id, report, args.sample_video, options)
        report['status'] = 'PASS'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    dest = Path(args.report).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f"\n{report['status']}: {report.get('dataset_root', args.root)}")
    for label, key in [('OK','checks'),('WARN','warnings'),('ERROR','errors')]:
        for message in report[key]:
            print(f'[{label}] {message}')
    print(f'Report: {dest}')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
