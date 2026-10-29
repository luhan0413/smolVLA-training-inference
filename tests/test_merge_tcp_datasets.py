import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from merge_tcp_datasets import merge, stats, POSE_NAMES, ACTION_NAMES


def snapshot(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


class MergeTests(unittest.TestCase):
    def test_same_robot_distinct_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root/'open', root/'close'
            for source in (a, b):
                self.make_source(source, 'observation.images.arm', 'ur7e', 1, 10)
            pd.DataFrame({'task_index': [0]}, index=pd.Index(['close drawer'], name='task')).to_parquet(b/'meta/tasks.parquet')
            ep_path = b/'meta/episodes/chunk-000/file-000.parquet'
            rows = pq.read_table(ep_path).to_pylist()
            rows[0]['tasks'] = ['close drawer']
            pq.write_table(pa.Table.from_pylist(rows), ep_path)
            out = merge([a, b], root/'merged')
            info = json.loads((out/'meta/info.json').read_text())
            self.assertEqual(info['robot_type'], 'ur7e')
            self.assertEqual(info['total_tasks'], 2)
            tasks = pd.read_parquet(out/'meta/tasks.parquet')
            episodes = pq.read_table(out/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
            for episode in episodes:
                frames = pq.read_table(out/f"data/chunk-000/episode_{episode['episode_index']:06d}.parquet")
                for task_id in frames['task_index'].to_pylist():
                    self.assertEqual(tasks.iloc[task_id].name, episode['tasks'][0])
            provenance = json.loads((out/'merge_provenance.json').read_text())
            self.assertEqual(provenance['robot_type'], 'ur7e')

    def make_source(self, root, camera, robot, n_episodes, base):
        (root/'meta/episodes/chunk-000').mkdir(parents=True)
        (root/'data/chunk-000').mkdir(parents=True)
        (root/f'videos/{camera}/chunk-000').mkdir(parents=True)
        features={key:{'dtype':'float32','shape':[7],'names':names} for key,names in [
            ('observation.state',['joint']*7),('observation.tcp_pose',POSE_NAMES),('action',ACTION_NAMES)]}
        for key,dtype in [('timestamp','float32'),('frame_index','int64'),('episode_index','int64'),('index','int64'),('task_index','int64'),('next.done','bool')]:
            features[key]={'dtype':dtype,'shape':[1],'names':None}
        features[camera]={'dtype':'video','shape':[2,2,3],'names':['height','width','channels'],
                         'info':{'video.codec':'h264','video.pix_fmt':'yuv420p','video.fps':10}}
        arrays={'observation.state':pa.array([[base+100]*7]*(n_episodes*2),type=pa.list_(pa.float32(),7)),
                'observation.tcp_pose':pa.array([[base+i]*7 for i in range(n_episodes*2)],type=pa.list_(pa.float32(),7)),
                'action':pa.array([[base/100]*7]*(n_episodes*2),type=pa.list_(pa.float32(),7)),
                'timestamp':pa.array([0.,.1]*n_episodes,type=pa.float32()),
                'frame_index':pa.array([0,1]*n_episodes,type=pa.int64()),
                'episode_index':pa.array([i for i in range(n_episodes) for _ in range(2)],type=pa.int64()),
                'index':pa.array(range(n_episodes*2),type=pa.int64()),'task_index':pa.array([0]*n_episodes*2,type=pa.int64()),
                'next.done':pa.array([False,True]*n_episodes)}
        table=pa.table(arrays);pq.write_table(table,root/'data/chunk-000/file-000.parquet')
        image_stats={k:np.zeros((3,1,1)).tolist() for k in ['min','max','mean','std','q01','q10','q50','q90','q99']};image_stats['count']=[n_episodes*2]
        rows=[]
        for i in range(n_episodes):
            row={'episode_index':i,'length':2,'tasks':['open drawer'],'data/chunk_index':0,'data/file_index':0,
                 'dataset_from_index':i*2,'dataset_to_index':i*2+2,f'videos/{camera}/chunk_index':0,
                 f'videos/{camera}/file_index':0,f'videos/{camera}/from_timestamp':i*.2,f'videos/{camera}/to_timestamp':(i+1)*.2}
            row.update({f'stats/{camera}/{k}':v for k,v in image_stats.items()});rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows),root/'meta/episodes/chunk-000/file-000.parquet')
        info={'codebase_version':'v3.0','fps':10,'features':features,'robot_type':robot,'total_episodes':n_episodes,
              'total_frames':n_episodes*2,'total_tasks':1,'chunks_size':1000,'splits':{'train':f'0:{n_episodes}'},
              'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
        (root/'meta/info.json').write_text(json.dumps(info));numbers={k:stats(v.to_pylist()) for k,v in arrays.items()};numbers[camera]=image_stats
        (root/'meta/stats.json').write_text(json.dumps(numbers))
        pd.DataFrame({'task_index':[0]},index=pd.Index(['open drawer'],name='task')).to_parquet(root/'meta/tasks.parquet')
        (root/f'videos/{camera}/chunk-000/file-000.mp4').write_bytes(b'video-test')

    def test_packed_episodes_tcp_mapping_camera_offsets_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'ur7e';b=root/'ur5'
            self.make_source(a,'observation.images.left','ur7e',2,10)
            self.make_source(b,'observation.images.right','ur5',1,20)
            before=(snapshot(a),snapshot(b));out=merge([a,b],root/'merged',42)
            mapping=json.loads((out/'merge_provenance.json').read_text())['episodes']
            rows=pq.read_table(out/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
            frames=[]
            for row,provenance in zip(rows,mapping):
                source=Path(provenance['source_root']);old=provenance['source_episode_index']
                original=pq.read_table(source/'data/chunk-000/file-000.parquet').slice(old*2,2)
                table=pq.read_table(out/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
                self.assertEqual(table['observation.state'],original['observation.tcp_pose'])
                self.assertEqual(table['action'],original['action'])
                self.assertEqual(row['videos/observation.images.arm/from_timestamp'],old*.2)
                self.assertTrue((out/f"videos/observation.images.arm/chunk-000/file-{row['videos/observation.images.arm/file_index']:03d}.mp4").exists())
                frames.append(table)
            table=pa.concat_tables(frames)
            self.assertEqual(table['index'].to_pylist(),list(range(6)))
            self.assertEqual(json.loads((out/'meta/stats.json').read_text())['observation.state'],stats(table['observation.state'].to_pylist()))
            again=merge([a,b],root/'again',42)
            self.assertEqual(mapping,json.loads((again/'merge_provenance.json').read_text())['episodes'])
            self.assertEqual(before,(snapshot(a),snapshot(b)))
            with self.assertRaises(ValueError):merge([a,b],out,42)


if __name__=='__main__':unittest.main()
