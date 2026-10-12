"""CPU tests of resume, failure recording, and task assignment. No GPU quality claims."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]


class Orchestration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root/'package'
        (self.package/'scripts').mkdir(parents=True)
        (self.package/'configs').mkdir()
        for name in ['run_batch.py', 'split_tasks.py']:
            shutil.copyfile(PACKAGE/'scripts'/name, self.package/'scripts'/name)
        (self.package/'scripts/common.py').write_text('def qc(video, profile):\n assert video.read_bytes()==b"valid"\n return {"full_decode_ok":True}\n')
        (self.package/'scripts/worker.py').write_text('import json,sys\nfrom pathlib import Path\nr=json.loads(Path(sys.argv[1]).read_text())\no=Path(r["output"])\n(o/"video.mp4").write_bytes(b"valid")\n')
        (self.package/'configs/hunyuan.json').write_text('{}')
        self.source = self.root/'source'
        self.source.mkdir()
        self.tasks = self.root/'tasks.json'
        self.tasks.write_text(json.dumps([dict(prompt_id='test_001', prompt='A horse.', n_objects=5)]))
        self.output = self.root/'output'

    def tearDown(self):
        self.temp.cleanup()

    def run_batch(self):
        return subprocess.run([sys.executable, str(self.package/'scripts/run_batch.py'), '--model', 'hunyuan',
                               '--source', str(self.source), '--weights', str(self.root/'weights'),
                               '--tasks', str(self.tasks), '--output-dir', str(self.output)], capture_output=True, text=True)

    def test_resume_does_not_repeat_successful_video(self):
        self.assertEqual(self.run_batch().returncode, 0)
        self.assertEqual(self.run_batch().returncode, 0)
        self.assertEqual(len(list((self.output/'test_001').glob('attempt-*'))), 1)

    def test_changed_prompt_cannot_reuse_completed_result(self):
        self.assertEqual(self.run_batch().returncode, 0)
        self.tasks.write_text(json.dumps([dict(prompt_id='test_001', prompt='A goat.', n_objects=5)]))
        self.assertNotEqual(self.run_batch().returncode, 0)
        record = json.loads((self.output/'test_001/result.json').read_text())
        self.assertEqual(record['prompt'], 'A horse.')

    def test_failed_worker_preserves_error_record(self):
        (self.package/'scripts/worker.py').write_text('raise RuntimeError("kernel failed")\n')
        self.assertNotEqual(self.run_batch().returncode, 0)
        record = json.loads((self.output/'test_001/result.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertIn('CalledProcessError', record['error'])
        self.assertFalse((self.output/'test_001/video.mp4').exists())

    def test_corrupt_completed_video_is_not_silently_skipped(self):
        self.assertEqual(self.run_batch().returncode, 0)
        (self.output/'test_001/video.mp4').write_bytes(b'broken')
        self.assertNotEqual(self.run_batch().returncode, 0)

    def test_shards_exclude_reservations_and_do_not_overlap(self):
        rows = [dict(prompt_id=f'id_{i}', prompt='Objects.', n_objects=20-i) for i in range(6)]
        self.tasks.write_text(json.dumps(rows))
        reservations = self.root/'reserved.json'
        reservations.write_text(json.dumps(dict(snapshot_date='test', reserved_ids=dict(hunyuan=['id_0']))))
        destination = self.root/'shards'
        result = subprocess.run([sys.executable, str(self.package/'scripts/split_tasks.py'), '--model', 'hunyuan',
                                 '--prompts', str(self.tasks), '--workers', '2', '--output-dir', str(destination),
                                 '--reservations', str(reservations)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        shards = [json.loads(p.read_text()) for p in sorted(destination.glob('*.json'))]
        ids = [{r['prompt_id'] for r in shard} for shard in shards]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(ids[0] | ids[1], {f'id_{i}' for i in range(1, 6)})
        self.assertTrue(all(shard[i]['n_objects'] >= shard[i+1]['n_objects'] for shard in shards for i in range(len(shard)-1)))


if __name__ == '__main__':
    unittest.main()
