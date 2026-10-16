"""Real Git compare-and-swap tests against local bare remotes; no network or GPU."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from git_claims import ClaimError, GitClaims


class Claims(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.remote = self.base/'remote.git'
        subprocess.run(['git', 'init', '--bare', str(self.remote)], check=True, capture_output=True)
        self.first = self.base/'first'
        subprocess.run(['git', 'clone', str(self.remote), str(self.first)], check=True, capture_output=True)
        for key, value in [('user.name', 'tester'), ('user.email', 'tester@example.invalid')]:
            subprocess.run(['git', '-C', str(self.first), 'config', key, value], check=True)
        (self.first/'README.md').write_text('Task registry test\n')
        subprocess.run(['git', '-C', str(self.first), 'add', 'README.md'], check=True)
        subprocess.run(['git', '-C', str(self.first), 'commit', '-m', 'Initialize'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(self.first), 'push', 'origin', 'HEAD'], check=True, capture_output=True)
        self.second = self.base/'second'
        subprocess.run(['git', 'clone', str(self.remote), str(self.second)], check=True, capture_output=True)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, owner, claim_id):
        return dict(worker=owner, claim_id=claim_id, manifest_sha256='frozen-manifest', status='claimed')

    def test_second_worker_cannot_take_claim_and_owner_can_resume(self):
        claim = GitClaims(self.first, 'wan-test-01', 'alice')
        record = self.record('alice', 'one')
        sha = claim.acquire(record)
        self.assertEqual(claim.acquire(record), sha)
        with self.assertRaises(ClaimError):
            GitClaims(self.second, 'wan-test-01', 'bob').acquire(self.record('bob', 'two'))

    def test_same_username_on_another_machine_cannot_duplicate(self):
        GitClaims(self.first, 'wan-test-01', 'alice').acquire(self.record('alice', 'one'))
        with self.assertRaises(ClaimError):
            GitClaims(self.second, 'wan-test-01', 'alice').acquire(self.record('alice', 'different-local-receipt'))

    def test_simultaneous_claims_have_exactly_one_winner(self):
        def acquire(path, name):
            try:
                GitClaims(path, 'wan-test-01', name).acquire(self.record(name, name))
                return True
            except ClaimError:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: acquire(*item), [(self.first, 'alice'), (self.second, 'bob')]))
        self.assertEqual(sum(results), 1)

    def test_completion_is_visible_and_does_not_change_main(self):
        initial = subprocess.check_output(['git', '-C', str(self.first), 'rev-parse', 'HEAD'], text=True).strip()
        claim = GitClaims(self.first, 'wan-test-01', 'alice')
        record = self.record('alice', 'one')
        sha = claim.acquire(record)
        record['status'] = 'complete'
        claim.finish(record, sha)
        self.assertEqual(claim.remote()[1]['status'], 'complete')
        with self.assertRaises(ClaimError):
            claim.acquire(record)
        self.assertEqual(subprocess.check_output(['git', '-C', str(self.first), 'rev-parse', 'HEAD'], text=True).strip(), initial)


if __name__ == '__main__':
    unittest.main()
