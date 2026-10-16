"""Persistent exclusive task claims, stored as Git refs with compare-and-swap."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


class ClaimError(RuntimeError):
    pass


class GitClaims:
    def __init__(self, repository, lot, worker):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,80}', lot):
            raise ClaimError('Invalid task lot')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', worker):
            raise ClaimError('--worker must be your GitHub username')
        self.repository = Path(repository)
        self.lot, self.worker = lot, worker
        self.ref = 'refs/heads/claims/'+lot
        self.record_path = '.claims/'+lot+'.json'

    def git(self, args, *, input=None, env=None, check=True):
        result = subprocess.run(['git', '-C', str(self.repository)]+args, input=input, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if check and result.returncode:
            raise ClaimError(result.stderr.strip() or 'Git operation failed')
        return result

    def remote(self):
        result = self.git(['ls-remote', '--exit-code', 'origin', self.ref], check=False)
        if result.returncode == 2:
            return None
        if result.returncode:
            raise ClaimError(result.stderr.strip() or 'Cannot check task claim; refuse generating')
        sha = result.stdout.split()[0]
        self.git(['fetch', '--quiet', '--no-tags', 'origin', self.ref])
        fetched = self.git(['rev-parse', 'FETCH_HEAD']).stdout.strip()
        if fetched != sha:
            raise ClaimError('Claim changed while reading; retry')
        record = json.loads(self.git(['show', sha+':'+self.record_path]).stdout)
        return sha, record

    def commit(self, record, parent):
        env = os.environ.copy()
        env.update(GIT_AUTHOR_NAME=self.worker, GIT_COMMITTER_NAME=self.worker,
                   GIT_AUTHOR_EMAIL=self.worker+'@users.noreply.github.com',
                   GIT_COMMITTER_EMAIL=self.worker+'@users.noreply.github.com')
        with tempfile.TemporaryDirectory() as temporary:
            env['GIT_INDEX_FILE'] = str(Path(temporary)/'index')
            self.git(['read-tree', parent], env=env)
            blob = self.git(['hash-object', '-w', '--stdin'], input=json.dumps(record, indent=2)+'\n', env=env).stdout.strip()
            self.git(['update-index', '--add', '--cacheinfo', '100644', blob, self.record_path], env=env)
            tree = self.git(['write-tree'], env=env).stdout.strip()
            return self.git(['commit-tree', tree, '-p', parent, '-m', 'Task '+self.lot+': '+record['status']], env=env).stdout.strip()

    def acquire(self, record):
        existing = self.remote()
        if existing:
            sha, current = existing
            if (current.get('claim_id'), current.get('worker'), current.get('manifest_sha256')) != (
                    record['claim_id'], self.worker, record['manifest_sha256']):
                raise ClaimError('Task already claimed by '+str(current.get('worker'))+'; refuse duplicate generation')
            if current.get('status') == 'complete':
                raise ClaimError('Task already complete; no new generation required')
            return sha
        parent = self.git(['rev-parse', 'HEAD']).stdout.strip()
        sha = self.commit(record, parent)
        # Empty expected ref means CREATE ONLY. Concurrent claims cannot overwrite each other.
        result = self.git(['push', '--porcelain', '--force-with-lease='+self.ref+':',
                           'origin', sha+':'+self.ref], check=False)
        if result.returncode:
            raise ClaimError('Claim not acquired; no generation started. '+result.stderr.strip())
        return sha

    def finish(self, record, expected_sha):
        sha = self.commit(record, expected_sha)
        self.git(['push', '--porcelain', '--force-with-lease='+self.ref+':'+expected_sha,
                  'origin', sha+':'+self.ref])
        return sha
