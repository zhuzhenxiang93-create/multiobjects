"""Shared resumable orchestration. Run inside each model's independent environment."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import qc

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load_tasks(path):
    rows = json.loads(path.read_text())
    seen = set()
    for row in rows:
        identifier = row['prompt_id']
        if not isinstance(identifier, str) or not identifier or identifier in ['.', '..'] or '/' in identifier or '\\' in identifier:
            raise ValueError('Unsafe prompt_id')
        if identifier in seen:
            raise ValueError('Duplicate prompt_id: '+identifier)
        if not isinstance(row['prompt'], str) or not row['prompt'].strip():
            raise ValueError('Empty prompt: '+identifier)
        seen.add(identifier)
    return rows


def versions():
    result = {}
    for name in ['torch', 'transformers', 'diffusers', 'sageattention', 'lightx2v', 'av']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['wan'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    rows = load_tasks(args.tasks)
    source, weights = args.source.resolve(), args.weights.resolve()
    profile = json.loads((ROOT/'configs'/f'{args.model}.json').read_text())
    try:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = 'unavailable'
    code_hash = digest({f.name: f.read_text() for f in [Path(__file__), ROOT/'scripts/worker.py', ROOT/'scripts/common.py']})
    software = versions()
    config = dict(model=args.model, source=str(source), weights=str(weights), source_commit=commit,
                  profile=profile, seed=args.seed, software=software, runner_hash=code_hash,
                  text_encoder=None)
    if args.model == 'wan':
        config['lightx2v_config'] = json.loads((ROOT/'configs/wan_lightx2v.json').read_text())
    if args.dry_run:
        print(json.dumps(dict(model=args.model, tasks=len(rows), first=rows[0]['prompt_id'] if rows else None,
                              last=rows[-1]['prompt_id'] if rows else None, config=config), ensure_ascii=False, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Local locking only. Distributed workers must use disjoint manifests.
    with (output/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for row in rows:
            out = output/row['prompt_id']
            out.mkdir(exist_ok=True)
            result, video = out/'result.json', out/'video.mp4'
            fingerprint = digest(dict(prompt=row['prompt'], config=config))
            if result.exists():
                old = json.loads(result.read_text())
                if old.get('status') == 'complete':
                    if old.get('fingerprint') != fingerprint:
                        raise RuntimeError('Prompt, code, environment or configuration changed; use a new output directory')
                    qc(video, profile)
                    print('REUSE', row['prompt_id'], flush=True)
                    continue
            attempt = out/f'attempt-{time.time_ns()}'
            attempt.mkdir()
            request = dict(config=config, row=row, output=str(attempt))
            write(attempt/'request.json', request)
            record = dict(status='running', model=args.model, prompt_id=row['prompt_id'], prompt=row['prompt'],
                          category_counts=row.get('category_counts'), n_objects=row.get('n_objects'), seed=args.seed,
                          fingerprint=fingerprint, config=config, attempt=str(attempt), started_unix=time.time(),
                          job_id=os.getenv('SLURM_JOB_ID'))
            write(result, record)
            try:
                command = [sys.executable, str(ROOT/'scripts/worker.py'), str(attempt/'request.json')]
                record['command'] = command
                with (attempt/'generation.log').open('w') as log:
                    subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
                videos = list(attempt.rglob('*.mp4'))
                if len(videos) != 1:
                    raise RuntimeError(f'Expected one video, found {len(videos)}')
                checked = qc(videos[0], profile)
                videos[0].replace(video)
                record.update(status='complete', video=str(video), qc=checked)
            except BaseException:
                record.update(status='failed', error=traceback.format_exc())
                raise
            finally:
                record['elapsed_seconds'] = time.time()-record['started_unix']
                write(result, record)
            print(row['prompt_id'], row.get('n_objects'), record['status'], flush=True)
        write(output/'summary.json', dict(status='complete', planned=len(rows), complete=len(rows)))


if __name__ == '__main__':
    main()
