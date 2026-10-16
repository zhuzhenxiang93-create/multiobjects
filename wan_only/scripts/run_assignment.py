"""One-command claimed assignment launcher. Each model still needs its GPU environment/weights."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from git_claims import ClaimError, GitClaims

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def expand(value):
    value = os.path.expandvars(value)
    if '$' in value:
        raise ClaimError('Workspace variable is unset. Pass --workspace /data/YOUR_DIRECTORY')
    return Path(value).expanduser().resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('lot', nargs='?')
    p.add_argument('--worker', help='Your GitHub username; repository write permission is required')
    p.add_argument('--config', type=Path, default=ROOT/'profiles/local.json')
    p.add_argument('--workspace', type=Path, help='Sets VIDEO_WORKSPACE for the default profile')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    index = json.loads((ROOT/'tasks/collaboration.json').read_text())
    if args.list:
        for name, lot in index['lots'].items():
            print(name, lot['tasks'], 'videos; check claims branch before execution')
        return
    if args.lot not in index['lots'] or not args.worker:
        p.error('Choose an existing lot and pass --worker YOUR_GITHUB_USERNAME')
    lot = index['lots'][args.lot]
    if lot['model'] != 'wan':
        raise ClaimError('This package only runs Wan')
    tasks = ROOT/lot['path']
    digest = hashlib.sha256(tasks.read_bytes()).hexdigest()
    if digest != lot['sha256']:
        raise ClaimError('Assignment file changed; refuse generating')
    if args.workspace:
        os.environ['VIDEO_WORKSPACE'] = str(args.workspace.expanduser().resolve())
    config_path = args.config
    if not config_path.exists() and config_path == ROOT/'profiles/local.json':
        config_path = ROOT/'profiles/local.example.json'
    config = json.loads(config_path.read_text())
    model = lot['model']
    profile = config['models'][model]
    workspace = expand(config['workspace'])
    python, source, weights = [expand(profile[k]) for k in ['python', 'source', 'weights']]
    output = workspace/'results'/model/args.lot
    command = [str(python), str(ROOT/'scripts/run_batch.py'), '--model', model,
               '--source', str(source), '--weights', str(weights), '--tasks', str(tasks), '--output-dir', str(output)]
    if args.dry_run:
        print(json.dumps(dict(lot=args.lot, videos=lot['tasks'], command=command,
                              note='No remote claim and no GPU generation started'), ensure_ascii=False, indent=2))
        return
    if not python.is_file() or not source.is_dir() or not weights.is_dir():
        raise ClaimError('Model environment, source or weights missing. Read README.md and fill profiles/local.json first')
    for name, value in profile.get('environment', {}).items():
        os.environ[name] = os.path.expandvars(value)
    for name, suffix in [('TMPDIR','tmp'), ('HF_HOME','cache/huggingface'), ('XDG_CACHE_HOME','cache'),
                         ('TRITON_CACHE_DIR','cache/triton'), ('TORCHINDUCTOR_CACHE_DIR','cache/torchinductor'),
                         ('CUDA_CACHE_PATH','cache/cuda'), ('FLASHINFER_WORKSPACE_BASE','cache/flashinfer')]:
        directory = workspace/suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(directory))
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    claims = GitClaims(Path(subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], text=True).strip()), args.lot, args.worker)
    origin = claims.git(['remote', 'get-url', 'origin']).stdout.strip()
    if 'github.com/zhuzhenxiang93-create/multiobjects' not in origin and 'github.com:zhuzhenxiang93-create/multiobjects' not in origin:
        raise ClaimError('Use the shared upstream repository as origin to ensure one shared claim registry')
    head = claims.git(['rev-parse', 'HEAD']).stdout.strip()
    latest = claims.git(['ls-remote', 'origin', 'refs/heads/main']).stdout.split()[0]
    if head != latest:
        raise ClaimError('Checkout is stale or is not main. Pull main before starting a new run')
    state_dir = workspace/'collaboration'/args.lot
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir/'local.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = state_dir/'preflight.json'
        write(request_path, dict(model=model, source=str(source), weights=str(weights),
                                 text_encoder=None))
        subprocess.run([str(python), str(ROOT/'scripts/preflight.py'), str(request_path)], check=True)
        state_path = state_dir/'claim.json'
        if state_path.exists():
            record = json.loads(state_path.read_text())
            if record['worker'] != args.worker or record['manifest_sha256'] != digest:
                raise ClaimError('Local claim does not match this assignment')
        else:
            record = dict(lot=args.lot, model=model, worker=args.worker, claim_id=str(uuid.uuid4()),
                          manifest_sha256=digest, status='claimed', source_commit=head,
                          claimed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(state_path, record)
        claim_sha = claims.acquire(record)
        print('CLAIMED', args.lot, args.worker, flush=True)
        result = subprocess.run(command)
        record.update(status='complete' if result.returncode == 0 else 'failed', exit_code=result.returncode,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(state_path, record)
        claims.finish(record, claim_sha)
        print('RESULT', record['status'], args.lot, 'Videos:', output, flush=True)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    try:
        main()
    except (ClaimError, BlockingIOError) as exc:
        print('STOP:', exc, file=sys.stderr)
        raise SystemExit(2)
