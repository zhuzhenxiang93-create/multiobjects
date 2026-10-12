"""Create disjoint collaborator manifests, conservatively excluding server reservations."""
import argparse
import json
from pathlib import Path
from run_batch import load_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['hunyuan', 'wan', 'ltx25', 'sana'], required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reservations', type=Path, default=Path(__file__).resolve().parents[1]/'tasks/server_reservations.json')
    p.add_argument('--min-objects', type=int, default=5)
    p.add_argument('--max-objects', type=int, default=20)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    reservations = json.loads(args.reservations.read_text())
    excluded = set(reservations['reserved_ids'][args.model])
    rows = [r for r in load_tasks(args.prompts) if args.min_objects <= r['n_objects'] <= args.max_objects and r['prompt_id'] not in excluded]
    rows.sort(key=lambda r: (-r['n_objects'], r['prompt_id']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = [rows[i::args.workers] for i in range(args.workers)]
    for i, shard in enumerate(shards):
        path = args.output_dir/f'{args.model}_worker{i+1:02d}.json'
        if path.exists():
            raise FileExistsError('Refuse replacing assigned task files: '+str(path))
    for i, shard in enumerate(shards):
        path = args.output_dir/f'{args.model}_worker{i+1:02d}.json'
        path.write_text(json.dumps(shard, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(model=args.model, tasks=len(rows), per_worker=[len(x) for x in shards],
                          excluded_reserved=len(excluded), snapshot=reservations['snapshot_date'])))


if __name__ == '__main__':
    main()
