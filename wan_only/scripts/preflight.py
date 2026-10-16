"""Import and allocated-GPU checks; this is not a generation acceptance test."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    assert request['model'] == 'wan'
    source, weights = Path(request['source']), Path(request['weights'])
    expected = json.loads((ROOT/'verification/software_versions.json').read_text())['wan']
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != expected['source_commit']:
        raise RuntimeError('Checkout LightX2V '+expected['source_commit'])
    sys.path.insert(0, str(source))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU allocated; run inside your GPU allocation')
    for name in ['lightx2v', 'sageattention', 'flashinfer', 'zmq']:
        importlib.import_module(name)
    if not any(weights.iterdir()):
        raise RuntimeError('Weight directory is empty')
    for program in ['ffmpeg', 'ffprobe']:
        subprocess.run([program, '-version'], check=True, stdout=subprocess.DEVNULL)
    from sageattention import sageattn
    q = torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.bfloat16)
    y = sageattn(q, q, q)
    if not torch.isfinite(y).all():
        raise RuntimeError('SageAttention returned NaN/Inf')
    print(json.dumps(dict(preflight='passed', source_commit=commit, torch=torch.__version__,
                          cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)), indent=2))


if __name__ == '__main__':
    main()
