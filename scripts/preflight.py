"""Readiness check inside the selected model environment; does not generate a video."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    model, source, weights = request['model'], Path(request['source']), Path(request['weights'])
    expected = json.loads((ROOT/'verification/software_versions.json').read_text())[model]
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != expected['source_commit']:
        raise RuntimeError('Source version differs. Checkout '+expected['source_commit'])
    sys.path.insert(0, str(source))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU allocated; run inside your GPU job')
    required = {'hunyuan': ['sageattention', 'flash_attn'], 'wan': ['lightx2v', 'sageattention', 'flashinfer', 'zmq'],
                'ltx25': ['ltx_pipelines', 'av'], 'sana': ['diffusers', 'transformers', 'av', 'mmcv', 'xformers']}[model]
    for name in required:
        importlib.import_module(name)
    if model == 'sana':
        text = Path(request['text_encoder'])
        for file in ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
            if not (text/file).is_file():
                raise RuntimeError('Gemma component missing: '+file)
        for file in ['transformer/checkpoints/SANA_Video_2.0_5B_720p.pth', 'LTX-2.3-Diffusers/vae/config.json',
                     'LTX-2.3-Diffusers/vae/diffusion_pytorch_model.safetensors']:
            if not (weights/file).is_file():
                raise RuntimeError('SANA component missing: '+file)
        include = [sysconfig.get_path('include')]+os.getenv('CPATH', '').split(os.pathsep)+os.getenv('C_INCLUDE_PATH', '').split(os.pathsep)
        if not any(p and (Path(p)/'Python.h').is_file() for p in include):
            raise RuntimeError('Matching Python development headers missing. Set CPATH/C_INCLUDE_PATH; see docs/INSTALL.md')
    elif model == 'ltx25':
        template = json.loads((ROOT/'configs/ltx25_native.json').read_text())
        for item in template['argv']:
            if item.startswith('{WEIGHTS}') and not Path(item.replace('{WEIGHTS}', str(weights))).is_file():
                raise RuntimeError('Missing LTX component: '+item)
    for program in ['ffmpeg', 'ffprobe']:
        subprocess.run([program, '-version'], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps(dict(preflight='passed', model=model, source_commit=commit, torch=torch.__version__,
                          cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)), indent=2))


if __name__ == '__main__':
    main()
