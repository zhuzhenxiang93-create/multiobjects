"""Generate one video through the pinned LightX2V Wan pipeline."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    c, row = request['config'], request['row']
    if c['model'] != 'wan':
        raise ValueError('Only Wan is supported')
    sys.path.insert(0, c['source'])
    from lightx2v import LightX2VPipeline
    out = Path(request['output'])
    config = out/'lightx2v.json'
    config.write_text(json.dumps(c['lightx2v_config'], indent=2))
    pipe = LightX2VPipeline(model_path=c['weights'], model_cls='wan2.2_moe', task='t2v')
    pipe.create_generator(config_json=str(config))
    pipe.generate(seed=c['seed'], prompt=row['prompt'], negative_prompt='',
                  save_result_path=str(out/'video.mp4'))


if __name__ == '__main__':
    main()
