"""Single-video adapters using the accepted server configurations."""
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    c, row = request['config'], request['row']
    source, weights, out = Path(c['source']), Path(c['weights']), Path(request['output'])
    sys.path.insert(0, str(source))
    model, seed = c['model'], c['seed']
    if model == 'hunyuan':
        cmd = [str(Path(sys.executable).parent/'torchrun'), '--standalone', '--nproc_per_node=1',
               str(source/'generate.py'), '--prompt', row['prompt'], '--negative_prompt', '', '--image_path', 'none',
               '--resolution', '720p', '--aspect_ratio', '16:9', '--video_length', '121', '--num_inference_steps', '50',
               '--seed', str(seed), '--model_path', str(weights), '--output_path', str(out/'video.mp4'),
               '--rewrite', 'false', '--sr', 'false', '--cfg_distilled', 'false', '--enable_step_distill', 'false',
               '--sparse_attn', 'false', '--use_sageattn', 'true', '--enable_cache', 'true', '--cache_type', 'deepcache',
               '--total_steps', '50', '--offloading', 'false', '--group_offloading', 'false', '--overlap_group_offloading', 'false']
        subprocess.run(cmd, cwd=source, check=True)
        return
    if model == 'wan':
        from lightx2v import LightX2VPipeline
        config = out/'lightx2v.json'
        config.write_text(json.dumps(c['lightx2v_config']))
        pipe = LightX2VPipeline(model_path=str(weights), model_cls='wan2.2_moe', task='t2v')
        pipe.create_generator(config_json=str(config))
        pipe.generate(seed=seed, prompt=row['prompt'], negative_prompt='', save_result_path=str(out/'video.mp4'))
        return
    values = dict(SOURCE=str(source), WEIGHTS=str(weights), CONFIG=str(out/'sana.yaml'),
                  PROMPT_FILE=str(out/'prompt.txt'), OUTPUT_DIR=str(out), OUTPUT_FILE=str(out/'video.mp4'), PROMPT=row['prompt'])
    argv = [v.format_map(values) for v in c['manifest_template']['argv']]
    if '--seed' in argv:
        argv[argv.index('--seed')+1] = str(seed)
    if model == 'ltx25':
        sys.argv = argv
        runpy.run_module('ltx_pipelines.distilled', run_name='__main__')
        return
    # Same two deployment fixes used by successful SANA job24782.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from diffusers import AutoencoderKLLTX2Video
    def local_loader(original):
        def load(path, *args, **kwargs):
            if str(path) == 'Efficient-Large-Model/gemma-2-2b-it':
                path = c['text_encoder']
                kwargs['local_files_only'] = True
            return original(path, *args, **kwargs)
        return load
    AutoTokenizer.from_pretrained = staticmethod(local_loader(AutoTokenizer.from_pretrained))
    AutoModelForCausalLM.from_pretrained = staticmethod(local_loader(AutoModelForCausalLM.from_pretrained))
    original_vae_loader = AutoencoderKLLTX2Video.from_pretrained
    def tiled_loader(*args, **kwargs):
        vae = original_vae_loader(*args, **kwargs)
        vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512, tile_sample_min_num_frames=16,
                          tile_sample_stride_height=448, tile_sample_stride_width=448, tile_sample_stride_num_frames=8)
        vae.use_framewise_decoding = True
        return vae
    AutoencoderKLLTX2Video.from_pretrained = staticmethod(tiled_loader)
    (out/'sana.yaml').write_text(c['yaml_template'].replace('{WEIGHTS}', str(weights)))
    (out/'prompt.txt').write_text(row['prompt']+'\n')
    argv = [f'--seed={seed}' if v.startswith('--seed=') else v for v in argv]
    sys.argv = argv
    runpy.run_path(str(source/'inference_video_scripts/inference_sana_video.py'), run_name='__main__')


if __name__ == '__main__':
    main()
