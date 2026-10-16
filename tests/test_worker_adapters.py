"""Validate native argument adapters and SANA deployment fixes without GPUs."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import worker


class Adapters(unittest.TestCase):
    def invoke(self, model):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = dict(model=model, seed=7, source='/data/source', weights='/data/weights',
                          text_encoder='/data/gemma', lightx2v_config={},
                          yaml_template=(ROOT/'configs/sana_native.yaml').read_text())
            if model in ['sana', 'ltx25']:
                config['manifest_template'] = json.loads((ROOT/'configs'/f'{model}_native.json').read_text())
            request = out/'request.json'
            request.write_text(json.dumps(dict(config=config, row=dict(prompt='Two horses.'), output=str(out))))
            with patch.object(sys, 'argv', ['worker.py', str(request)]):
                worker.main()
                return list(sys.argv), out.joinpath('sana.yaml').read_text() if (out/'sana.yaml').exists() else None

    def test_hunyuan_uses_native_source_and_seed(self):
        with patch.object(worker.subprocess, 'run') as run:
            self.invoke('hunyuan')
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--seed')+1], '7')
            self.assertIn('/data/source/generate.py', command)

    def test_ltx_seed_and_weight_template(self):
        with patch.object(worker.runpy, 'run_module') as run:
            argv, _ = self.invoke('ltx25')
            self.assertEqual(argv[argv.index('--seed')+1], '7')
            self.assertIn('/data/weights/vae/ltx-2.5-video-vae-bf16.safetensors', argv)
            self.assertEqual(run.call_args.args[0], 'ltx_pipelines.distilled')

    def test_sana_local_encoder_and_tiling_are_applied(self):
        tokenizer = type('Tokenizer', (), {'from_pretrained': staticmethod(MagicMock(return_value='tokenizer'))})
        encoder = type('Encoder', (), {'from_pretrained': staticmethod(MagicMock(return_value='encoder'))})
        vae = MagicMock()
        vae_class = type('VAE', (), {'from_pretrained': staticmethod(MagicMock(return_value=vae))})
        modules = {'transformers': types.SimpleNamespace(AutoTokenizer=tokenizer, AutoModelForCausalLM=encoder),
                   'diffusers': types.SimpleNamespace(AutoencoderKLLTX2Video=vae_class)}
        def run_native(path, run_name):
            self.assertIn('--seed=7', sys.argv)
            self.assertEqual(tokenizer.from_pretrained('Efficient-Large-Model/gemma-2-2b-it'), 'tokenizer')
            loaded = vae_class.from_pretrained('/data/weights')
            self.assertTrue(loaded.use_framewise_decoding)
            loaded.enable_tiling.assert_called_once()
        with patch.dict(sys.modules, modules), patch.object(worker.runpy, 'run_path', side_effect=run_native):
            argv, yaml = self.invoke('sana')
            self.assertNotIn('{WEIGHTS}', yaml)
            self.assertIn('/data/weights/LTX-2.3-Diffusers', yaml)

    def test_wan_uses_existing_accepted_pipeline_interface(self):
        pipeline = MagicMock()
        with patch.dict(sys.modules, {'lightx2v': types.SimpleNamespace(LightX2VPipeline=pipeline)}):
            self.invoke('wan')
            self.assertEqual(pipeline.call_args.kwargs['model_cls'], 'wan2.2_moe')
            self.assertEqual(pipeline.return_value.generate.call_args.kwargs['seed'], 7)


if __name__ == '__main__':
    unittest.main()
