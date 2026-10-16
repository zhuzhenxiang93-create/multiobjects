import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

class Adapter(unittest.TestCase):
    def test_preserves_wan_prompt_seed_and_model(self):
        spec = importlib.util.spec_from_file_location('wan_worker', Path(__file__).resolve().parents[1]/'scripts/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            request = out/'request.json'
            request.write_text(json.dumps(dict(config=dict(model='wan', seed=42, source=folder,
                weights='/weights', lightx2v_config={'infer_steps':40}),
                row={'prompt':'Two horses.'}, output=folder)))
            pipeline = MagicMock()
            with patch.dict(sys.modules, {'lightx2v':types.SimpleNamespace(LightX2VPipeline=pipeline)}), patch.object(sys,'argv',['worker.py',str(request)]):
                worker.main()
            pipeline.assert_called_once_with(model_path='/weights',model_cls='wan2.2_moe',task='t2v')
            args = pipeline.return_value.generate.call_args.kwargs
            self.assertEqual((args['prompt'],args['seed']),('Two horses.',42))
            self.assertEqual(json.loads((out/'lightx2v.json').read_text()),{'infer_steps':40})

if __name__=='__main__':
    unittest.main()
