from pathlib import Path
import json,time,traceback,sys
from lightx2v import LightX2VPipeline
d=Path("/data/zzhu126/multi_object_benchmark/generations/two_models150_accelerated_v1")
sys.path.insert(0,"/data/zzhu126/multi_object_benchmark/generations/t2v_crossmodel_pilot20_v1")
from run import qc
pipe=LightX2VPipeline(model_path="/data/zzhu126/multi_object_benchmark/runtime/models/current_video_generators/Wan2.2-T2V-A14B",model_cls="wan2.2_moe",task="t2v")
pipe.create_generator(config_json="/data/zzhu126/multi_object_benchmark/generations/wan22_sage_lightx2v_v1/config.json")
for row in json.loads((d/"prompts150.json").read_text()):
    out=d/"outputs"/"wan"/row["prompt_id"];out.mkdir(parents=True,exist_ok=True)
    result=out/"result.json";video=out/"video.mp4"
    if result.exists() and json.loads(result.read_text()).get("status")=="complete":continue
    r=dict(status="running",prompt_id=row["prompt_id"],prompt=row["prompt"],seed=42,started_unix=time.time())
    result.write_text(json.dumps(r,indent=2))
    try:
        pipe.generate(seed=42,prompt=row["prompt"],negative_prompt="",save_result_path=str(video))
        r.update(status="complete",qc=qc(video,dict(height=720,width=1280,num_frames=81,fps=16)),video=str(video))
    except Exception:
        r.update(status="failed",error=traceback.format_exc());raise
    finally:
        r["elapsed_seconds"]=time.time()-r["started_unix"];result.write_text(json.dumps(r,indent=2))
        print(row["prompt_id"],r["status"],r["elapsed_seconds"],flush=True)
