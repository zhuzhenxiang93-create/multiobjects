from pathlib import Path
import json,subprocess,sys,time,traceback
root=Path("/data/zzhu126/multi_object_benchmark")
d=root/"generations/two_models150_accelerated_v1"
source=root/"runtime/source_repos/HunyuanVideo-1.5"
sys.path.insert(0,str(root/"generations/t2v_crossmodel_pilot20_v1"))
from run import qc
from sageattention import sageattn
failed=[]
for row in json.loads((d/"prompts150.json").read_text()):
    out=d/"outputs"/"hunyuan"/row["prompt_id"];out.mkdir(parents=True,exist_ok=True)
    video=out/"video.mp4"
    result=out/"result.json"
    if result.exists() and json.loads(result.read_text()).get("status")=="complete": continue
    cmd=[str(Path(sys.executable).parent/"torchrun"),"--standalone","--nproc_per_node=1",str(source/"generate.py"),"--prompt",row["prompt"],"--negative_prompt","","--image_path","none","--resolution","720p","--aspect_ratio","16:9","--video_length","121","--num_inference_steps","50","--seed","42","--model_path",str(root/"runtime/models/HunyuanVideo-1.5-native"),"--output_path",str(video),"--rewrite","false","--sr","false","--cfg_distilled","false","--enable_step_distill","false","--sparse_attn","false","--use_sageattn","true","--enable_cache","true","--cache_type","deepcache","--total_steps","50","--offloading","false","--group_offloading","false","--overlap_group_offloading","false"]
    record=dict(status="running",prompt_id=row["prompt_id"],prompt=row["prompt"],command=cmd,started_unix=time.time())
    result.write_text(json.dumps(record,indent=2))
    try:
        with (out/"generation.log").open("w") as log:
            subprocess.run(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        record.update(status="complete",qc=qc(video,dict(height=720,width=1280,num_frames=121,fps=24)),video=str(video))
    except Exception:
        record.update(status="failed",error=traceback.format_exc());failed.append(row["prompt_id"])
        raise
    finally:
        record["elapsed_seconds"]=time.time()-record["started_unix"]
        result.write_text(json.dumps(record,indent=2))
        print(row["prompt_id"],record["status"],record["elapsed_seconds"],flush=True)
(d/"hunyuan_summary.json").write_text(json.dumps(dict(total=150,failed=failed),indent=2))
if failed: raise SystemExit(1)
