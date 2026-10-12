import os,sys,json,time,runpy,pathlib,threading,subprocess,traceback
import torch
base=pathlib.Path(sys.argv[1]); kind=sys.argv[2]
manifest=json.loads((base/'acceptance_config.json').read_text())
assert (base/'install.ok').exists() and (base/'download.ok').exists(), 'Deployment not ready'
assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one allocated CUDA GPU required'
# Deployment-only path resolution; model computations stay in official source.
if kind=='sana':
 from transformers import AutoTokenizer,AutoModelForCausalLM
 snapshot='/data/zzhu126/cache/huggingface/hub/models--Efficient-Large-Model--gemma-2-2b-it/snapshots/569d9809d0c8b6722d4d31b5a77a2ec7a400650a'
 def local_loader(original):
  def load(path,*args,**kwargs):
   if str(path)=='Efficient-Large-Model/gemma-2-2b-it':
    path=snapshot;kwargs['local_files_only']=True
   return original(path,*args,**kwargs)
  return load
 AutoTokenizer.from_pretrained=staticmethod(local_loader(AutoTokenizer.from_pretrained))
 AutoModelForCausalLM.from_pretrained=staticmethod(local_loader(AutoModelForCausalLM.from_pretrained))


# Use Diffusers' built-in temporal/spatial tiled decoding to bound activation memory.
if kind=='sana':
 from diffusers import AutoencoderKLLTX2Video
 original_vae_loader=AutoencoderKLLTX2Video.from_pretrained
 def tiled_vae_loader(*args,**kwargs):
  instance=original_vae_loader(*args,**kwargs)
  instance.enable_tiling(tile_sample_min_height=512,tile_sample_min_width=512,tile_sample_min_num_frames=16,tile_sample_stride_height=448,tile_sample_stride_width=448,tile_sample_stride_num_frames=8)
  instance.use_framewise_decoding=True
  return instance
 AutoencoderKLLTX2Video.from_pretrained=staticmethod(tiled_vae_loader)

out=pathlib.Path(manifest['output_dir']); out.mkdir(parents=True,exist_ok=True)
assert not list(out.rglob('*.mp4')), 'Refuse duplicate acceptance generation'
record={'job_id':os.getenv('SLURM_JOB_ID'),'dependency':manifest['dependency'],'prompt':manifest['prompt'],'seed':42,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'config':manifest}
# Exercise CUDA in the actual allocated job.
a=torch.ones((32,32),device='cuda'); assert (a@a).sum().item()==32768; del a
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
stop=threading.Event(); observed=[]
def sample():
 while not stop.wait(1):
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
   for line in raw.splitlines():
    pid,mem=line.split(',')
    if int(pid)==os.getpid(): observed.append(int(mem))
  except Exception: pass
thread=threading.Thread(target=sample,daemon=True); thread.start()
start=time.monotonic(); success=False
try:
 sys.argv=manifest['argv']
 if kind=='sana': runpy.run_path(manifest['entrypoint'],run_name='__main__')
 else: runpy.run_module('ltx_pipelines.distilled',run_name='__main__')
 success=True
except BaseException:
 record['error']=traceback.format_exc(); raise
finally:
 stop.set();thread.join()
 record.update(generation_success=success,generation_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_gpu_mib=max(observed,default=None),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 (base/'acceptance_runtime.json').write_text(json.dumps(record,indent=2))
import av
videos=list(out.rglob('*.mp4'))
assert len(videos)==1, f'Expected exactly one video, found {len(videos)}'
with av.open(str(videos[0])) as c:
 s=c.streams.video[0]; width=s.codec_context.width;height=s.codec_context.height;fps=float(s.average_rate)
 count=sum(1 for frame in c.decode(s))
result=dict(path=str(videos[0]),width=width,height=height,frames=count,fps=fps,complete_decode=True)
(base/'decode_result.json').write_text(json.dumps(result,indent=2))
assert (width,height,count)==tuple(manifest['expected']), result
assert abs(fps-24)<0.001, result
(base/'acceptance.ok').touch()
print(json.dumps(result),flush=True)

