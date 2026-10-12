import json, subprocess
from pathlib import Path

def qc(video, config):
    meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate', '-of', 'json', str(video)]))['streams'][0]
    num, den = map(int, meta['avg_frame_rate'].split('/'))
    fps = num / den
    if (int(meta['width']), int(meta['height']), int(meta['nb_read_frames'])) != (config['width'], config['height'], config['num_frames']) or abs(fps-config['fps']) > .02:
        raise RuntimeError('Video dimensions, frames or FPS mismatch: '+str(meta))
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
    return dict(full_decode_ok=True, width=int(meta['width']), height=int(meta['height']), decoded_frames=int(meta['nb_read_frames']), fps=fps)
