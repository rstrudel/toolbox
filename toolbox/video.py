import skvideo
skvideo.setFFmpegPath('/sequoia/data1/rstrudel/miniconda3/envs/bullet_cpu/bin/')
import skvideo.io as skv
import numpy as np

def write_video(frames, path):
    skv.vwrite(path, np.array(frames).astype(np.uint8))
