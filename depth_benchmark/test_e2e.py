import torch, sys, os, warnings, logging
os.environ['XFORMERS_DISABLED'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings('ignore')
sys.path.insert(0, 'D:/MY_Projects/Models-Check/depth_benchmark')

from common import BenchmarkConfig
from benchmark import BenchmarkPipeline
import yaml

conf = {
    'data': {
        'video_path': 'D:/MY_Projects/Models-Check/depth_benchmark/upstream_repos/Depth-Anything/assets/examples_video/davis_dolphins.mp4',
        'bag_path': None,
        'output_fps': 5,
        'resolution': [480, 640]
    },
    'processing': {
        'apply_temporal_refinement': False,
        'batch_size': 1,
        'use_fp16': False,
    },
    'pose': {
        'enable_pose': False,
    },
    'models': {
        'midas': {'model_type': 'midas', 'input_size': [384, 384], 'use_cuda': True, 'mixed_precision': False},
        'dpt_large': {'model_type': 'dpt_large', 'input_size': [384, 384], 'use_cuda': True, 'mixed_precision': False},
    },
    'output_dir': './outputs/test_run',
    'device': 'cuda',
    'num_workers': 0,
    'seed': 42,
}

os.makedirs('./outputs/test_run', exist_ok=True)
with open('./outputs/test_run/config_temp.yaml', 'w') as f:
    yaml.dump(conf, f)

config = BenchmarkConfig.from_yaml('./outputs/test_run/config_temp.yaml')
pipeline = BenchmarkPipeline(config)

print('Loading models...')
pipeline.load_models()
print('Loaded: %s' % list(pipeline.models.keys()))

print('Processing video...')
results, rgb_frames = pipeline.process_video(config.data.video_path)
print('Processed %d frames' % len(rgb_frames))

for model_name in results['models']:
    depths = results['models'][model_name]['depths']
    print('%s: %d depth maps, shape=%s' % (model_name, len(depths), str(depths[0].shape)))

print('SUCCESS: End-to-end pipeline works!')
