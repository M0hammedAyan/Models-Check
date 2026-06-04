"""Batch test all depth models: load and run one forward pass."""
import sys, os, time, logging, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
import torch
import numpy as np
import cv2

from models import ModelFactory

device = 'cpu'
results = {}

for model_name in ModelFactory.list_models():
    print(f"\n{'='*60}")
    print(f"Testing {model_name}...")
    print(f"{'='*60}")
    sys.stdout.flush()
    try:
        t0 = time.time()
        model = ModelFactory.create_model(model_name, device=device, use_fp16=False)
        t_load = time.time() - t0

        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy = np.ascontiguousarray(dummy)

        t1 = time.time()
        depth = model.predict(dummy)
        t_pred = time.time() - t1

        results[model_name] = {
            'status': 'PASS',
            'load_time': round(t_load, 1),
            'pred_time': round(t_pred, 1),
            'depth_shape': depth.shape,
            'depth_range': (float(depth.min()), float(depth.max())),
        }
        print(f"  PASS | load={t_load:.1f}s pred={t_pred:.1f}s shape={depth.shape} range=[{depth.min():.3f},{depth.max():.3f}]")

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        tb = traceback.format_exc()
        # Print just the last line of the error
        err_msg = str(e).split('\n')[-1][:200]
        results[model_name] = {'status': 'FAIL', 'error': err_msg}
        print(f"  FAIL | {err_msg}")
        # Print full traceback in a compact way
        for line in tb.strip().split('\n')[-5:]:
            print(f"  > {line.strip()}")

    sys.stdout.flush()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
passed = sum(1 for v in results.values() if v['status'] == 'PASS')
failed = sum(1 for v in results.values() if v['status'] == 'FAIL')
for name, res in results.items():
    status = res['status']
    detail = f" ({res['load_time']}s load, {res['pred_time']}s pred, shape={res['depth_shape']})" if status == 'PASS' else f" ({res['error']})"
    print(f"  {status}: {name}{detail}")
print(f"\nPassed: {passed}/{len(results)}, Failed: {failed}/{len(results)}")
