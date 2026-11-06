"""Size-parameterised copy of the frozen v12 directional pre-registration.

Source: research/current_chat_flow_boundary_v12_directional_20261024/run.py
Only change: the hardcoded `h,w=1248,2000` is read from the input images instead.
Every numeric gate, sigma, kernel, squaring count, boundary rule, SVF and alpha
is unchanged.  This is a pre-component and is never delivered on its own.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, json, time, hashlib, importlib.util, subprocess
ROOT = Path('D:/CODE/NR_models')
HERE = Path(__file__).resolve().parent
OUT = Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_boundary_v12_directional')
BASE = OUT.parent / 'research_current_chat_flow_v5_hardcases'


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def work(ident):
    import cv2, numpy as np, psutil
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:2])
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    spec = importlib.util.spec_from_file_location('utils', ROOT / 'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py')
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    dst = OUT / f'Train_{ident}'
    dst.mkdir(parents=True)
    old = BASE / f'Train_{ident}'
    A = np.load(old / 'global.npz')['affine']
    z = np.load(old / 'correspondences_for_26m.npz')
    # Only deviation from the frozen script: frame size from the bound images.
    src = cv2.imread(str(ROOT / f'Train/input_f/{ident}.png'))
    h, w = src.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack([xx, yy], -1).astype(np.float32)
    del xx, yy
    xy = z['source_xy']
    k = z['known']
    ix, iy = xy[k].astype(int).T
    weights = z['weight'][k]
    mass = np.zeros((h, w), np.float32)
    num = np.zeros((h, w, 2), np.float32)
    seed = np.ones((h, w), np.uint8)
    mass[iy, ix] = weights
    num[iy, ix] = z['residual_source_px'][k] * weights[:, None]
    seed[iy, ix] = 0
    mass = cv2.GaussianBlur(mass, (385, 385), 32, borderType=cv2.BORDER_REFLECT_101)
    num = cv2.GaussianBlur(num, (385, 385), 32, borderType=cv2.BORDER_REFLECT_101)
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    gate = np.clip((128 - distance) / 64, 0, 1)
    gate = gate * gate * (3 - 2 * gate)
    velocity = (num / (mass[:, :, None] + 1e-7) * gate[:, :, None]).astype(np.float32)
    del mass, num, distance, gate
    base_target = grid @ A[:, :2].T + A[:, 2]
    target_velocity = velocity @ A[:, :2].T
    for axis, limit in ((0, w - 1), (1, h - 1)):
        inward_distance = np.where(target_velocity[:, :, axis] < 0, base_target[:, :, axis], limit - base_target[:, :, axis])
        fraction = np.clip(inward_distance / 64, 0, 1)
        target_velocity[:, :, axis] *= fraction * fraction * (3 - 2 * fraction)
    velocity = (target_velocity @ np.linalg.inv(A[:, :2]).T).astype(np.float32)
    del target_velocity, base_target, inward_distance, fraction
    displacement = velocity / 64
    integration = []
    for i in range(6):
        g = grid + displacement
        from scipy.ndimage import map_coordinates
        sampled = np.stack([map_coordinates(displacement[:, :, c], [g[:, :, 1], g[:, :, 0]], order=1, mode='nearest', prefilter=False) for c in range(2)], -1)
        displacement += sampled
        integration.append({'squaring': i + 1, 'max_displacement_px': float(np.linalg.norm(displacement, axis=2).max())})
    local = grid + displacement
    mapping = (local @ A[:, :2].T + A[:, 2]).astype(np.float32)
    base = (grid @ A[:, :2].T + A[:, 2]).astype(np.float32)

    def valid(m):
        return np.isfinite(m).all(2) & (m[:, :, 0] >= 0) & (m[:, :, 0] <= w - 1) & (m[:, :, 1] >= 0) & (m[:, :, 1] <= h - 1)

    good = valid(mapping)
    gb = valid(base)
    det = u.jacobian_min(mapping)
    target = cv2.imread(str(ROOT / f'Train/output_f/{ident}.png'))
    src = cv2.imread(str(ROOT / f'Train/input_f/{ident}.png'))
    candidate = cv2.remap(target, mapping[:, :, 0], mapping[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    np.save(dst / 'composite_source_to_target_float32.npy', mapping)
    cv2.imwrite(str(dst / 'aligned_target_on_source.png'), candidate)
    report = {'id': ident, 'n_squarings': 6, 'alpha': 1, 'integration': integration, 'jacobian_min': det,
              'fold_check_pass': bool(det >= .05), 'full_support_loss': int((~good).sum()),
              'baseline_support_loss': int((~gb).sum()), 'new_support_loss_pixels': int((gb & ~good).sum()),
              'map_sha256': sha(dst / 'composite_source_to_target_float32.npy'),
              'png_sha256': sha(dst / 'aligned_target_on_source.png'),
              'input_sha256': sha(ROOT / f'Train/input_f/{ident}.png'),
              'target_sha256': sha(ROOT / f'Train/output_f/{ident}.png'),
              'no_eval_used_for_fit': True, 'gt': False, 'global_injectivity_proven': False,
              'character_edges_proven': False,
              'render': 'original color target once INTER_LINEAR, no filtering or sharpening',
              'frame_wh': [w, h], 'frame_wh_source': 'read from input_f (size-parameterised copy of frozen v12)'}
    write(dst / 'FIT_RESULT.json', report)
    print(json.dumps(report))
