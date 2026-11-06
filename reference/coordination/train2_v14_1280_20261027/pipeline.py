"""Train2 full v14 chain (v5 global+flow -> v12 prereg -> v14 residual) applied from
the current 4K raw once, delivered at 1920x1080 with a derived 1280x720 set.

Coordinate contract (decided 2026-10-27, user delegated the crop choice; delivery
retargeted to 1920x1080 primary + scaled 1280x720 on the same day):
  source frame  = Train2/input/<id>.png full frame 2208x1242 (already the centred
                  exact-16:9 crop of Train/input)
  chain canvas  = interior crop [80,45,2128,1197] = 2048x1152 (exact 16:9), keeps a
                  raw margin so the correction never leaves the raw support
  baseline      = normalised current raw 3840x2160 scaled by the nominal affine
                  sx=2208/3840, sy=1242/H with the crop offset, INTER_LANCZOS4
  deliverable   = Train2/output_f/<id>.png at 1920x1080, exactly INTER_AREA(0.9375)
                  of the once-sampled 2048x1152 candidate; Train2/output_f_1280 at
                  1280x720 is exactly INTER_AREA(0.625) of that same candidate (not a
                  resample of the 1920 file).  input_f is the fixed source crop.

The frozen v5 module runs unmodified.  v12 runs as a size-parameterised copy whose
only deviation is reading h,w from the bound images.  v14 residual is the accepted
pipeline residual verbatim.  No fallback, no source warp/fill, no blur masking.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, hashlib, shutil, time, uuid, traceback, argparse, importlib.util
from pathlib import Path
import cv2, numpy as np

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

ROOT = Path('D:/CODE/NR_models')
HERE = ROOT / 'coordination/train2_v14_1280_20261027'
ART = ROOT / 'artifacts/train2_v14_1280_20261027'
WORK = ART / '_work'
SYN = ART / '_synthetic_root'
T2 = ROOT / 'Train2'
NEW = Path('D:/DL/NEW')
SRCMAN = T2 / 'manifest_16_9.json'
MAN = T2 / 'manifest_v14_1920.json'
V5SRC = ROOT / 'research/current_chat_flow_pyramid_v5_20261024/run.py'
V12SRC = HERE / 'v12_size.py'
V12FROZEN = ROOT / 'research/current_chat_flow_boundary_v12_directional_20261024/run.py'
V14SRC = ROOT / 'research/current_chat_foot_local_v14_20261024/multi_image.py'
UTIL = ROOT / 'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py'
EVAL = ROOT / 'experiments/alignment_pair2_20261023/generation/batch26m_current_20261023/evaluation/evaluator.py'
LOCK = ROOT / 'research/regenerated_selection_20261024/integration/.apply.lock'
MODE = 'V14_FULL_PREREG_LOCAL_RESIDUAL_RAW_ONCE_1920_1080_PLUS_1280'
FRAME_WH = (2208, 1242)
CROP = (80, 45, 2128, 1197)
SRC_WH = (2048, 1152)
PRIMARY_WH = (1920, 1080)
SMALL_WH = (1280, 720)
PNG_LEVEL = 3
POLICY_TEXT = {
    'mode': MODE,
    'source_frame': 'Train2/input[45:1197, 80:2128] = 2048x1152 exact 16:9, interior crop mirroring the old [104,24,2104,1272] framing and keeping a margin so the correction never leaves the raw',
    'baseline': 'normalised raw 3840x2160 -> nominal affine raw->frame scale 0.575 with crop offset (-80,-45) -> 2048x1152 INTER_LANCZOS4',
    'chain': 'frozen v5 global+flow+render (unmodified) -> v12 prereg (size-parameterised copy, h,w from bound images) -> accepted v14 local residual verbatim',
    'deliverable': 'candidate_2048 = remap(raw, map_2048) INTER_LINEAR once; candidate_1920 = exact 0.9375 INTER_AREA (primary); candidate_1280 = exact 0.625 INTER_AREA of the same 2048 candidate; input_f = INTER_AREA of the fixed 2048x1152 source crop',
    'maps': 'source(2048x1152)->raw map saved exactly; derived source(1920x1080)->raw and source(1280x720)->raw maps saved for geometry reference',
    'gates': 'added_support_loss==0 at the chain grid and both delivery grids, jacobian_min>=0.05, map finite, pixel-exact replay of the once-sample and of both INTER_AREA steps, raw sampling calls==1, no fallback',
    'explicit_non_claims': 'strict <2px/<1.5px UNPROVEN; v12 is a pre-component and never delivered alone; visual review precedes apply',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp-' + uuid.uuid4().hex)
    with t.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


class SupportGate(RuntimeError):
    pass


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def support(m, w, h):
    return np.isfinite(m).all(2) & (m[..., 0] >= 0) & (m[..., 0] <= w - 1) & (m[..., 1] >= 0) & (m[..., 1] <= h - 1)


def grid_xy(w, h):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.stack([xx, yy], 2)


def jac(m):
    a = m[:-1, :-1]; b = m[:-1, 1:]; c = m[1:, :-1]; d = m[1:, 1:]; out = []
    for dx, dy in ((b - a, c - a), (b - a, d - b), (d - c, c - a), (d - c, d - b)):
        out.append(float(np.min(dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0])))
    return min(out)


def backup(src, dst):
    dst = Path(dst); require(not dst.exists(), 'Backup exists ' + str(dst)); dst.parent.mkdir(parents=True, exist_ok=True)
    with Path(src).open('rb') as a, dst.open('xb') as b:
        shutil.copyfileobj(a, b, 1048576); b.flush(); os.fsync(b.fileno())
    require(sha(src) == sha(dst) and dst.stat().st_nlink == 1, 'Independent backup failed')
    return {'path': str(dst), 'sha256': sha(dst)}


def put_new(src, dst, expect):
    """Create a destination that must not exist yet; CAS-style."""
    dst = Path(dst); require(not dst.exists(), 'Destination exists ' + str(dst)); dst.parent.mkdir(parents=True, exist_ok=True)
    t = dst.with_name(dst.name + '.full-v14-' + uuid.uuid4().hex + '.tmp')
    try:
        with Path(src).open('rb') as a, t.open('xb') as b:
            shutil.copyfileobj(a, b, 1048576); b.flush(); os.fsync(b.fileno())
        require(sha(t) == expect and t.stat().st_nlink == 1, 'Temp validation failed')
        require(not dst.exists(), 'Concurrent creation ' + str(dst))
        os.replace(t, dst); require(sha(dst) == expect, 'Written hash mismatch')
    finally:
        if t.exists():
            t.unlink()


def put_new_bytes(data, dst, expect):
    dst = Path(dst); require(not dst.exists(), 'Destination exists ' + str(dst)); dst.parent.mkdir(parents=True, exist_ok=True)
    t = dst.with_name(dst.name + '.full-v14-' + uuid.uuid4().hex + '.tmp')
    try:
        with t.open('xb') as b:
            b.write(data); b.flush(); os.fsync(b.fileno())
        require(sha(t) == expect and t.stat().st_nlink == 1, 'Temp validation failed')
        require(not dst.exists(), 'Concurrent creation ' + str(dst))
        os.replace(t, dst); require(sha(dst) == expect, 'Written hash mismatch')
    finally:
        if t.exists():
            t.unlink()


def png_bytes(img):
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
    require(ok, 'PNG encode failed')
    return buf.tobytes()


def load_bgr(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(im is not None, 'Cannot read ' + str(path))
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    return im


def raw_normalized(i):
    p = NEW / f'{i}.png'
    require(p.exists(), f'RAW_MISSING {p}')
    im = load_bgr(p)
    info = {'raw_path': str(p), 'raw_sha256': sha(p), 'raw_wh': [int(im.shape[1]), int(im.shape[0])], 'dropped_top_rows': 0}
    if im.shape[0] % 2 == 1:
        d = np.abs(im[0].astype(np.int16) - im[1].astype(np.int16))
        info['top_row_dup'] = {'max': int(d.max()), 'sum': int(d.sum())}
        if int(d.max()) <= 8 and int(d.sum()) <= 4000:
            im = im[1:]
            info['dropped_top_rows'] = 1
    info['normalized_wh'] = [int(im.shape[1]), int(im.shape[0])]
    require(info['normalized_wh'][0] == 3840, 'RAW_WIDTH_UNEXPECTED ' + str(info['normalized_wh']))
    require(info['normalized_wh'][1] in (2160, 2158), 'RAW_HEIGHT_UNEXPECTED ' + str(info['normalized_wh']))
    info['nonuniform_nominal'] = info['normalized_wh'][1] != 2160
    return im, info


def nominal_affine(raw_wh):
    """raw -> source-crop affine: frame scale, then the crop offset."""
    sx = FRAME_WH[0] / raw_wh[0]; sy = FRAME_WH[1] / raw_wh[1]
    return np.array([[sx, 0.0, -CROP[0]], [0.0, sy, -CROP[1]]], np.float64)


def source_crop(frame):
    require([frame.shape[1], frame.shape[0]] == list(FRAME_WH), 'SOURCE_SIZE_UNEXPECTED ' + str(frame.shape))
    return np.ascontiguousarray(frame[CROP[1]:CROP[3], CROP[0]:CROP[2]])


def srcman_index():
    m = read(SRCMAN)
    return {str(r['id']): r for r in m['input']}


def new_manifest():
    return {'operation': 'train2_v14_full_chain_1920_1080_plus_1280', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'source_root': str(T2 / 'input'), 'new_root': str(NEW), 'final_wh': list(PRIMARY_WH),
            'final_wh_small': list(SMALL_WH),
            'pipeline': MODE, 'policy': POLICY_TEXT, 'strict_boundary_certification': 'UNPROVEN',
            'records': []}


def synthetic(i, source_path, baseline, source_sha, attempt):
    ip = SYN / 'Train/input_f' / f'{i}.png'; tp = SYN / 'Train/output_f' / f'{i}.png'
    ip.parent.mkdir(parents=True, exist_ok=True); tp.parent.mkdir(parents=True, exist_ok=True)
    if not ip.exists():
        try:
            os.link(source_path, ip)
        except OSError:
            shutil.copy2(source_path, ip)
    require(sha(ip) == source_sha, 'Synthetic source binding mismatch')
    if tp.exists():
        require(sha(tp) == read(tp.with_suffix('.json'))['sha256'], 'Synthetic baseline drift')
    else:
        cv2.imwrite(str(tp), baseline, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
        write(tp.with_suffix('.json'), {'id': i, 'raw_sha256': None, 'affine': None, 'sha256': sha(tp),
                                        'source': 'current raw once for independent uncorrected baseline; 2208x1242 source frame'})
    return ip, tp


def prereg(i, ip, tp, source_sha, baseline_sha, attempt):
    root = WORK / f'item_{i}_a{attempt}'; v5out = root / 'v5'; v12out = root / 'v12'
    dst = v5out / f'Train_{i}'
    v5out.mkdir(parents=True, exist_ok=True); dst.mkdir(parents=True, exist_ok=True); v12out.mkdir(parents=True, exist_ok=True)
    oldpolicy = read(Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_pyramid_v5/POLICY.json'))
    bound = {str(V5SRC): sha(V5SRC), str(UTIL): sha(UTIL), str(EVAL): sha(EVAL), str(ip): sha(ip), str(tp): sha(tp)}
    policy = {**oldpolicy, 'ids': [i], 'development': None, 'fixed_new_image': [i], 'bound_files': bound,
              'raw_baseline_sha256': baseline_sha, 'old_adapter_output_f_used_for_fit': False,
              'frame_wh': list(SRC_WH), 'profile': 'train2_1920_1080'}
    write(v5out / 'POLICY.json', policy)
    write(v5out / 'POLICY_FROZEN.json', {'sha256': sha(v5out / 'POLICY.json'), 'before_fit': True})
    v5 = load('frozen_v5_' + str(i), V5SRC); v5.ROOT = SYN; v5.OUT = v5out; v5.UTIL = UTIL
    for stage in ('global', 'flow', 'render'):
        t = time.monotonic(); v5.worker(stage, i)
        write(dst / f'{stage}_FULL_CHAIN_RESOURCE.json', {'stage': stage, 'seconds': time.monotonic() - t, 'exit_code': 0})
    util_shadow = SYN / 'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py'
    util_shadow.parent.mkdir(parents=True, exist_ok=True)
    if not util_shadow.exists():
        shutil.copy2(UTIL, util_shadow)
    require(sha(util_shadow) == sha(UTIL), 'Util shadow hash mismatch')
    v12 = load('v12_size_' + str(i), V12SRC); v12.ROOT = SYN; v12.OUT = v12out; v12.BASE = v5out
    t = time.monotonic(); v12.work(i)
    write(v12out / f'Train_{i}' / 'FULL_CHAIN_RESOURCE.json', {'seconds': time.monotonic() - t, 'exit_code': 0})
    fit = read(v12out / f'Train_{i}' / 'FIT_RESULT.json')
    require(fit['input_sha256'] == source_sha and fit['target_sha256'] == baseline_sha, 'Prereg version binding mismatch')
    m = np.load(v12out / f'Train_{i}' / 'composite_source_to_target_float32.npy')
    require(m.shape == (SRC_WH[1], SRC_WH[0], 2) and m.dtype == np.float32 and np.isfinite(m).all(), 'Prereg map contract failed')
    prov = {'kind': 'NEW_FROZEN_V5_GLOBAL_LOCAL_PLUS_V12_PREREG',
            'v5_dir': str(dst), 'v12_dir': str(v12out / f'Train_{i}'),
            'v5_script_sha256': sha(V5SRC), 'v12_script_sha256': sha(V12SRC), 'v12_frozen_reference_sha256': sha(V12FROZEN),
            'v12_change': 'only h,w read from bound images instead of hardcoded 1248,2000',
            'v12_map_sha256': sha(v12out / f'Train_{i}' / 'composite_source_to_target_float32.npy'),
            'synthetic_baseline_sha256': baseline_sha, 'precomponent_only_not_deliverable_v12': True,
            'v12_fit': {k: fit[k] for k in ('jacobian_min', 'fold_check_pass', 'new_support_loss_pixels', 'integration')}}
    return m, v12out / f'Train_{i}' / 'aligned_target_on_source.png', prov


def residual_v14(m, parent, source, item):
    """Accepted full-chain v14 residual, verbatim from v14_full_raw_current_20261024."""
    h, w = m.shape[:2]
    sg = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY); pg = cv2.cvtColor(parent, cv2.COLOR_BGR2GRAY)
    new = m.copy(); flowinfo = []; changed_domain = np.zeros((h, w), bool)
    boxes = [(0, 0, w, h)]
    for j, (a, b, c, d) in enumerate(boxes):
        x0 = max(0, a - 64); y0 = max(0, b - 64); x1 = min(w, c + 64); y1 = min(h, d + 64)
        s = sg[y0:y1, x0:x1].copy(); p = pg[y0:y1, x0:x1].copy()

        def flow(a1, b1):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(0); dis.setPatchSize(8); dis.setPatchStride(3)
            dis.setGradientDescentIterations(25); dis.setVariationalRefinementIterations(5)
            return dis.calc(a1, b1, None)

        f = flow(s, p); r = flow(p, s)
        yy, xx = np.mgrid[:s.shape[0], :s.shape[1]].astype(np.float32)
        rx = xx + f[:, :, 0]; ry = yy + f[:, :, 1]
        br = cv2.remap(r, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cycle = np.linalg.norm(f + br, axis=2); length = np.linalg.norm(f, axis=2)
        grad = cv2.magnitude(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))
        known = (cycle <= .5) & (length <= 4) & (grad >= 4) & (rx >= 2) & (rx < s.shape[1] - 2) & (ry >= 2) & (ry < s.shape[0] - 2)
        confidence = known.astype(np.float32) * np.exp(-cycle ** 2 / (2 * .5 ** 2))
        mass = cv2.GaussianBlur(confidence, (0, 0), 8)
        num = cv2.GaussianBlur(f * confidence[:, :, None], (0, 0), 8)
        field = num / (mass[:, :, None] + .05)
        X = xx + x0; Y = yy + y0
        distance = np.sqrt(np.maximum(np.maximum(a - X, X - (c - 1)), 0) ** 2 + np.maximum(np.maximum(b - Y, Y - (d - 1)), 0) ** 2)
        gate = np.clip(1 - distance / 64, 0, 1); gate = gate * gate * (3 - 2 * gate)
        field *= gate[:, :, None]
        sx = X + field[:, :, 0]; sy = Y + field[:, :, 1]
        bx = np.clip(np.floor(sx).astype(np.int32), 0, w - 2); by = np.clip(np.floor(sy).astype(np.int32), 0, h - 2)
        fx = (sx - bx)[:, :, None]; fy = (sy - by)[:, :, None]
        comp = ((1 - fy) * ((1 - fx) * m[by, bx] + fx * m[by, bx + 1]) + fy * ((1 - fx) * m[by + 1, bx] + fx * m[by + 1, bx + 1])).astype(np.float32)
        active = np.any(field != 0, axis=2)
        local = new[y0:y1, x0:x1]; local[active] = comp[active]
        changed_domain[y0:y1, x0:x1] |= active
        flowinfo.append({'box': [a, b, c, d], 'halo': [x0, y0, x1, y1], 'solver_known': int(known.sum()),
                         'solver_UNKNOWN': int((~known).sum()), 'field_max_px': float(np.linalg.norm(field, axis=2).max())})
        np.savez_compressed(item / 'v14_field_observations.npz', forward=f[::8, ::8], reverse=r[::8, ::8],
                            cycle=cycle[::8, ::8], known=known[::8, ::8], field=field[::8, ::8], gate=gate[::8, ::8])
    require(not np.any((new != m).any(2) & ~changed_domain), 'Residual changed outside local domain')
    return new, flowinfo


def map_to_grid(m, wh):
    """Resample the chain-canvas map onto a delivery grid (pixel-centre exact)."""
    sx, sy = SRC_WH[0] / wh[0], SRC_WH[1] / wh[1]
    yy, xx = np.mgrid[:wh[1], :wh[0]].astype(np.float32)
    gx = (xx + .5) * sx - .5; gy = (yy + .5) * sy - .5
    out = cv2.remap(m, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32)


def panels(i, source, baseline, candidate, sets, item, rawhash):
    from PIL import Image, ImageDraw
    cv2.imwrite(str(item / 'before_blend_2048.png'), cv2.addWeighted(source, .5, baseline, .5, 0))
    cv2.imwrite(str(item / 'after_blend_2048.png'), cv2.addWeighted(source, .5, candidate, .5, 0))
    blends = {}
    for tag, s in sets.items():
        b = cv2.addWeighted(s['src'], .5, s['base'], .5, 0)
        a = cv2.addWeighted(s['src'], .5, s['img'], .5, 0)
        bp, ap = item / f'before_blend_{tag}.png', item / f'after_blend_{tag}.png'
        cv2.imwrite(str(bp), b); cv2.imwrite(str(ap), a)
        blends[tag] = (b, a)
    ev = []
    b, a = blends['1920']
    tw, th = 640, 540
    for n, (x, y) in enumerate(((0, 0), (640, 0), (1280, 0), (0, 540), (640, 540), (1280, 540))):
        panel = Image.new('RGB', (2 * tw, th + 88), '#202020'); d = ImageDraw.Draw(panel)
        d.text((4, 4), f'Train2 {i} raw SHA256 {rawhash}', fill='white')
        d.text((4, 22), f'ROI [{x},{y},{x+tw},{y+th}] native 1:1 alpha=.5 deliverable 1920x1080', fill='white')
        d.text((4, 40), 'LEFT source+UNCORRECTED RAW BASELINE | RIGHT source+FULL V14', fill='white')
        d.text((4, 58), 'No source warp/fill/fallback; strict thresholds UNPROVEN', fill='white')
        for k, im in enumerate((b, a)):
            panel.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).crop((x, y, x + tw, y + th)), (k * tw, 88))
        p = item / f'tile_{n:02d}_native_pair.png'; panel.save(p)
        ev.append({'path': str(p), 'sha256': sha(p), 'crop_roi_xyxy': [x, y, x + tw, y + th],
                   'scale': 1, 'alpha': .5, 'frame': 'deliverable_1920x1080', 'status': 'NOT_VIEWED'})
    return ev


def prepare(i):
    require(shutil.disk_usage(ART if ART.exists() else ROOT).free > 2 * 1024 ** 3, 'DISK_BUDGET')
    item = ART / 'items' / str(i)
    if item.exists():
        require((item / 'ERROR.json').exists() and not (item / 'RECEIPT.json').exists(), 'Item dir exists; evidence is terminal')
        n = len(list(item.glob('ERROR*.json')))
        os.replace(item / 'ERROR.json', item / f'ERROR_attempt{n}.json')
    item.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    idx = srcman_index()
    require(str(i) in idx, 'NO_TRAIN2_SOURCE')
    srcrow = idx[str(i)]
    sp = T2 / 'input' / f'{i}.png'
    require(sp.exists(), 'SOURCE_MISSING ' + str(sp))
    source = source_crop(load_bgr(sp))
    source_sha = sha(sp)
    require(source_sha == srcrow['output_sha256'], 'SOURCE_DRIFT vs manifest_16_9')
    crop_path = item / 'source_crop_2048.png'
    cv2.imwrite(str(crop_path), source, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
    crop_sha = sha(crop_path)
    raw, rinfo = raw_normalized(i)
    A = nominal_affine(rinfo['normalized_wh']); Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
    baseline = cv2.warpAffine(raw, A, SRC_WH, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    baseline_path = item / 'uncorrected_baseline_2048.png'
    cv2.imwrite(str(baseline_path), baseline, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
    baseline_sha = sha(baseline_path)
    attempt = 1
    while (WORK / f'item_{i}_a{attempt}').exists():
        attempt += 1
    ip, tp = synthetic(i, crop_path, baseline, crop_sha, attempt)
    write(tp.with_suffix('.json'), {'id': i, 'raw_sha256': rinfo['raw_sha256'], 'affine': A.tolist(),
                                    'sha256': sha(tp), 'source': 'current raw once for independent uncorrected baseline; 2208x1242 source frame'})
    pre, parent_path, prov = prereg(i, ip, tp, crop_sha, baseline_sha, attempt)
    parent = cv2.imread(str(parent_path))
    m, flowinfo = residual_v14(pre, parent, source, item)
    prov['v14_residual_script_sha256'] = sha(V14SRC)
    prov['v14_residual_roi'] = 'full canvas [0,0,2048,1152], halo clipped by canvas; same frozen DIS/gates/sigma/mass and map composition'
    np.save(item / 'prereg_output_f_map.npy', pre)
    rawmap = (m @ Ai[:, :2].T + Ai[:, 2]).astype(np.float32)
    basemap = (grid_xy(*SRC_WH) @ Ai[:, :2].T + Ai[:, 2]).astype(np.float32)
    sb = support(basemap, *rinfo['normalized_wh']); sn = support(rawmap, *rinfo['normalized_wh'])
    j = jac(rawmap)
    geom = {'baseline_unsupported': int((~sb).sum()), 'candidate_unsupported': int((~sn).sum()),
            'added_support_loss': int((sb & ~sn).sum()), 'saved_map_jacobian_min': j,
            'full_map_finite': bool(np.isfinite(rawmap).all()),
            'changed_map_pixels_vs_uncorrected_raw': int(np.any(rawmap != basemap, 2).sum()),
            'map_domain': 'source_crop_2048x1152_to_normalized_raw_3840x2160'}
    gate = geom['added_support_loss'] == 0 and j >= .05 and geom['full_map_finite']
    candidate = cv2.remap(raw, rawmap[..., 0], rawmap[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cp = item / 'candidate_output_f.png'
    cv2.imwrite(str(cp), candidate, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
    mp = item / 'source_to_raw_float32.npz'
    np.savez_compressed(mp, map_xy=rawmap)
    require(np.array_equal(np.load(mp)['map_xy'], rawmap), 'Saved map replay mismatch')
    require(np.array_equal(cv2.imread(str(cp)), cv2.remap(raw, np.load(mp)['map_xy'][..., 0], np.load(mp)['map_xy'][..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)), 'Candidate map replay mismatch')
    sets = {}
    for tag, wh in (('1920', PRIMARY_WH), ('1280', SMALL_WH)):
        img = cv2.resize(candidate, wh, interpolation=cv2.INTER_AREA)
        ip_ = item / f'candidate_output_f_{tag}.png'
        cv2.imwrite(str(ip_), img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
        dmap = (map_to_grid(m, wh) @ Ai[:, :2].T + Ai[:, 2]).astype(np.float32)
        dmp = item / f'map_{tag}_to_raw_float32.npz'
        np.savez_compressed(dmp, map_xy=dmap)
        require(np.array_equal(np.load(dmp)['map_xy'], dmap), f'Saved {tag} map mismatch')
        basemap_d = (grid_xy(*wh) @ Ai[:, :2].T + Ai[:, 2]).astype(np.float32)
        sb_d = support(basemap_d, *rinfo['normalized_wh']); sn_d = support(dmap, *rinfo['normalized_wh'])
        g = {'baseline_unsupported': int((~sb_d).sum()), 'candidate_unsupported': int((~sn_d).sum()),
             'added_support_loss': int((sb_d & ~sn_d).sum()), 'saved_map_jacobian_min': jac(dmap),
             'full_map_finite': bool(np.isfinite(dmap).all()),
             'changed_map_pixels_vs_uncorrected_raw': int(np.any(dmap != basemap_d, 2).sum()),
             'map_domain': f'deliverable_{wh[0]}x{wh[1]}_grid_to_normalized_raw (derived, not the direct RGB generator)'}
        if not (g['added_support_loss'] == 0 and g['full_map_finite'] and g['saved_map_jacobian_min'] >= .05):
            raise SupportGate(f'{tag} map gate failed: added_support_loss={g["added_support_loss"]} '
                              f'unsupported={g["candidate_unsupported"]} jacobian_min={g["saved_map_jacobian_min"]} '
                              f'finite={g["full_map_finite"]}; no clipping/fill/fallback')
        require(np.array_equal(img, cv2.resize(cv2.imread(str(cp)), wh, interpolation=cv2.INTER_AREA)), f'{tag} resize replay mismatch')
        sets[tag] = {'wh': wh, 'img': img, 'path': ip_, 'sha': sha(ip_), 'geom': g,
                     'map_path': dmp, 'map_sha': sha(dmp),
                     'src': cv2.resize(source, wh, interpolation=cv2.INTER_AREA),
                     'base': cv2.resize(baseline, wh, interpolation=cv2.INTER_AREA)}
    ev = panels(i, source, baseline, candidate, sets, item, rinfo['raw_sha256'])
    require(sha(sp) == source_sha and sha(NEW / f'{i}.png') == rinfo['raw_sha256'], 'Source/raw drift during prepare')
    status = 'PREPARED_NOT_VIEWED' if gate else 'HELD_SUPPORT'
    if geom['changed_map_pixels_vs_uncorrected_raw'] == 0:
        status = 'NO_EFFECT'
    staged = {}
    for tag in ('1920', '1280'):
        b = png_bytes(sets[tag]['src'])
        pb = item / f'input_f_{tag}.png'; pb.write_bytes(b)
        require(sha(pb) == sha_bytes(b), f'Staged input_f_{tag} hash mismatch')
        staged[f'input_f_{tag}'] = {'sha256': sha_bytes(b), 'png_level': PNG_LEVEL}
    raw_bytes = png_bytes(raw)
    (item / 'raw_normalized.png').write_bytes(raw_bytes)
    require(sha(item / 'raw_normalized.png') == sha_bytes(raw_bytes), 'Staged raw hash mismatch')
    staged['raw_normalized'] = {'sha256': sha_bytes(raw_bytes), 'png_level': PNG_LEVEL}
    p1920, p1280 = sets['1920'], sets['1280']
    r = {'id': i, 'status': status,
         'reason': None if gate else 'Full-chain raw support/fold gate failed; no clipping/fill/fallback; not a visible-failure decision.',
         'frame_wh': list(FRAME_WH), 'crop_box_xyxy': list(CROP),
         'source_frame_wh': list(SRC_WH), 'deliverable_wh': list(PRIMARY_WH), 'deliverable_wh_small': list(SMALL_WH),
         'source_crop_sha256': crop_sha,
         'hashes_before': {'source': source_sha, 'raw': rinfo['raw_sha256']},
         'paths': {'source': str(sp), 'raw': str(NEW / f'{i}.png'),
                   'input_out': str(T2 / 'input_f' / f'{i}.png'), 'input_out_small': str(T2 / 'input_f_1280' / f'{i}.png'),
                   'raw_out': str(T2 / 'output' / f'{i}.png'),
                   'output_f': str(T2 / 'output_f' / f'{i}.png'), 'output_f_small': str(T2 / 'output_f_1280' / f'{i}.png')},
         'raw_normalization': rinfo, 'nominal_affine': A.tolist(),
         'pending_output_bytes': staged,
         'pipeline': MODE, 'policy': POLICY_TEXT, 'pipeline_script_sha256': sha(__file__),
         'preregistration': prov, 'v14_flow': flowinfo,
         'geometry': geom, 'geometry_1920': p1920['geom'], 'geometry_1280': p1280['geom'], 'gate_pass': gate,
         'candidate_canvas_path': str(cp), 'candidate_canvas_sha256': sha(cp),
         'candidate_full_path': str(cp), 'candidate_full_sha256': sha(cp),
         'candidate_path': str(p1920['path']), 'candidate_sha256': p1920['sha'],
         'candidate_1280_path': str(p1280['path']), 'candidate_1280_sha256': p1280['sha'],
         'map_path': str(mp), 'map_sha256': sha(mp),
         'map_1920_path': str(p1920['map_path']), 'map_1920_sha256': p1920['map_sha'],
         'map_1280_path': str(p1280['map_path']), 'map_1280_sha256': p1280['map_sha'],
         'map_replay_pixel_exact': True, 'resize_1920_replay_pixel_exact': True, 'resize_1280_replay_pixel_exact': True,
         'raw_sampling_calls_for_candidate': 1,
         'final_resize': {'from': 'candidate_output_f.png 2048x1152 (single raw sample)',
                          'primary': '2048x1152 -> 1920x1080 INTER_AREA (0.9375)',
                          'small': '2048x1152 -> 1280x720 INTER_AREA (0.625), derived from the same candidate, not from the 1920 file',
                          'input_f_resize': 'fixed source 2048x1152 crop -> 1920x1080 and 1280x720 INTER_AREA (no warp)'},
         'baseline_source': 'current raw + nominal affine; old adapter output_f never fitted or resampled',
         'evidence': ev, 'visual_status': 'NOT_VIEWED', 'strict_boundary_certification': 'UNPROVEN',
         'fallback_used': False, 'source_warped': False, 'source_fill': False,
         'seconds': time.monotonic() - t, 'train_writes': False}
    write(item / 'RECEIPT.json', r)
    print(json.dumps({'id': i, 'status': status, 'geometry': geom,
                      'geometry_1920': {k: v for k, v in p1920['geom'].items() if k != 'changed_map_pixels_vs_uncorrected_raw'},
                      'geometry_1280': {k: v for k, v in p1280['geom'].items() if k != 'changed_map_pixels_vs_uncorrected_raw'},
                      'seconds': r['seconds']}), flush=True)
    return r


def review(i, decision, note):
    rp = ART / 'items' / str(i) / 'RECEIPT.json'; r = read(rp)
    require(r['status'] in ('PREPARED_NOT_VIEWED', 'HELD_VISIBLE'), 'Cannot review state')
    require(decision in ('ACCEPT', 'HOLD'), 'Bad decision')
    for e in r['evidence']:
        require(sha(e['path']) == e['sha256'], 'Evidence drift'); e['status'] = 'VIEWED_NATIVE_50_PERCENT'
    r.update(status='REVIEWED_ACCEPT' if decision == 'ACCEPT' else 'HELD_VISIBLE',
             visual_status='NO_OBVIOUS_REGRESSION_IN_VIEWED_ROIS' if decision == 'ACCEPT' else 'FAIL_VISIBLE',
             review_note=note, whole_image_visual_pass=False)
    write(rp, r)


def journal(item, x):
    with (item / 'JOURNAL.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(x, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())


def apply(i):
    item = ART / 'items' / str(i); rp = item / 'RECEIPT.json'; r = read(rp)
    require(r['status'] == 'REVIEWED_ACCEPT' and r['gate_pass'], 'Requires actual ACCEPT and gate pass')
    require(sha(r['paths']['source']) == r['hashes_before']['source'], 'Source drift')
    require(sha(r['paths']['raw']) == r['hashes_before']['raw'], 'Raw drift')
    require(sha(r['candidate_path']) == r['candidate_sha256'], 'Candidate drift')
    require(sha(r['candidate_1280_path']) == r['candidate_1280_sha256'], 'Candidate 1280 drift')
    require(sha(item / 'input_f_1920.png') == r['pending_output_bytes']['input_f_1920']['sha256'], 'Staged input_f_1920 drift')
    require(sha(item / 'input_f_1280.png') == r['pending_output_bytes']['input_f_1280']['sha256'], 'Staged input_f_1280 drift')
    require(sha(item / 'raw_normalized.png') == r['pending_output_bytes']['raw_normalized']['sha256'], 'Staged raw drift')
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(item).encode()); os.close(fd)
    ops = []; done = []
    try:
        require(not (item / 'JOURNAL.jsonl').exists(), 'Prior transaction needs recovery')
        man_exists = MAN.exists()
        mb = None
        if man_exists:
            mb = backup(MAN, item / 'commit_backups/manifest_v14_1920.json')
            m = read(MAN)
        else:
            m = new_manifest()
        require(all(x.get('id') != i for x in m['records']), 'Manifest already has this id')
        rec = {'id': i, 'source_sha256': r['hashes_before']['source'],
               'source_crop_sha256': r['source_crop_sha256'], 'crop_box_xyxy': r['crop_box_xyxy'],
               'raw_path': r['paths']['raw'],
               'raw_sha256': r['hashes_before']['raw'], 'raw_wh': r['raw_normalization']['raw_wh'],
               'normalized_wh': r['raw_normalization']['normalized_wh'],
               'dropped_top_rows': r['raw_normalization']['dropped_top_rows'],
               'input_sha256': r['hashes_before']['source'], 'output_sha256': r['hashes_before']['raw'],
               'output_sha256_semantics': 'raw source file hash (root_dataset_pairs convention); not the applied Train2/output bytes',
               'output_file_sha256': r['pending_output_bytes']['raw_normalized']['sha256'],
               'input_f_sha256': r['pending_output_bytes']['input_f_1920']['sha256'],
               'output_f_sha256': r['candidate_sha256'],
               'f_wh': list(PRIMARY_WH), 'processing_mode': MODE, 'local_warp_used': True,
               'input_f_1280_sha256': r['pending_output_bytes']['input_f_1280']['sha256'],
               'output_f_1280_sha256': r['candidate_1280_sha256'], 'f_wh_1280': list(SMALL_WH),
               'raw_target_sampling_calls': 1, 'source_output_f': r['candidate_path'],
               'active_raw_sampling_map': r['map_path'], 'active_raw_sampling_map_sha256': r['map_sha256'],
               'active_raw_sampling_map_direction': 'source_crop_2048x1152_to_raw_3840x2160',
               'derived_1920_map': r['map_1920_path'], 'derived_1920_map_sha256': r['map_1920_sha256'],
               'derived_1280_map': r['map_1280_path'], 'derived_1280_map_sha256': r['map_1280_sha256'],
               'final_resize': r['final_resize'],
               'source_report': str(rp), 'source_report_sha256': sha(rp),
               'v14_core_applied': True, 'v14_full_chain_applied': True,
               'v14_preregistration': r['preregistration'], 'v14_geometry': r['geometry'],
               'v14_geometry_1920': r['geometry_1920'], 'v14_geometry_1280': r['geometry_1280'],
               'v14_visual_status': r['visual_status'],
               'v14_fallback_used': False,
               'geometry_acceptance': 'VIEWED_ROIS_NO_OBVIOUS_REGRESSION_STRICT_UNPROVEN',
               'training_ready': False, 'needsreview': True,
               'review_reason': 'Full pre-registration + v14 local residual applied from current raw once, delivered 1920x1080 with a derived 1280x720 set; native listed tiles reviewed; strict thresholds UNPROVEN',
               'applied_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        m['records'].append(rec); m['records'].sort(key=lambda x: int(x['id']))
        pm = item / 'manifest.PROPOSED.json'; write(pm, m)
        ops = [{'dest': r['paths']['input_out'], 'kind': 'absent', 'src': str(item / 'input_f_1920.png'), 'new': r['pending_output_bytes']['input_f_1920']['sha256']},
               {'dest': r['paths']['input_out_small'], 'kind': 'absent', 'src': str(item / 'input_f_1280.png'), 'new': r['pending_output_bytes']['input_f_1280']['sha256']},
               {'dest': r['paths']['raw_out'], 'kind': 'absent', 'src': str(item / 'raw_normalized.png'), 'new': r['pending_output_bytes']['raw_normalized']['sha256']},
               {'dest': r['paths']['output_f'], 'kind': 'absent', 'src': r['candidate_path'], 'new': r['candidate_sha256']},
               {'dest': r['paths']['output_f_small'], 'kind': 'absent', 'src': r['candidate_1280_path'], 'new': r['candidate_1280_sha256']},
               {'dest': str(MAN), 'kind': 'replace' if man_exists else 'absent', 'src': str(pm), 'new': sha(pm), 'old': sha(MAN) if man_exists else None, 'backup': mb['path'] if mb else None}]
        write(item / 'PLAN.json', {'id': i, 'operations': ops, 'protected_reads': [r['paths']['source'], r['paths']['raw']]})
        journal(item, {'phase': 'BEGIN', 'id': i})
        for o in ops:
            journal(item, {'phase': 'INTENT', 'dest': o['dest'], 'new': o['new']})
            done.append(o)
            if o['kind'] == 'absent':
                put_new(Path(o['src']), Path(o['dest']), o['new'])
            else:
                require(sha(o['backup']) == o['old'], 'Commit backup mismatch')
                require(sha(o['dest']) == o['old'], 'CAS conflict ' + o['dest'])
                t = Path(o['dest']).with_name(Path(o['dest']).name + '.full-v14-' + uuid.uuid4().hex + '.tmp')
                try:
                    with Path(o['src']).open('rb') as a, t.open('xb') as b:
                        shutil.copyfileobj(a, b, 1048576); b.flush(); os.fsync(b.fileno())
                    require(sha(t) == o['new'] and t.stat().st_nlink == 1, 'Temp validation failed')
                    require(sha(o['dest']) == o['old'], 'Concurrent modification')
                    os.replace(t, Path(o['dest'])); require(sha(o['dest']) == o['new'], 'Replacement mismatch')
                finally:
                    if t.exists():
                        t.unlink()
            journal(item, {'phase': 'WROTE', 'dest': o['dest']})
        require(sha(r['paths']['source']) == r['hashes_before']['source'] and sha(r['paths']['raw']) == r['hashes_before']['raw'], 'Protected file changed')
        journal(item, {'phase': 'COMMIT'})
        result = {'id': i, 'status': 'APPLIED_FULL_V14_1920',
                  'output_f_sha256': sha(r['paths']['output_f']),
                  'output_f_1280_sha256': sha(r['paths']['output_f_small']),
                  'input_f_sha256': sha(r['paths']['input_out']),
                  'input_f_1280_sha256': sha(r['paths']['input_out_small']),
                  'raw_out_sha256': sha(r['paths']['raw_out']),
                  'candidate_sha256': r['candidate_sha256'], 'map_sha256': r['map_sha256'],
                  'receipt_sha256': sha(rp), 'manifest_sha256_after': sha(MAN),
                  'protected_source_unchanged': True, 'protected_raw_unchanged': True,
                  'exact_1920_replay': True, 'exact_1280_replay': True, 'fallback_used': False,
                  'strict_boundary_certification': 'UNPROVEN', 'operations': ops}
        write(item / 'APPLY_RECEIPT.json', result)
        print(json.dumps({'id': i, 'status': result['status']}), flush=True)
    except Exception as e:
        errors = []
        for o in reversed(done):
            try:
                dst = Path(o['dest'])
                if not dst.exists():
                    continue
                cur = sha(dst)
                if cur != o['new']:
                    errors.append({'dest': str(dst), 'reason': 'rollback CAS conflict, left in place', 'sha256': cur})
                    continue
                if o['kind'] == 'absent':
                    dst.unlink()
                else:
                    t = dst.with_name(dst.name + '.rollback-' + uuid.uuid4().hex + '.tmp')
                    with Path(o['backup']).open('rb') as a, t.open('xb') as b:
                        shutil.copyfileobj(a, b, 1048576); b.flush(); os.fsync(b.fileno())
                    require(sha(t) == o['old'], 'Rollback backup mismatch')
                    os.replace(t, dst)
                journal(item, {'phase': 'ROLLED_BACK', 'dest': str(dst)})
            except Exception as err:
                errors.append({'dest': o['dest'], 'error': str(err)})
        write(item / 'APPLY_ERROR.json', {'status': 'ERROR', 'error': str(e), 'rollback_errors': errors})
        raise
    finally:
        LOCK.unlink()


def summary():
    idx = srcman_index()
    raws = sorted([p for p in NEW.glob('*.png')], key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 60)
    rows = []
    for p in raws:
        i = p.stem
        item = ART / 'items' / i
        if not p.stem.isdigit():
            st = 'NON_NUMERIC_ID'
        elif i not in idx:
            st = 'NO_TRAIN2_SOURCE'
        elif (item / 'APPLY_RECEIPT.json').exists():
            st = 'APPLIED_FULL_V14_1920'
        elif (item / 'RECEIPT.json').exists():
            st = read(item / 'RECEIPT.json')['status']
        elif (item / 'HELD_SUPPORT.json').exists():
            st = 'HELD_SUPPORT'
        elif (item / 'ERROR.json').exists():
            st = 'ERROR'
        else:
            st = 'PENDING_FULL_CHAIN'
        rows.append({'id': i, 'status': st})
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    out = {'counts': counts, 'new_count': len(raws), 'train2_sources': len(idx),
           'applied_in_manifest': len(read(MAN)['records']) if MAN.exists() else 0,
           'items': rows, 'strict_boundary_certification': 'UNPROVEN',
           'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write(HERE / 'STATUS.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'items'}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'review', 'apply', 'summary'])
    p.add_argument('--ids', nargs='+', type=int)
    p.add_argument('--decision'); p.add_argument('--note', default='')
    a = p.parse_args()
    if a.command == 'summary':
        summary(); return
    require(a.ids and len(a.ids) <= 20, 'Batch size must be 1..20')
    for i in a.ids:
        try:
            if a.command == 'prepare':
                prepare(i)
            elif a.command == 'review':
                review(i, a.decision, a.note)
            else:
                apply(i)
        except SupportGate as e:
            item = ART / 'items' / str(i); item.mkdir(parents=True, exist_ok=True)
            write(item / 'HELD_SUPPORT.json', {'id': i, 'status': 'HELD_SUPPORT', 'reason': str(e),
                                               'classification': 'SUPPORT_DOMAIN_NOT_VISUAL',
                                               'action': 'no candidate delivered; no clipping/fill/fallback; needs a raw whose framing keeps the chain map inside the source',
                                               'traceback': traceback.format_exc()})
            print(json.dumps({'id': i, 'status': 'HELD_SUPPORT', 'reason': str(e)}, ensure_ascii=False), flush=True)
        except Exception as e:
            item = ART / 'items' / str(i)
            if (item / 'RECEIPT.json').exists():
                print(json.dumps({'id': i, 'status': 'TERMINAL_RECEIPT_RETAINED', 'error': str(e)}, ensure_ascii=False), flush=True)
            else:
                item.mkdir(parents=True, exist_ok=True)
                write(item / 'ERROR.json', {'id': i, 'status': 'ERROR', 'error': str(e), 'traceback': traceback.format_exc()})
                print(json.dumps({'id': i, 'status': 'ERROR', 'error': str(e)}, ensure_ascii=False), flush=True)
    summary()


if __name__ == '__main__':
    main()
