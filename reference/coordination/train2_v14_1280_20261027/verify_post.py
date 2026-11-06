"""Post-commit verification for an applied Train2 item.

usage: verify_post.py <id>

Checks delivered bytes against APPLY_RECEIPT, replays the direct 2048 chain map,
confirms both derived sizes are exact INTER_AREA of the 2048 candidate, confirms
the manifest record, protected source/raw bytes, lock release, and writes
POST_COMMIT.json into the per-item review record directory.
"""
import json, sys, hashlib
from pathlib import Path
import cv2, numpy as np

ROOT = Path('D:/CODE/NR_models')
ART = ROOT / 'artifacts/train2_v14_1280_20261027/items'
RUN = ROOT / 'coordination/current_train_review/record_train2_v14_1280_20261027'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    i = int(sys.argv[1])
    item = ART / str(i)
    r = json.loads((item / 'APPLY_RECEIPT.json').read_text(encoding='utf-8'))
    checks = {
        'Train2/output_f/%d.png' % i: (ROOT / 'Train2/output_f' / f'{i}.png', r['output_f_sha256']),
        'Train2/output_f_1280/%d.png' % i: (ROOT / 'Train2/output_f_1280' / f'{i}.png', r['output_f_1280_sha256']),
        'Train2/input_f/%d.png' % i: (ROOT / 'Train2/input_f' / f'{i}.png', r['input_f_sha256']),
        'Train2/input_f_1280/%d.png' % i: (ROOT / 'Train2/input_f_1280' / f'{i}.png', r['input_f_1280_sha256']),
        'Train2/output/%d.png' % i: (ROOT / 'Train2/output' / f'{i}.png', r['raw_out_sha256']),
    }
    out = {'id': i, 'status': 'POST_COMMIT_VERIFIED', 'apply_receipt': str(item / 'APPLY_RECEIPT.json'),
           'actual_file_sha256_match': {}}
    for name, (p, exp) in checks.items():
        out['actual_file_sha256_match'][name] = bool(p.exists() and sha(p) == exp)
    z = np.load(item / 'source_to_raw_float32.npz')
    key = 'map_xy' if 'map_xy' in z.files else z.files[0]
    raw = cv2.imread(str(item / 'raw_normalized.png'))
    cand = cv2.imread(str(item / 'candidate_output_f.png'))
    rep = cv2.remap(raw, z[key].astype(np.float32), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    c192 = cv2.imread(str(ROOT / 'Train2/output_f' / f'{i}.png'))
    c128 = cv2.imread(str(ROOT / 'Train2/output_f_1280' / f'{i}.png'))
    out['replay_2048_pixel_exact'] = bool(np.array_equal(rep, cand))
    out['exact_1920_resize_from_2048'] = bool(np.array_equal(c192, cv2.resize(cand, (1920, 1080), interpolation=cv2.INTER_AREA)))
    out['exact_1280_resize_from_2048'] = bool(np.array_equal(c128, cv2.resize(cand, (1280, 720), interpolation=cv2.INTER_AREA)))
    out['derived_grid_map_note'] = 'map_1920/map_1280 are geometry references; authoritative replay is the direct 2048 chain map'
    man = json.loads((ROOT / 'Train2/manifest_v14_1920.json').read_text(encoding='utf-8'))
    rec = [x for x in man['records'] if x['id'] == i][0]
    out['manifest'] = {
        'output_f_sha256_match': rec.get('output_f_sha256') == r['output_f_sha256'],
        'output_file_sha256_match_applied_output_bytes': rec.get('output_file_sha256') == r['raw_out_sha256'],
        'output_sha256_semantics': rec.get('output_sha256_semantics'),
    }
    rc = json.loads((item / 'RECEIPT.json').read_text(encoding='utf-8'))
    src = Path(rc['paths']['source']); rawp = Path(rc['paths']['raw'])
    out['protected_source_and_raw_unchanged'] = bool(sha(src) == rc['hashes_before']['source'] and sha(rawp) == rc['hashes_before']['raw'])
    out['lock_released'] = not (ROOT / 'research/regenerated_selection_20261024/integration/.apply.lock').exists()
    out['journal_phases_tail'] = [json.loads(l)['phase'] for l in (item / 'JOURNAL.jsonl').read_text().splitlines()][-3:]
    out['strict_boundary_certification'] = 'UNPROVEN'
    bad = [k for k, v in out['actual_file_sha256_match'].items() if not v]
    ok = (not bad and out['replay_2048_pixel_exact'] and out['exact_1920_resize_from_2048']
          and out['exact_1280_resize_from_2048'] and out['manifest']['output_f_sha256_match']
          and out['manifest']['output_file_sha256_match_applied_output_bytes']
          and out['protected_source_and_raw_unchanged'] and out['lock_released'])
    out['all_checks_pass'] = bool(ok)
    dest = RUN / str(i); dest.mkdir(parents=True, exist_ok=True)
    (dest / 'POST_COMMIT.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'id': i, 'all_checks_pass': out['all_checks_pass'], 'sha_mismatches': bad,
                      'replay': out['replay_2048_pixel_exact'], 'r1920': out['exact_1920_resize_from_2048'],
                      'r1280': out['exact_1280_resize_from_2048'],
                      'manifest_f': out['manifest']['output_f_sha256_match'],
                      'manifest_bytes': out['manifest']['output_file_sha256_match_applied_output_bytes'],
                      'protected': out['protected_source_and_raw_unchanged'], 'lock': out['lock_released']},
                     ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
