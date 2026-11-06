"""Portable entry point for the complete, manually reviewed V14 workflow."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Isolated dataset directory; never point at a production dataset')
    parser.add_argument('command', choices=['init', 'prepare', 'review', 'apply', 'verify', 'summary'])
    parser.add_argument('--ids', nargs='+', type=int)
    parser.add_argument('--decision', choices=['ACCEPT', 'HOLD'])
    parser.add_argument('--note', default='')
    parser.add_argument('--viewed-all-evidence', action='store_true',
                        help='Attest that you actually viewed every native overlay in the receipt')
    args = parser.parse_args()
    root = args.workspace.resolve()
    if root == CODE or CODE.is_relative_to(root):
        parser.error('Choose a separate workspace outside the source repository')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['V14_WORKSPACE'] = str(root)
    spec = importlib.util.spec_from_file_location('portable_v14', CODE / 'portable/pipeline.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    if args.command == 'init':
        rows = []
        for path in sorted((root / 'Train2/input').glob('*.png')):
            if not path.stem.isdecimal():
                continue
            image = p.load_bgr(path)
            p.require(tuple(image.shape[1::-1]) == p.FRAME_WH, f'Expected 2208x1242: {path}')
            rows.append({'id': int(path.stem), 'output_sha256': sha(path)})
        p.require(rows, 'Place numbered PNGs in Train2/input first')
        p.require(not p.SRCMAN.exists(), 'Source manifest exists; do not rebind existing evidence')
        p.write(p.SRCMAN, {'input': rows, 'source_frame_wh': list(p.FRAME_WH)})
        (root / 'raw').mkdir(exist_ok=True)
        print(json.dumps({'initialized': len(rows), 'workspace': str(root)}))
        return
    if args.command == 'summary':
        p.summary()
        return
    if not args.ids or not 1 <= len(args.ids) <= 20 or len(set(args.ids)) != len(args.ids):
        parser.error('Provide 1..20 unique IDs')
    if args.command == 'review' and (not args.decision or not args.note.strip() or not args.viewed_all_evidence):
        parser.error('Review requires --decision, a concrete --note and --viewed-all-evidence')
    for ident in args.ids:
        if args.command == 'prepare':
            p.prepare(ident)
        elif args.command == 'review':
            p.review(ident, args.decision, args.note)
        elif args.command == 'apply':
            p.apply(ident)
        else:
            verify(p, ident)


def verify(p, ident):
    import cv2
    import numpy as np
    item = p.ART / 'items' / str(ident)
    receipt = p.read(item / 'RECEIPT.json')
    applied = p.read(item / 'APPLY_RECEIPT.json')
    for key, digest in [('source', 'source'), ('raw', 'raw')]:
        p.require(sha(Path(receipt['paths'][key])) == receipt['hashes_before'][digest], key + ' changed')
    for key, digest in [('output_f', 'output_f_sha256'), ('output_f_small', 'output_f_1280_sha256'),
                        ('input_out', 'input_f_sha256'), ('input_out_small', 'input_f_1280_sha256'),
                        ('raw_out', 'raw_out_sha256')]:
        p.require(sha(Path(receipt['paths'][key])) == applied[digest], key + ' mismatch')
    p.require(sha(Path(receipt['map_path'])) == receipt['map_sha256'], 'Map hash mismatch')
    mapping = np.load(receipt['map_path'])['map_xy']
    raw = p.load_bgr(item / 'raw_normalized.png')
    candidate = cv2.remap(raw, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    p.require(np.array_equal(candidate, p.load_bgr(Path(receipt['candidate_canvas_path']))), 'Raw map replay mismatch')
    for key, wh in [('output_f', p.PRIMARY_WH), ('output_f_small', p.SMALL_WH)]:
        p.require(np.array_equal(cv2.resize(candidate, wh, interpolation=cv2.INTER_AREA),
                                p.load_bgr(Path(receipt['paths'][key]))), 'Resize replay mismatch')
    record = next(r for r in p.read(p.MAN)['records'] if r['id'] == ident)
    p.require(record['output_f_sha256'] == applied['output_f_sha256'], 'Manifest mismatch')
    p.require(not p.LOCK.exists(), 'Writer lock present')
    print(json.dumps({'id': ident, 'verified': True, 'strict_boundary_certification': 'UNPROVEN'}))


if __name__ == '__main__':
    main()
