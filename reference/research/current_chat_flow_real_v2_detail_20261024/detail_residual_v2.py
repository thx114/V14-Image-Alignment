import os
import json
import hashlib
from pathlib import Path

os.environ.update({k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

ROOT = Path("D:/CODE/NR_models")
HERE = ROOT / "research/current_chat_flow_real_v2_detail_20261024"
ART = Path("C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_real_v2_detail")
OLD = Path("C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261023/research_current_chat_flow_real_v1/Train_2_retry1")
MAN = ROOT / "experiments/alignment_pair2_20261023/generation/batch26m_rollout_v2_20261023/parent-evaluation/current_manifest.json"
EVAL = ROOT / "experiments/alignment_pair2_20261023/generation/batch26m_current_20261023/evaluation/evaluator.py"

POLICY = {
    "version": "FLOW_REAL_DETAIL_V2_20261024",
    "input_id": 2,
    "source": "Train/input_f/2.png",
    "target": "Train/output_f/2.png",
    "reuse_existing_flow_observation": True,
    "fit_only": True,
    "detail_residual_max_px": 2.0,
    "detail_smoothing_sigma_px": 8.0,
    "detail_alpha_fixed": 1.0,
    "minimum_cell_det": 0.05,
    "fixed_query_step_px": 32,
    "fixed_query_margin_px": 64,
    "fixed_query_offset_px": [16, 16],
    "no_eval_for_policy_or_alpha": True,
    "gt_used": False,
    "support": "full-raster loss and fixed ROI loss both reported; no coordinate clipping",
    "render": "target[composite_source_to_target_map]",
    "resources": {"cpu_threads": 2, "gpu_calls": 0, "child_seconds_max": 85, "memory_commit_mib_max": 900},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def affine(p, q, w):
    c = np.array([p[:, 0].max(), p[:, 1].max()]) / 2.0
    s = max(p[:, 0].max(), p[:, 1].max())
    X = np.c_[(p - c) / s, np.ones(len(p))]
    y = q - p
    sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], y * sw[:, None], rcond=None)[0]
    for _ in range(6):
        e = np.linalg.norm(X @ b - y, axis=1)
        sw = np.sqrt(w * np.minimum(1.0, 1.0 / np.maximum(e, 1e-9)))
        b = np.linalg.lstsq(X * sw[:, None], y * sw[:, None], rcond=None)[0]
    return np.c_[np.eye(2) + b[:2].T / s, b[2] - c @ b[:2] / s]


def jacobian_min(mapping):
    a, b = mapping[:-1, :-1], mapping[:-1, 1:]
    c, d = mapping[1:, :-1], mapping[1:, 1:]
    vals = []
    for dx, dy in ((b - a, c - a), (b - a, d - b), (d - c, c - a), (d - c, d - b)):
        vals.append(float((dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0]).min()))
    return min(vals)


def query_grid(width, height):
    return np.array([[x + 16.0, y + 16.0] for y in range(64, height - 64, 32) for x in range(64, width - 64, 32)], dtype=np.float32)


def stat(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None, "bad1": None}
    return {"n": int(len(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": float(v.max()), "bad1": float((v >= 1).mean())}


def independent_eval(src, image, q):
    import importlib.util
    spec = importlib.util.spec_from_file_location("fixed_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.measure_arrays(src, image, query_xy=q, time_limit=60)
    known = np.array([r["known"] for r in result["records"]], dtype=bool)
    observed = np.array([r.get("observed_target_frame_xy", [np.nan, np.nan]) for r in result["records"]], dtype=np.float64)
    err = np.linalg.norm(observed - q, axis=1)
    return known, err


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    write(HERE / "POLICY.json", POLICY)
    write(ART / "POLICY.json", POLICY)
    obs_path = OLD / "flow_observations.npz"
    result_old = json.loads((OLD / "result.json").read_text(encoding="utf-8"))
    z = np.load(obs_path)
    fit_xy = z["fit_source_xy"].astype(np.float32)
    global_xy = z["global_target_xy"].astype(np.float32)
    global_known = z["global_known"].astype(bool)
    global_weight = z["global_weight"].astype(np.float32)
    local_xy = z["detail_target_frame_xy"].astype(np.float32)
    local_known = z["detail_known"].astype(bool) & global_known
    local_weight = z["detail_weight"].astype(np.float32)
    residual = local_xy - fit_xy
    residual_norm = np.linalg.norm(residual, axis=1)
    detail_fit = local_known & np.isfinite(residual_norm) & (residual_norm <= POLICY["detail_residual_max_px"])
    A = affine(fit_xy[global_known], global_xy[global_known], global_weight[global_known])
    src = cv2.imread(str(ROOT / "Train/input_f/2.png"), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(ROOT / "Train/output_f/2.png"), cv2.IMREAD_GRAYSCALE)
    h, w = src.shape
    field = np.zeros((h, w, 2), np.float32)
    mass = np.zeros((h, w), np.float32)
    for p, r, ww in zip(fit_xy[detail_fit], residual[detail_fit], local_weight[detail_fit]):
        x, y = p.astype(int)
        field[y, x] += r * ww
        mass[y, x] += ww
    mass[mass == 0] = 1.0
    field /= mass[:, :, None]
    field = cv2.GaussianBlur(field, (0, 0), POLICY["detail_smoothing_sigma_px"])
    yy, xx = np.mgrid[:h, :w]
    source_xy = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    base = source_xy @ A[:, :2].T + A[:, 2]
    detail = field.reshape(-1, 2) @ A[:, :2].T
    mapping = (base + POLICY["detail_alpha_fixed"] * detail).reshape(h, w, 2).astype(np.float32)
    support = np.isfinite(mapping).all(2) & (mapping[:, :, 0] >= 0) & (mapping[:, :, 0] <= w - 1) & (mapping[:, :, 1] >= 0) & (mapping[:, :, 1] <= h - 1)
    roi = support[64:h - 64, 64:w - 64]
    min_det = jacobian_min(mapping)
    out = ART / "Train_2_detail_alpha1"
    out.mkdir(parents=True, exist_ok=True)
    map_path = out / "composite_source_to_target_float32.npy"
    png_path = out / "aligned_target_on_source.png"
    np.save(map_path, mapping)
    aligned = cv2.remap(target, mapping[:, :, 0], mapping[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(png_path), aligned)
    replay = cv2.remap(target, mapping[:, :, 0], mapping[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    np.savez_compressed(out / "flow_observations_reused.npz", fit_source_xy=fit_xy, global_target_xy=global_xy, global_known=global_known, detail_target_frame_xy=local_xy, detail_known=local_known, detail_fit_selected=detail_fit, detail_residual_norm=residual_norm)
    q = query_grid(w, h)
    before_known, before_err = independent_eval(src, target, q)
    after_known, after_err = independent_eval(src, aligned, q)
    common = before_known & after_known
    report = {
        "id": 2, "status": "REAL_FLOW_DETAIL_V2_DIAGNOSTIC_REVIEW", "policy": POLICY,
        "input_f_sha256": sha(ROOT / "Train/input_f/2.png"), "target_output_f_sha256": sha(ROOT / "Train/output_f/2.png"),
        "reused_source_flow_observations_sha256": sha(obs_path), "global_flow_known": int(global_known.sum()),
        "detail_flow_known_before_outlier_gate": int(local_known.sum()), "detail_fit_selected": int(detail_fit.sum()),
        "detail_rejected_as_unknown_over_max_px": int((local_known & ~detail_fit).sum()), "detail_residual_p95": float(np.percentile(residual_norm[local_known], 95)),
        "fixed_query_denominator": len(q), "before_known": int(before_known.sum()), "after_known": int(after_known.sum()),
        "before_unknown": int((~before_known).sum()), "after_unknown": int((~after_known).sum()),
        "lost_known": int((before_known & ~after_known).sum()), "gained_known": int((~before_known & after_known).sum()),
        "before_common_source_px": stat(before_err[common]), "after_common_source_px": stat(after_err[common]),
        "fixed_query_coverage_before": float(before_known.mean()), "fixed_query_coverage_after": float(after_known.mean()),
        "fixed_query_support_loss": int((~roi).sum()), "full_raster_support_loss": int((~support).sum()),
        "jacobian_no_fold": bool(min_det >= POLICY["minimum_cell_det"]), "jacobian_min": min_det,
        "png_replay_different_channels": int(np.count_nonzero(aligned != replay)),
        "selected_alpha": POLICY["detail_alpha_fixed"], "eval_used_for_alpha": False, "gt_used": False,
        "map_sha256": sha(map_path), "png_sha256": sha(png_path), "flow_observations_reused_sha256": sha(out / "flow_observations_reused.npz"),
        "map_path": str(map_path), "png_path": str(png_path), "old_candidate_preserved": True,
        "old_candidate_result_sha256": sha(OLD / "result.json"), "old_candidate_alpha": result_old["selected_alpha"],
    }
    write(out / "RESULT.json", report)
    write(out / "RECEIPT.json", {"result_sha256": sha(out / "RESULT.json"), "map_sha256": report["map_sha256"], "png_sha256": report["png_sha256"], "flow_observations_reused_sha256": report["flow_observations_reused_sha256"], "map_direction": "source_to_target", "render": "target[map]", "real_data": True, "gt_used": False})
    write(HERE / "STATUS.json", {"stage": report["status"], "result": str(out / "RESULT.json"), "policy": str(HERE / "POLICY.json"), "old_candidate_modified": False, "training_write": False})


if __name__ == "__main__":
    main()
