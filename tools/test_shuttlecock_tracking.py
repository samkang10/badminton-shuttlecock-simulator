"""셔틀콕 추적 파이프라인 검증 스위트.

정답을 아는 합성 영상으로 다음을 확인한다.

  1) 요구된 조건 전부에서 추적이 되는가
     - 정상 셔틀콕: 선명 / 빠름(모션 블러) / 작게 보이는 경우
     - 손상 셔틀콕: 2개 대칭, 2개 비대칭, 4개 대칭, 4개 비대칭
  2) 위치 오차와 물리량 오차가 얼마인가 (연구에 쓸 수 있는 근거)
  3) 중심 정의가 실행 내내 하나로 유지되는가
  4) 예측으로 채운 프레임이 관측 프레임과 구분되어 기록되는가
  5) 손상 조건이 달라져도 같은 기준으로 중심을 잡는가

실행:
    python3 tools/test_shuttlecock_tracking.py            # 요약만
    python3 tools/test_shuttlecock_tracking.py --verbose  # 조건별 상세
"""

import argparse
import importlib.util
import logging
import os
import sys
import tempfile

from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_shuttlecock import render                            

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tracker(path: str = None):
    path = path or os.path.join(_HERE, "AIPhysicsTracker.py")
    spec = importlib.util.spec_from_file_location("aipt", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aipt"] = mod
    spec.loader.exec_module(mod)
    logging.getLogger("AIPhysicsTracker").setLevel(logging.ERROR)
    return mod


                                                                              
                                                                              
CASES = [
    dict(name="정상 · 선명", damage="NORMAL", blur_frames=1, base_scale=2.2, shrink=0.0, v0=1500.0),
    dict(name="정상 · 빠름+블러", damage="NORMAL", blur_frames=7, base_scale=2.2, shrink=0.0, v0=1900.0),
    dict(name="정상 · 작게 보임", damage="NORMAL", blur_frames=1, base_scale=1.1, shrink=0.35, v0=1500.0),
    dict(name="손상 2개 대칭", damage="2_FEATHERS_SYMMETRIC", blur_frames=2, base_scale=2.2, shrink=0.0, v0=1600.0),
    dict(name="손상 2개 비대칭", damage="2_FEATHERS_ASYMMETRIC", blur_frames=2, base_scale=2.2, shrink=0.0, v0=1600.0),
    dict(name="손상 4개 대칭", damage="4_FEATHERS_SYMMETRIC", blur_frames=2, base_scale=2.2, shrink=0.0, v0=1600.0),
    dict(name="손상 4개 비대칭", damage="4_FEATHERS_ASYMMETRIC", blur_frames=2, base_scale=2.2, shrink=0.0, v0=1600.0),
]

N_FRAMES = 70
FPS = 120.0
                                                                              
                                                                              
SCALE_M_PX = 1.0 / 300.0
MASS_KG = 0.005


def run_case(apt, case: dict, workdir: str, use_damage_prior: bool = True) -> dict:
    """한 조건에서 추적을 돌리고 기준 궤적과 비교한 수치를 돌려준다."""
    path = os.path.join(workdir, case["name"].replace(" ", "_").replace("·", "-") + ".mp4")
    path, truth = render(path, n_frames=N_FRAMES, fps=FPS, damage=case["damage"],
                         blur_frames=case["blur_frames"], base_scale=case["base_scale"],
                         shrink=case["shrink"], v0=case["v0"])

    vp = apt.VideoProcessor()
    vp.tracking_profile = "Shuttlecock"
    vp.tracking_mode = "Balanced"
    vp.damage_condition = case["damage"] if use_damage_prior else "NORMAL"
    vp.center_method = "auto"
    vp.raw_tracking_only = True

    obj = apt.TrackedObject("obj1", "shuttlecock")
                                                                                
    box = max(16.0, 26.0 * case["base_scale"])
    obj.trajectory = pd.DataFrame([{
        'frame': 0, 't': 0.0, 'x_px': float(truth[0][0]), 'y_px': float(truth[0][1]),
        'w': box, 'h': box, 'conf': 1.0, 'keyframe': True,
    }])
    fps = vp.process_and_track(path, {"obj1": obj}, "CSRT", False, 0, "Forward", False)

    ref = pd.DataFrame({'frame': np.arange(len(truth)),
                        'x_px': truth[:, 0], 'y_px': truth[:, 1]})
    res = apt.ShuttlecockTrajectoryValidator.run(
        obj.trajectory, ref, fps, scale_m_px=SCALE_M_PX, mass_kg=MASS_KG)

    tr = obj.trajectory
    methods = set(str(v) for v in tr.get('center_method', pd.Series(dtype=str)).dropna().unique()
                  if str(v) not in ('nan', ''))
    sources = tr['source'].astype(str).value_counts().to_dict() if 'source' in tr.columns else {}
    return {'case': case["name"], 'damage': case["damage"], 'result': res,
            'trajectory': tr, 'center_methods': methods, 'sources': sources,
            'resolved_center': vp.get_center_method(), 'stats': vp.last_tracking_stats,
            'center_stats': vp.last_center_stats}


                                                                              
                                                                              
HARD_CASES = [
    dict(name="배경과 비슷한 색", damage="NORMAL", blur_frames=2, base_scale=2.2,
         shrink=0.0, v0=1600.0, bg_level=150, occlude=(0, 0)),
    dict(name="일시적 가림", damage="NORMAL", blur_frames=2, base_scale=2.2,
         shrink=0.0, v0=1600.0, bg_level=58, occlude=(24, 31)),
    dict(name="긴 비행 구간", damage="NORMAL", blur_frames=2, base_scale=2.2,
         shrink=0.25, v0=1250.0, bg_level=58, occlude=(0, 0), n_frames=140),
]


def run_hard_case(apt, case: dict, workdir: str) -> dict:
    n_frames = int(case.get("n_frames", N_FRAMES))
    path = os.path.join(workdir, "hard_" + case["name"].replace(" ", "_") + ".mp4")
    path, truth = render(path, n_frames=n_frames, fps=FPS, damage=case["damage"],
                         blur_frames=case["blur_frames"], base_scale=case["base_scale"],
                         shrink=case["shrink"], v0=case["v0"],
                         bg_level=case["bg_level"], occlude=case["occlude"])
    vp = apt.VideoProcessor()
    vp.tracking_profile = "Shuttlecock"
    vp.damage_condition = case["damage"]
    vp.center_method = "auto"
    vp.raw_tracking_only = True
    obj = apt.TrackedObject("obj1", "shuttlecock")
    box = max(16.0, 26.0 * case["base_scale"])
    obj.trajectory = pd.DataFrame([{
        'frame': 0, 't': 0.0, 'x_px': float(truth[0][0]), 'y_px': float(truth[0][1]),
        'w': box, 'h': box, 'conf': 1.0, 'keyframe': True}])
    fps = vp.process_and_track(path, {"obj1": obj}, "CSRT", False, 0, "Forward", False)
    ref = pd.DataFrame({'frame': np.arange(len(truth)), 'x_px': truth[:, 0], 'y_px': truth[:, 1]})
    res = apt.ShuttlecockTrajectoryValidator.run(obj.trajectory, ref, fps,
                                                 scale_m_px=SCALE_M_PX, mass_kg=MASS_KG)
    return {'case': case["name"], 'occlude': case["occlude"], 'result': res,
            'trajectory': obj.trajectory, 'truth': truth, 'n_frames': n_frames,
            'resolved_center': vp.get_center_method(), 'center_stats': vp.last_center_stats}


def check_occlusion(run: dict) -> str:
    """가려진 구간에서 '틀린 좌표를 관측값처럼' 기록하지 않았는지 본다.

    추적이 가려진 구간을 예측으로 잇거나 비워 두는 것은 정상이다. 문제가 되는 것은
    가려진 동안 엉뚱한 위치를 DETECTED 로, 그것도 높은 confidence 로 남기는 경우다.
    """
    a, b = run['occlude']
    if b <= a:
        return "가림 구간 없음"
    tr = run['trajectory']
    truth = run['truth']
    seg = tr[(pd.to_numeric(tr['frame'], errors='coerce') >= a)
             & (pd.to_numeric(tr['frame'], errors='coerce') <= b)]
    if seg.empty:
        return f"가림 구간 {a}~{b}: 좌표를 기록하지 않음 (LOST 처리)"
    idx = seg['frame'].astype(int).to_numpy()
    err = np.hypot(seg['x_px'].to_numpy(dtype=float) - truth[idx, 0],
                   seg['y_px'].to_numpy(dtype=float) - truth[idx, 1])
    src = seg['source'].astype(str).to_numpy() if 'source' in seg.columns else np.array([''] * len(seg))
    conf = pd.to_numeric(seg.get('conf'), errors='coerce').to_numpy()
    bad = int(np.count_nonzero((err > 3.0 * 26.0) & (src == 'DETECTED') & (conf > 0.5)))
    return (f"가림 구간 {a}~{b}: {len(seg)}프레임 기록 "
            f"(source={dict(pd.Series(src).value_counts())}, "
            f"최대오차 {np.nanmax(err):.1f}px, 관측으로 위장한 오검출 {bad}개)")


def worst_trusted_error(run: dict) -> Tuple[float, int]:
    """HIGH/MEDIUM 으로 기록된 프레임만 놓고 최대 위치 오차를 본다.

    연구자가 실제로 물리 분석에 쓰는 것은 신뢰도가 높은 프레임이다. LOW/LOST 로 표시되고
    이상 프레임 목록에도 오르는 값이 섞여 있는 것은 설계된 동작이지만, 그런 값이
    HIGH/MEDIUM 으로 기록된다면 그것은 실패를 정상 데이터로 위장한 것이다.
    """
    tr = run['trajectory']
    truth = run['truth']
    if 'tracking_status' not in tr.columns:
        return float('nan'), 0
    m = tr.dropna(subset=['x_px', 'y_px'])
    m = m[m['tracking_status'].astype(str).isin(['HIGH', 'MEDIUM'])]
    if m.empty:
        return float('nan'), 0
    idx = m['frame'].astype(int).to_numpy()
    err = np.hypot(m['x_px'].to_numpy(dtype=float) - truth[idx, 0],
                   m['y_px'].to_numpy(dtype=float) - truth[idx, 1])
    return float(np.nanmax(err)), int(len(err))


def compare_center_methods(apt, workdir: str) -> pd.DataFrame:
    """손상 조건이 달라질 때 어느 중심 정의가 가장 덜 흔들리는지 직접 재어 본다.

    같은 자세·같은 위치에 조건만 바꿔 셔틀콕을 그린 뒤, 각 정의가 내놓는 중심이
    정상 셔틀콕일 때의 중심에서 얼마나 벗어나는지 본다. 이 값이 작을수록 손상 조건
    사이의 비교에 쓰기 좋은 정의다.
    """
    import cv2
    from make_synthetic_shuttlecock import draw_shuttlecock, _background
    rng = np.random.default_rng(3)
    conditions = ["NORMAL", "2_FEATHERS_SYMMETRIC", "2_FEATHERS_ASYMMETRIC",
                  "4_FEATHERS_SYMMETRIC", "4_FEATHERS_ASYMMETRIC"]
    per_method = {m: [] for m in ("bbox", "mask_centroid", "weighted_centroid", "cork_center")}
    for axis in (0.0, 35.0, 90.0, 215.0):
        base = {}
        for cond in conditions:
            clean = _background(120, 120, rng)
            img = clean.copy()
                                                              
            draw_shuttlecock(img, 60.0, 60.0, axis, 2.4, cond)
                                                                                     
                                                                                     
                                                                                     
            diff = cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))
            mask = np.where(diff > 12, 255, 0).astype(np.uint8)
            cands = apt.ShuttlecockCenterEstimator.candidates(img, mask)
            for m in per_method:
                if m not in cands:
                    continue
                if cond == "NORMAL":
                    base[m] = cands[m]
                elif m in base:
                    per_method[m].append(float(np.hypot(cands[m][0] - base[m][0],
                                                        cands[m][1] - base[m][1])))
    rows = []
    for m, vals in per_method.items():
        rows.append({'중심 정의': m,
                     '손상에 따른 중심 이동 평균(px)': round(float(np.mean(vals)), 2) if vals else np.nan,
                     '최대(px)': round(float(np.max(vals)), 2) if vals else np.nan,
                     '표본': len(vals)})
    return pd.DataFrame(rows).sort_values('손상에 따른 중심 이동 평균(px)').reset_index(drop=True)


def summarize(runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        pos, vel, trk = r['result']['position'], r['result']['velocity'], r['result']['tracking']
        phys = r['result']['physics_table']
        worst = float('nan')
        if phys is not None and not phys.empty:
            rel = pd.to_numeric(phys['상대오차(%)'], errors='coerce')
            worst = float(rel.max()) if rel.notna().any() else float('nan')
        rows.append({
            '조건': r['case'],
            'MAE(px)': round(pos.get('mae_px', float('nan')), 2),
            'RMSE(px)': round(pos.get('rmse_px', float('nan')), 2),
            '최대(px)': round(pos.get('max_px', float('nan')), 2),
            '성공률(%)': round(trk.get('success_rate_pct', float('nan')), 1),
            '손실(%)': round(trk.get('lost_ratio_pct', float('nan')), 1),
            '예측프레임': int(trk.get('predicted_frames', 0)),
            '속도오차(%)': round(vel.get('v_mae_rel_pct', float('nan')), 1),
            '물리량 최대오차(%)': round(worst, 2),
            '중심정의': r.get('resolved_center', '-'),
        })
    return pd.DataFrame(rows)


def check_center_consistency(runs: list) -> dict:
    """모든 조건에서 중심 정의가 같은지, 그리고 그 정의를 실제로 몇 프레임에 적용했는지 본다.

    조건마다 다른 정의를 쓰면 손상 조건 사이의 비교 자체가 성립하지 않는다. 한 실행
    안에서도 물체 분리에 실패하면 상자 중심으로 대체되는데, 그런 프레임은 center_method
    열에 '(fallback)' 으로 표시되므로 그 비율을 확인한다.
    """
    resolved = {r['case']: r['resolved_center'] for r in runs}
    fallback = {}
    for r in runs:
        stats = r.get('center_stats') or {}
        total = int(stats.get('frames', 0) or 0)
        fb = int(stats.get('fallback_frames', 0) or 0)
        fallback[r['case']] = round(fb / total * 100.0, 1) if total else 0.0
    return {'resolved': resolved,
            'all_same': len(set(resolved.values())) == 1,
            'fallback_pct': fallback,
            'worst_fallback_pct': max(fallback.values()) if fallback else 0.0}


def check_source_integrity(runs: list) -> dict:
    """예측 프레임이 관측 프레임과 구분되어 기록되는지 확인한다."""
    problems = []
    for r in runs:
        tr = r['trajectory']
        if 'source' not in tr.columns or 'tracking_status' not in tr.columns:
            problems.append(f"{r['case']}: source/tracking_status 열이 없습니다")
            continue
        pred = tr[tr['source'].astype(str) == 'PREDICTED']
        bad = pred[pred['tracking_status'].astype(str).isin(['HIGH', 'MEDIUM'])]
        if len(bad):
            problems.append(f"{r['case']}: 예측 프레임 {len(bad)}개가 HIGH/MEDIUM 으로 기록됨")
        conf = pd.to_numeric(pred.get('conf'), errors='coerce')
        if len(conf) and conf.max() > 0.5:
            problems.append(f"{r['case']}: 예측 프레임 confidence 가 {conf.max():.2f} 로 너무 높음")
    return {'ok': not problems, 'problems': problems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tracker", default=None, help="AIPhysicsTracker.py 경로")
    args = ap.parse_args()

    apt = load_tracker(args.tracker)
    workdir = tempfile.mkdtemp(prefix="shuttle_test_")
    runs = [run_case(apt, c, workdir) for c in CASES]

    table = summarize(runs)
    print("\n=== 조건별 추적 결과 (합성 영상, 정답 궤적 대비) ===")
    print(table.to_string(index=False))

    center = check_center_consistency(runs)
    print("\n=== 중심 정의 일관성 ===")
    print(f"조건별 중심 정의: {center['resolved']}")
    print(f"모든 조건에서 동일: {center['all_same']}")
    print(f"물체 분리 실패로 상자 중심을 대신 쓴 비율(%): {center['fallback_pct']}")

    integ = check_source_integrity(runs)
    print("\n=== 데이터 무결성 (예측/관측 구분) ===")
    print("문제 없음" if integ['ok'] else "\n".join(integ['problems']))

                                                                              
    hard = [run_hard_case(apt, c, workdir) for c in HARD_CASES]
    print("\n=== 어려운 상황 ===")
    print(summarize(hard).to_string(index=False))
    for h in hard:
        if h['occlude'][1] > h['occlude'][0]:
            print("  " + check_occlusion(h))
    if args.verbose:
        for h in hard:
            print(f"\n--- {h['case']} 물리량 ---")
            print(h['result']['physics_table'].to_string(index=False))
            obs = h['result'].get('physics_table_observed')
            if obs is not None and not obs.empty:
                print("  (예측 프레임 제외)")
                print(obs.to_string(index=False))

                                                                              
    print("\n=== 중심 정의별 손상 민감도 (작을수록 조건 간 비교에 안정적) ===")
    cmp_table = compare_center_methods(apt, workdir)
    print(cmp_table.to_string(index=False))
    print("  주의: 여기서 bbox 는 상자를 정답 위치에 고정해 두고 잰 값이라 0 이 나온다. "
          "실제 추적에서는 상자 자체가 추적기 출력이므로 이 표의 bbox 값은 하한이다.")

    if args.verbose:
        for r in runs:
            print(f"\n--- {r['case']} ---")
            print("source 분포:", r['sources'])
            print(r['result']['physics_table'].to_string(index=False))

                                                                              
                                                                              
    fails = []
    for _i, row in table.iterrows():
        if not np.isfinite(row['MAE(px)']):
            fails.append(f"{row['조건']}: 비교 가능한 프레임 없음")
        elif row['성공률(%)'] < 90.0:
            fails.append(f"{row['조건']}: 추적 성공률 {row['성공률(%)']}%")
    if not center['all_same']:
        fails.append("손상 조건에 따라 중심 정의가 달라짐")
    if center['worst_fallback_pct'] > 25.0:
        fails.append(f"상자 중심 대체 비율이 너무 높음 ({center['worst_fallback_pct']}%) — "
                     f"중심 정의가 프레임마다 달라진 셈이 됨")
    if not integ['ok']:
        fails.extend(integ['problems'])
    hard_table = summarize(hard)
    for _i, row in hard_table.iterrows():
        if not np.isfinite(row['MAE(px)']):
            fails.append(f"[어려운 상황] {row['조건']}: 비교 가능한 프레임 없음")
                                                                                     
                                                                                     
                                                                                     
    for h in hard:
        worst, n = worst_trusted_error(h)
        print(f"  {h['case']}: HIGH/MEDIUM 프레임 {n}개의 최대 위치 오차 "
              f"{worst:.1f}px" if np.isfinite(worst) else f"  {h['case']}: 신뢰 프레임 없음")
        if np.isfinite(worst) and worst > 60.0:
            fails.append(f"[어려운 상황] {h['case']}: 신뢰(HIGH/MEDIUM) 프레임에 "
                         f"{worst:.0f}px 오차가 기록됨 — 실패가 정상 데이터로 남음")
    occ = [h for h in hard if h['occlude'][1] > h['occlude'][0]]
    for h in occ:
        a, b = h['occlude']
        tr = h['trajectory']
        seg = tr[(pd.to_numeric(tr['frame'], errors='coerce') >= a)
                 & (pd.to_numeric(tr['frame'], errors='coerce') <= b)]
        if not seg.empty and 'source' in seg.columns:
            idx = seg['frame'].astype(int).to_numpy()
            err = np.hypot(seg['x_px'].to_numpy(dtype=float) - h['truth'][idx, 0],
                           seg['y_px'].to_numpy(dtype=float) - h['truth'][idx, 1])
            src = seg['source'].astype(str).to_numpy()
            conf = pd.to_numeric(seg.get('conf'), errors='coerce').to_numpy()
            if int(np.count_nonzero((err > 78.0) & (src == 'DETECTED') & (conf > 0.5))):
                fails.append(f"[어려운 상황] {h['case']}: 가림 구간에서 오검출을 "
                             f"관측(DETECTED)으로 기록함")

    print("\n=== 판정 ===")
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        return 1
    print("  PASS  모든 조건에서 추적 성공률·중심 일관성·데이터 무결성 기준을 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
