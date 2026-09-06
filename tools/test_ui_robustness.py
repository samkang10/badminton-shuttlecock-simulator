"""UI 견고성 검사 — 아무것도 불러오지 않은 상태에서 버튼을 눌러도 앱이 멈추지 않는가.

연구용 프로그램에서 가장 흔한 사고는 추적 알고리즘이 아니라 '순서를 모르고 버튼을 누른'
경우다. 영상도 없고 추적도 안 한 상태에서 아무 버튼이나 누르면, 예전에는 핸들러가
int(None) 같은 오류를 내고 그 자리에서 멈췄다.

이 검사는 Gradio 에 등록된 이벤트 핸들러를 전부 찾아, 실제 사용자가 겪을 만한 입력
상태로 직접 호출한다.

  default      앱을 켜자마자 (모든 입력칸이 초기값)
  cleared      사용자가 숫자칸/선택칸을 지운 상태 (None)
  blank        공백 문자열 · NaN 이 들어간 상태
  adversarial  없는 파일 경로, 목록에 없는 선택지, 말도 안 되는 숫자

그리고 실제로 있을 법한 중간 상태(영상만 있음 / 물체만 지정함 / 물체를 다 지움 / 영상
파일이 사라짐)에서도 전체 버튼을 눌러 본다.

실행:
    python3 tools/test_ui_robustness.py
    python3 tools/test_ui_robustness.py --verbose
"""

import argparse
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
import time

import warnings

import numpy as np
import pandas as pd

                                                                              
                                                                              
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_shuttlecock import render                            

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tracker(path=None):
    path = path or os.path.join(_HERE, "AIPhysicsTracker.py")
    spec = importlib.util.spec_from_file_location("aipt_ui", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aipt_ui"] = mod
    spec.loader.exec_module(mod)
    logging.getLogger("AIPhysicsTracker").setLevel(logging.CRITICAL)
    return mod


def handler_list(demo):
    raw = getattr(demo, "fns", None)
    if raw is None:
        return []
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def find_controller(apt, items):
    """핸들러 클로저 안에 들어 있는 Controller 인스턴스를 꺼낸다."""
    for bf in items:
        fn = getattr(bf, "fn", None)
        if fn is None:
            continue
        inner = getattr(fn, "__wrapped__", fn)
        for cell in (inner.__closure__ or ()):
            try:
                if isinstance(cell.cell_contents, apt.Controller):
                    return cell.cell_contents
            except Exception:
                continue
    return None


def input_value(gr, component, mode):
    if mode == "cleared":
        if isinstance(component, (gr.Number, gr.Slider, gr.Dropdown, gr.Radio)):
            return None
        if isinstance(component, gr.Textbox):
            return ""
        if isinstance(component, gr.CheckboxGroup):
            return []
    elif mode == "blank":
        if isinstance(component, (gr.Number, gr.Slider)):
            return float("nan")
        if isinstance(component, (gr.Dropdown, gr.Radio)):
            return ""
        if isinstance(component, gr.Textbox):
            return "   "
        if isinstance(component, gr.CheckboxGroup):
            return []
    elif mode == "adversarial":
        if isinstance(component, (gr.Number, gr.Slider)):
            return -99999
        if isinstance(component, gr.Dropdown):
            return "존재하지않는값"
        if isinstance(component, gr.Radio):
            return "없는선택"
        if isinstance(component, gr.Textbox):
            return "존재하지않는경로.mp4"
        if isinstance(component, gr.CheckboxGroup):
            return ["없는항목"]
        if isinstance(component, gr.File):
            return "/없는/경로.csv"
    return getattr(component, "value", None)


def call_handler(gr, block_fn, args):
    """핸들러를 호출하고, 반환값을 출력 컴포넌트가 받아들이는지까지 확인한다."""
    fn = block_fn.fn
    try:
        sig = inspect.signature(fn)
        n_params = len([p for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        n_params = len(args)
    if n_params > len(args):
        try:
            evt = gr.SelectData(target=None,
                                data={"index": 0, "value": None, "selected": True})
        except Exception:
            evt = None
        args = list(args) + [evt] * (n_params - len(args))
    out = fn(*args)
    if inspect.isgenerator(out):
        last = None
        for last in out:
            pass
        out = last
                                                                          
    outs = block_fn.outputs or []
    if not outs:
        return
    values = out if isinstance(out, tuple) else (out,)
    for comp, value in zip(outs, values):
        if value is None or isinstance(value, dict):
            continue
        if isinstance(value, gr.blocks.Block):
                                                     
            continue
        comp.postprocess(value)


def scan(apt, gr, demo, items, mode, verbose=False):
    failures = []
    guard_hits = []
    original_hint = apt._handler_hint

    def spy(exc):
        guard_hits.append((type(exc).__name__, str(exc)[:100]))
        return original_hint(exc)

    apt._handler_hint = spy
    try:
        for i, bf in enumerate(items):
            if getattr(bf, "fn", None) is None:
                continue
            before = len(guard_hits)
            args = [input_value(gr, c, mode) for c in (bf.inputs or [])]
            name = getattr(getattr(bf.fn, "__wrapped__", bf.fn), "__name__", "?")
            try:
                call_handler(gr, bf, args)
            except Exception as exc:
                failures.append((i, name, f"{type(exc).__name__}: {exc}"[:120]))
            if len(guard_hits) > before:
                failures.append((i, name, f"보호막 발동 {guard_hits[-1][0]}: {guard_hits[-1][1]}"))
    finally:
        apt._handler_hint = original_hint
    if verbose:
        for i, name, msg in failures:
            print(f"    [{i:3d}] {name:28s} {msg}")
    return failures


def reset_controller(apt, ctrl) -> None:
    """컨트롤러를 '앱을 막 켠' 상태로 되돌린다(검사 사이의 상태 오염을 막는다)."""
    ctrl.camera_manager = apt.MultiCameraManager()
    ctrl.camera_manager.add_camera()
    ctrl.validation_result = pd.DataFrame()
    ctrl.tracking_validation = None
    ctrl.reference_trajectory = None
    ctrl.graph_generated = False
    for cache in ("_graph_cache", "_overlay_render_cache", "_overlay_valid_cache"):
        try:
            getattr(ctrl, cache).clear()
        except Exception:
            pass


def scenario_scan(apt, gr, demo, items, ctrl, workdir, verbose=False):
    """실제로 있을 법한 중간 상태에서 전체 버튼을 눌러 본다."""
    video, truth = render(os.path.join(workdir, "ui.mp4"), n_frames=25, base_scale=2.2)

    def mark(c):
        c.mark_object_box("obj1", 0, truth[0][0] - 15, truth[0][1] - 15,
                          truth[0][0] + 15, truth[0][1] + 15)

    scenarios = {
        "영상만 불러온 상태": lambda c: c.load_video(video),
        "영상 + 물체 지정, 추적 전": lambda c: (c.load_video(video), mark(c)),
        "물체를 모두 지운 상태": lambda c: (c.load_video(video),
                                    [c.remove_object(o) for o in list(c.state.objects)]),
        "추적까지 했지만 물리량 계산 전": lambda c: (
            c.load_video(video), mark(c),
            c.process_tracking(video, "CSRT", False, 0, "Forward", False,
                               tracking_profile="Shuttlecock", supervised=False)),
        "영상 파일이 사라진 상태": lambda c: (c.load_video(video),
                                     setattr(c.state, "video_path", "/없는/경로.mp4")),
    }
    results = {}
    for label, setup in scenarios.items():
        reset_controller(apt, ctrl)
        try:
            setup(ctrl)
        except Exception as exc:
            print(f"  (준비 실패: {label}: {exc})")
        results[label] = scan(apt, gr, demo, items, "default", verbose)
    return results


def gating_check(apt, gr, items, ctrl, workdir):
    """단계가 끝나기 전에는 눌리지 않고, 끝나면 열리는지 확인한다."""
    target = None
    for bf in items:
        inner = getattr(getattr(bf, "fn", None), "__wrapped__", getattr(bf, "fn", None))
        if getattr(inner, "__name__", "") == "on_state_refresh":
            target = bf
            break
    if target is None:
        return ["on_state_refresh 등록을 찾지 못했습니다"], {}

    labels = [getattr(c, "value", None) if isinstance(c, gr.Button) else None
              for c in target.outputs]

    def snapshot():
        out = target.fn("obj1")
        if not isinstance(out, tuple) or len(out) != len(target.outputs):
            raise AssertionError(f"출력 개수 불일치: {len(out) if isinstance(out, tuple) else 1}"
                                 f" != {len(target.outputs)}")
        return {str(lab)[:24]: bool(getattr(v, "interactive", None))
                for c, lab, v in zip(target.outputs, labels, out) if isinstance(c, gr.Button)}

    reset_controller(apt, ctrl)
    empty = snapshot()
    video, truth = render(os.path.join(workdir, "gate.mp4"), n_frames=20, base_scale=2.2)
    ctrl.load_video(video)
    with_video = snapshot()
    ctrl.mark_object_box("obj1", 0, truth[0][0] - 15, truth[0][1] - 15,
                         truth[0][0] + 15, truth[0][1] + 15)
    with_track = snapshot()

    problems = []
    on_at_start = [k for k, v in empty.items() if v]
    if on_at_start:
        problems.append(f"아무것도 없는데 눌리는 버튼: {on_at_start}")
    if not [k for k in with_video if with_video[k] and not empty[k]]:
        problems.append("영상을 불러와도 열리는 버튼이 없음")
    trajectory_buttons = [k for k in with_track
                          if any(w in k for w in ("이상 프레임", "다시 추적", "비교 검증", "되돌리기"))]
    still_closed = [k for k in trajectory_buttons if not with_track[k]]
    if still_closed:
        problems.append(f"궤적이 생겼는데도 닫혀 있는 버튼: {still_closed}")
    wrongly_open = [k for k in trajectory_buttons if with_video.get(k)]
    if wrongly_open:
        problems.append(f"궤적이 없는데 열려 있는 버튼: {wrongly_open}")
    counts = {"초기": sum(empty.values()), "영상 후": sum(with_video.values()),
              "궤적 후": sum(with_track.values()), "전체": len(empty)}
    return problems, counts


def outlier_speed_check(apt) -> str:
    """이상 프레임 검사가 긴 영상에서도 즉시 끝나는지 확인한다."""
    ctrl = apt.Controller()
    obj = ctrl.state.objects["obj1"]
    n = 6000
    f = np.arange(n).astype(float)
    obj.trajectory = pd.DataFrame({
        'frame': f, 't': f / 120.0, 'x_px': 60 + f * 1.2, 'y_px': 300 - f * 0.4,
        'w': 30.0, 'h': 30.0, 'conf': 0.9, 'keyframe': False,
        'source': 'DETECTED', 'tracking_status': 'HIGH'})
    t0 = time.perf_counter()
    ctrl.get_suspicious_frames("obj1", 0.5)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    apt = load_tracker()
    import gradio as gr
    demo = apt.build_gui()
    items = handler_list(demo)
    ctrl = find_controller(apt, items)
    workdir = tempfile.mkdtemp(prefix="ui_robust_")
    failures = []

    print(f"등록된 이벤트 핸들러: {len(items)}개")

    print("\n=== 입력 상태별 전체 버튼 클릭 ===")
    for mode in ("default", "cleared", "blank", "adversarial"):
        bad = scan(apt, gr, demo, items, mode, args.verbose)
        print(f"  {mode:12s} 문제 {len(bad)}건")
        failures += [f"[{mode}] {n}: {m}" for _i, n, m in bad]

    if ctrl is not None:
        print("\n=== 중간 상태별 전체 버튼 클릭 ===")
        for label, bad in scenario_scan(apt, gr, demo, items, ctrl, workdir,
                                        args.verbose).items():
            print(f"  {label:26s} 문제 {len(bad)}건")
            failures += [f"[{label}] {n}: {m}" for _i, n, m in bad]

        print("\n=== 단계별 버튼 열림/닫힘 (Tracker 식 순서 안내) ===")
        problems, counts = gating_check(apt, gr, items, ctrl, workdir)
        print(f"  켜진 버튼 수 — 초기 {counts.get('초기')} / 영상 후 {counts.get('영상 후')} "
              f"/ 궤적 후 {counts.get('궤적 후')} (전체 {counts.get('전체')})")
        for p in problems:
            print(f"  FAIL {p}")
        failures += problems
    else:
        print("\n  (Controller 를 찾지 못해 상태별 검사는 건너뜁니다)")

    print("\n=== 이상 프레임 검사 속도 ===")
    ms = outlier_speed_check(apt)
    print(f"  6000프레임 궤적: {ms:.1f} ms")
    if ms > 50.0:
        failures.append(f"이상 프레임 검사가 느립니다 ({ms:.0f} ms / 6000프레임)")

    print("\n=== 판정 ===")
    if failures:
        for f in failures[:25]:
            print(f"  FAIL  {f}")
        if len(failures) > 25:
            print(f"  ... 외 {len(failures) - 25}건")
        return 1
    print("  PASS  어떤 상태에서 어떤 버튼을 눌러도 앱이 멈추지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
