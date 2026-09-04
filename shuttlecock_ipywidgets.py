"""셔틀콕 Turnover · 궤적 시뮬레이터 v9.2 — ipywidgets 판 (Gradio 불필요).

`shuttlecock_simulator_9.2.py` 와 **같은 물리 코드**를 그대로 불러다 쓰고, 화면만
ipywidgets 로 다시 그린 것입니다. 물리는 한 줄도 복사하지 않았으므로 두 판의 계산
결과는 정의상 동일합니다.

왜 따로 만드나
--------------
Gradio 판은 노트북 안에서 HTTP 서버를 띄우고 브라우저가 그 서버에 붙어야 합니다.
Colab 에서는 그 구조 때문에 공유 링크(share.gradio.live)가 막히면 이벤트 스트림(SSE)이
전달되지 않아 화면은 떠도 버튼이 전부 죽습니다. ipywidgets 는 서버를 띄우지 않고
노트북의 커널 통신 채널을 그대로 쓰므로 포트도, 터널도, 공유 링크도 필요 없습니다.

쓰는 법 (Colab / Jupyter)
-------------------------
    %pip install -q ipywidgets plotly
    import shuttlecock_ipywidgets as app
    app.launch()

스크립트에서 UI 없이 물리만 쓰려면 원본 모듈을 직접 부르면 됩니다.
"""

import os
import sys
import glob
import importlib.util
import subprocess

APP_VERSION = "9.2"
UI_VARIANT = "ipywidgets"
PHYSICS_FILENAME = "shuttlecock_simulator_9.2.py"


def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
        return
    except ImportError:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", pkg], check=True)


_ensure("ipywidgets")
_ensure("plotly")

import math
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import ipywidgets as W
import plotly.graph_objects as go
from IPython.display import display, clear_output


def _find_physics(path=None):
    """물리 모듈 파일 경로를 찾는다.

    파일 이름에 점이 있어서 (`shuttlecock_simulator_9.2.py`) 평범한 import 가 되지
    않으므로 경로로 직접 로드한다.
    """
    if path and os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, PHYSICS_FILENAME),
                 os.path.join(os.getcwd(), PHYSICS_FILENAME)):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(here, "shuttlecock_simulator_*.py")) \
        or glob.glob(os.path.join(os.getcwd(), "shuttlecock_simulator_*.py"))
    if hits:
        return sorted(hits)[-1]
    raise FileNotFoundError(
        f"{PHYSICS_FILENAME} 을 찾지 못했습니다. 같은 폴더에 두거나 "
        "load_physics(path=...) 로 경로를 지정하세요.")


_PHYS = None


def load_physics(path=None):
    """물리 모듈을 로드한다 (gradio 는 불러오지 않는다)."""
    global _PHYS
    if _PHYS is not None:
        return _PHYS
    target = _find_physics(path)
    spec = importlib.util.spec_from_file_location("shuttlecock_physics_core", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PHYS = mod
    return mod


# ---------------------------------------------------------------------------
# 공통 위젯 도우미
# ---------------------------------------------------------------------------

_LBL = {"description_width": "initial"}
_WIDE = W.Layout(width="460px")
_HALF = W.Layout(width="300px")


def _num(value, desc, step=None, layout=None):
    return W.FloatText(value=value, description=desc, step=step,
                       style=_LBL, layout=layout or _WIDE)


def _int(value, desc, layout=None):
    return W.IntText(value=value, description=desc, style=_LBL,
                     layout=layout or _WIDE)


def _drop(options, value, desc, layout=None):
    return W.Dropdown(options=list(options), value=value, description=desc,
                      style=_LBL, layout=layout or _WIDE)


def _chk(value, desc):
    return W.Checkbox(value=value, description=desc, style=_LBL,
                      layout=W.Layout(width="620px"))


def _md(text=""):
    return W.HTML(value=_as_html(text))


def _as_html(text):
    """마크다운에 가까운 문자열을 최소한으로 HTML 로 바꾼다."""
    if text is None:
        return ""
    out = str(text)
    out = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # ### 제목
    out = "\n".join(
        ("<b style='font-size:1.05em'>%s</b>" % ln.lstrip("# ").strip())
        if ln.lstrip().startswith("#") else ln
        for ln in out.split("\n"))
    # **굵게**
    parts = out.split("**")
    if len(parts) > 2:
        rebuilt = []
        for i, seg in enumerate(parts):
            rebuilt.append(("<b>%s</b>" % seg) if i % 2 == 1 else seg)
        out = "".join(rebuilt)
    return "<div style='line-height:1.6'>%s</div>" % out.replace("\n", "<br>")


def _section(title, *children):
    return W.VBox([W.HTML("<h4 style='margin:6px 0 2px'>%s</h4>" % title)] + list(children))


def _note(text):
    return W.HTML("<div style='color:#64748b;font-size:0.9em;margin:2px 0 8px'>%s</div>"
                  % text)


# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------

class ShuttlecockApp:
    """Gradio 판과 같은 워크플로를 ipywidgets 로 제공한다.

    담당 범위: 셔틀콕 설정 · 파손 · 발사 조건 · 시뮬레이션(Level 1 / 고급 / 3D) ·
    결과 분석 · 수치 검증 · 결과 저장. 계산은 전부 원본 물리 모듈이 수행한다.
    """

    def __init__(self, physics_path=None):
        self.S = load_physics(physics_path)
        S = self.S
        self.db = S.ParameterDatabase()
        self.skirt = S.SkirtGeometry()
        self.state = dict(result=None, model=None, env=None, ic=None, kind=None)
        self.warnings = []
        self._build_widgets()

    # ---- 상태 표시 -------------------------------------------------------
    def _push(self, msg):
        if msg and msg not in self.warnings:
            self.warnings.append(str(msg))

    def _clear_warnings(self):
        self.warnings = []

    def _refresh_status(self):
        S, res = self.S, self.state["result"]
        gs = self.skirt.geometry_state()
        run = "없음" if res is None else ("정상" if res.valid and len(res.df) else "실패")
        self.status.value = _as_html(
            "**셔틀콕:** %s &nbsp;|&nbsp; **깃털:** %d/%d &nbsp;|&nbsp; "
            "**발사:** %.1f m/s, %.1f° &nbsp;|&nbsp; **모델:** %s &nbsp;|&nbsp; "
            "**결과:** %s"
            % (self.profile.value, gs["remaining_feathers"], gs["N_feathers"],
               self.V0.value, self.launch_angle.value,
               self.state["kind"] or "미실행", run))
        if self.warnings:
            self.warn_box.value = (
                "<div style='background:#fff7ed;border:1px solid #fdba74;"
                "border-radius:8px;padding:8px'>⚠ " +
                "<br>⚠ ".join(_as_html(w).replace("<div style='line-height:1.6'>", "")
                              .replace("</div>", "") for w in self.warnings) + "</div>")
        else:
            self.warn_box.value = ("<div style='color:#166534'>✓ 경고 없음</div>")
        self.comp_view.value = _as_html(self._components_text())

    def _components_text(self):
        try:
            th = math.radians(float(self.launch_angle.value))
            v = float(self.V0.value)
            return ("**vx₀ = %.4f m/s** &nbsp;|&nbsp; **vy₀ = %.4f m/s**\n"
                    "*vx₀ = V₀·cos(θ₀), vy₀ = V₀·sin(θ₀)*" % (v * math.cos(th),
                                                              v * math.sin(th)))
        except (TypeError, ValueError):
            return "발사 속도와 발사각에 숫자를 입력하세요."

    # ---- 위젯 정의 -------------------------------------------------------
    def _build_widgets(self):
        S = self.S
        # 1. 셔틀콕 설정
        self.profile = _drop(self.db.profile_names(), "BWF 천연 깃털 (기본)",
                             "셔틀콕 종류")
        self.skirt_d = _num(S.BWF_DEFAULTS["skirt_tip_diameter_mm"], "깃털 끝 지름 [mm]")
        self.cd_override = W.Text(value="", description="Cd 직접 지정 (비우면 프로파일)",
                                  style=_LBL, layout=_WIDE)
        self.tw_override = W.Text(value="", description="Tw [s] 직접 지정 (linear 전용)",
                                  style=_LBL, layout=_WIDE)
        self.zeta_override = W.Text(value="", description="ζ 직접 지정 (linear 전용)",
                                    style=_LBL, layout=_WIDE)
        self.rho = _num(1.2, "공기 밀도 ρ [kg/m³]")
        self.u_air_x = _num(0.0, "바람 x [m/s]")
        self.turnover_mode = W.RadioButtons(
            options=["aero_pendulum", "linear"], value="aero_pendulum",
            description="회전 방정식", style=_LBL)
        self.l_gc = _num(0.020, "l_GC [m]", step=0.001)
        self.damping_scale = _num(2.5, "감쇠 보정계수", step=0.1)
        self.cd_cross = _num(0.94, "가로자세 Cd(90°)", step=0.01)
        self.orient_cd = _chk(True, "자세에 따른 Cd 변화 사용 (정렬 0.65 → 가로 0.94)")
        self.aero_spin = _chk(True, "깃털 경사로 생기는 축 Spin 자동 계산 (RΩ/U ≈ 0.04)")
        self.param_preview = _md()

        # 2. 파손
        self.damage_preset = _drop(list(S.DAMAGE_PRESETS.keys()) + ["직접 지정"],
                                   "Normal", "파손 프리셋")
        self.n_remove = _int(0, "제거할 깃털 수")
        self.damage_pattern = _drop(["한쪽 집중", "좌우 대칭", "고르게 분산"],
                                    "한쪽 집중", "깃털이 빠지는 모양")
        self.damage_info = _md()
        self.damage_out = W.Output()

        # 3. 발사 조건
        self.preset = _drop(list(S.PRESETS.keys()), "Normal smash", "발사 방식 프리셋")
        self.V0 = _num(25.0, "① 발사 속도 V₀ [m/s]")
        self.launch_angle = _num(-10.0, "② 발사각 θ₀ [deg]")
        self.height = _num(1.8, "발사 높이 [m]")
        self.orient_preset = W.RadioButtons(
            options=list(S.ORIENTATION_PRESETS.keys()), value="Cork Forward",
            description="발사 순간 방향", style=_LBL)
        self.custom_orient = _num(180.0, "사용자 지정 자세각 [deg]")
        self.alpha0 = _num(145.0, "초기 받음각 α₀ [deg]")
        self.alpha_eq = _num(180.0, "평형 받음각 [deg]")
        self.omega0 = _num(0.0, "④ 초기 Turnover 각속도 [rad/s]")
        self.spin0 = _num(0.0, "⑤ 초기 축 Spin [rad/s]")
        self.comp_view = _md()

        # 4~6. 시뮬레이션
        self.level = _drop([1, 2], 2, "모델 수준 (1=점질량, 2=결합)")
        self.solver = _drop(["RK4", "Euler"], "RK4", "수치 적분기")
        self.dt = _num(0.001, "시간 간격 dt [s]", step=0.0005)
        self.t_max = _num(3.0, "종료 시간 [s]")
        self.elev3d = _num(-10.0, "수직 발사각 [deg]")
        self.azim3d = _num(0.0, "수평 발사각 [deg]")
        self.z0_3d = _num(0.0, "초기 z₀ [m]")
        self.orient_off_3d = _num(180.0, "자세각 (맞바람 기준) [deg]")
        self.cd_3d = _num(0.6, "항력계수 Cd")
        self.cl_mode_3d = _drop(["crossflow", "disabled", "constant"], "crossflow",
                                "양력계수 Cl 방식")
        self.geom_inertia = _chk(True, "형상 기반 관성텐서 사용 (비대칭 파손)")

        self.result_cards = _md("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
        self.plot_out = W.Output()
        self.status = _md()
        self.warn_box = W.HTML()
        self.verify_out = W.Output()
        self.export_out = W.Output()

    # ---- 모델 조립 (Gradio 판의 컨트롤러와 같은 순서) --------------------
    @staticmethod
    def _opt(text):
        """빈 칸은 '지정 안 함'. 빈 Number 가 0 으로 도착하는 문제를 피하려고
        문자열 입력을 쓰고, 양수일 때만 지정으로 본다."""
        try:
            v = float(str(text).strip())
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    def _geometry(self):
        return self.S.Geometry(skirt_diameter_m=float(self.skirt_d.value) / 1000.0)

    def _build_model(self, level, cd_value=None):
        S = self.S
        geom = self._geometry()
        removed = self.skirt.removed_ids
        dmg = S.build_porosity_damage_mapping() if removed else None
        cd_model = None
        if cd_value is not None:
            cd_model = S.CdModel(mode="constant", constant_value=float(cd_value),
                                 source_type="user_defined")
        model = S.build_full_model(self.db, self.profile.value, geom, level=level,
                                   removed_feathers=removed, cd_model=cd_model,
                                   damage_mapping=dmg,
                                   use_geometry_inertia=bool(self.geom_inertia.value),
                                   alpha_equilibrium_deg=float(self.alpha_eq.value))
        cd_ov = self._opt(self.cd_override.value)
        if cd_ov is not None:
            model.Cd_model.constant_value = cd_ov
        tw_ov = self._opt(self.tw_override.value)
        if tw_ov is not None:
            model.turnover.Tw = tw_ov
        z_ov = self._opt(self.zeta_override.value)
        if z_ov is not None:
            model.turnover.zeta = z_ov
        S.apply_flip_settings(model, self.turnover_mode.value, self.l_gc.value,
                              self.damping_scale.value, self.cd_cross.value,
                              self.orient_cd.value, self.aero_spin.value, geom)
        return model, geom

    def _env(self):
        return self.S.EnvironmentModel(rho=float(self.rho.value),
                                       u_air_x=float(self.u_air_x.value))

    def _guard(self, model, env):
        S = self.S
        errs = S.physical_sanity_checks(model, float(self.dt.value), env)
        errs += S.damage_domain_errors(model)
        for e in errs:
            self._push(e)
        return errs

    # ---- 실행 ------------------------------------------------------------
    def run_2d(self, level):
        S = self.S
        self._clear_warnings()
        model, geom = self._build_model(level)
        env = self._env()
        if self._guard(model, env):
            self.state.update(result=None, kind=None)
            self.result_cards.value = _as_html("실행하지 않았습니다 — 아래 경고를 확인하세요.")
            self._refresh_status()
            return None
        bo = S.resolved_body_orientation(self.orient_preset.value,
                                         self.custom_orient.value,
                                         float(self.alpha0.value),
                                         float(self.V0.value),
                                         float(self.launch_angle.value),
                                         float(self.u_air_x.value), 0.0)
        for w in S.attitude_warnings(self.orient_preset.value, float(self.alpha0.value),
                                     float(self.alpha_eq.value), self.omega0.value):
            self._push(w)
        ic = S.InitialCondition.from_launch(
            float(self.V0.value), float(self.launch_angle.value), float(self.height.value),
            float(self.alpha0.value), float(self.alpha_eq.value),
            omega0=float(self.omega0.value), body_orientation_deg=bo,
            spin0_rad_s=float(self.spin0.value), u_air_x=float(self.u_air_x.value))
        for w in S.geometry_warnings(model):
            self._push(w)
        res = S.SimulationEngine(model, env, ic, solver=self.solver.value,
                                 dt=float(self.dt.value), t_max=float(self.t_max.value),
                                 model_level=level).run()
        for w in res.warnings:
            self._push(w)
        if self._envelope_refused(res, model, env):
            return None
        self.state.update(result=res, model=model, env=env, ic=ic,
                          kind="Level %d (2D)" % level)
        return res

    def run_3d(self):
        S = self.S
        self._clear_warnings()
        model, geom = self._build_model(2, cd_value=float(self.cd_3d.value))
        model.aero.cl_model = S.ClModel(mode=self.cl_mode_3d.value)
        model.aero.cl_model.crossflow_delta_cd = (model.turnover.Cd_cross
                                                  - model.turnover.Cd_ref)
        env = self._env()
        if self._guard(model, env):
            self.state.update(result=None, kind=None)
            self.result_cards.value = _as_html("실행하지 않았습니다 — 아래 경고를 확인하세요.")
            self._refresh_status()
            return None
        ic = S.InitialCondition3D(
            x0=0.0, y0=float(self.height.value), z0=float(self.z0_3d.value),
            V0=float(self.V0.value), elevation_deg=float(self.elev3d.value),
            azimuth_deg=float(self.azim3d.value),
            body_axis_offset_deg=float(self.orient_off_3d.value),
            omega0=(0.0, 0.0, float(self.omega0.value)),
            spin0_rad_s=float(self.spin0.value))
        for w in S.geometry_warnings(model):
            self._push(w)
        res = S.SimulationEngine3D(model, env, ic, solver=self.solver.value,
                                   dt=float(self.dt.value),
                                   t_max=float(self.t_max.value)).run()
        for w in res.warnings:
            self._push(w)
        if self._envelope_refused(res, model, env):
            return None
        self.state.update(result=res, model=model, env=env, ic=ic, kind="3D (6-DOF)")
        return res

    def _envelope_refused(self, res, model, env):
        """코트를 크게 넘는 궤적은 결과로 내놓지 않는다.

        셔틀콕은 공기역학 길이 ℓ ≈ 4 m 하나로 움직이고 사거리가 로그로 포화하므로
        (Cohen 2015), 80 m 궤적은 항력이 꺼졌다는 뜻이지 배드민턴이 아니다.
        """
        errs = self.S.flight_envelope_errors(res.df, model, env)
        if not errs:
            return False
        for e in errs:
            self._push(e)
        self.state.update(result=None, kind=None)
        self.result_cards.value = _as_html(
            "물리적으로 불가능한 궤적이라 결과를 내지 않았습니다 — 경고를 확인하세요.")
        self._refresh_status()
        return True

    # ---- 결과 표시 -------------------------------------------------------
    def _fmt(self, v, unit=""):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return ("%.4g%s" % (v, unit)) if isinstance(v, (int, float)) else str(v)

    def show_result(self):
        S, res = self.S, self.state["result"]
        if res is None or not res.valid or len(res.df) == 0:
            self.result_cards.value = _as_html("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
            with self.plot_out:
                clear_output(wait=True)
            self._refresh_status()
            return
        df = res.df
        summ = S.summarize_result(res)
        rows = [("비행 시간", self._fmt(summ.get("flight_time_s"), " s")),
                ("최대 도달 거리", self._fmt(summ.get("max_range_m"), " m")),
                ("초기 속도", self._fmt(summ.get("initial_speed_m_s"), " m/s")),
                ("최종 속도", self._fmt(summ.get("final_speed_m_s"), " m/s")),
                ("최고 높이", self._fmt(summ.get("max_height_m"), " m")),
                ("Turnover 시간", self._fmt(summ.get("turnover_time_s"), " s")),
                ("안정화 시간", self._fmt(summ.get("stabilization_time_s"), " s")),
                ("최대 각속도", self._fmt(summ.get("max_angular_velocity_rad_s"), " rad/s")),
                ("최대 Wobble", self._fmt(summ.get("max_wobble_deg"), "°"))]
        cells = "".join(
            "<td style='padding:4px 14px 4px 0'><b>%s</b><br>%s</td>%s"
            % (k, v, "</tr><tr>" if (i + 1) % 3 == 0 else "")
            for i, (k, v) in enumerate(rows))
        self.result_cards.value = ("<table style='line-height:1.5'><tr>%s</tr></table>"
                                   % cells)
        with self.plot_out:
            clear_output(wait=True)
            display(self._trajectory_fig(df))
            display(self._speed_fig(df))
            display(self._angle_fig(df))
        self._refresh_status()

    def _base_layout(self, fig, title, xt, yt, h=340):
        fig.update_layout(template="plotly_white", height=h, title=title,
                          xaxis_title=xt, yaxis_title=yt,
                          margin=dict(l=60, r=20, t=48, b=45))
        return fig

    def _trajectory_fig(self, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.x_m, y=df.y_m, mode="lines", name="궤적"))
        fig.add_trace(go.Scatter(x=[df.x_m.iloc[0]], y=[df.y_m.iloc[0]], mode="markers",
                                 name="출발", marker=dict(size=10, color="#166534")))
        fig.add_trace(go.Scatter(x=[df.x_m.iloc[-1]], y=[df.y_m.iloc[-1]], mode="markers",
                                 name="도착", marker=dict(size=10, color="#b91c1c")))
        return self._base_layout(fig, "궤적 (x–y)", "x [m]", "y [m] (높이)", 380)

    def _speed_fig(self, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.time_s, y=df.speed_m_s, mode="lines", name="속력"))
        if "V_rel_m_s" in df.columns:
            fig.add_trace(go.Scatter(x=df.time_s, y=df.V_rel_m_s, mode="lines",
                                     name="상대풍속"))
        return self._base_layout(fig, "속도–시간", "t [s]", "속도 [m/s]")

    def _angle_fig(self, df):
        fig = go.Figure()
        col = ("wobble_signed_deg" if "wobble_signed_deg" in df.columns
               else "delta_alpha_deg")
        fig.add_trace(go.Scatter(x=df.time_s, y=df[col], mode="lines",
                                 name="Wobble (α − 평형)"))
        if "omega_rad_s" in df.columns:
            fig.add_trace(go.Scatter(x=df.time_s, y=df.omega_rad_s, mode="lines",
                                     name="각속도 [rad/s]", yaxis="y2"))
            fig.update_layout(yaxis2=dict(title="각속도 [rad/s]", overlaying="y",
                                          side="right"))
        return self._base_layout(fig, "자세각 · 각속도", "t [s]", "Wobble [deg]")

    def feather_figure(self):
        gs = self.skirt.geometry_state()
        fig = go.Figure()
        for f in self.skirt.feathers:
            a = math.radians(f.azimuth_deg)
            fig.add_trace(go.Scatter(
                x=[0, math.cos(a)], y=[0, math.sin(a)], mode="lines+markers",
                line=dict(color="#1d4ed8" if f.attached else "#e2e8f0", width=6),
                marker=dict(size=6), showlegend=False,
                hovertext="깃털 %d %s" % (f.id, "부착" if f.attached else "제거")))
        fig.update_layout(
            template="plotly_white", height=330, showlegend=False,
            title="남은 깃털 %d/%d · 비대칭도 %.3f"
                  % (gs["remaining_feathers"], gs["N_feathers"], gs["geometry_asymmetry"]),
            xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=44, b=10))
        return fig

    # ---- 파손 조작 --------------------------------------------------------
    def geometry_text(self):
        gs = self.skirt.geometry_state()
        asym = gs["geometry_asymmetry"]
        if gs["removed_feathers"] == 0:
            judge = "정상 상태 (BWF 규격 기준 셔틀콕)"
        elif asym < 0.05:
            judge = "대칭 파손 (깃털이 고르게 빠짐)"
        elif asym < 0.3:
            judge = "약한 비대칭 파손"
        else:
            judge = "강한 비대칭 파손 (한쪽에 집중)"
        f = self._fmt
        return (
            "**남은 깃털:** %d / %d &nbsp;|&nbsp; **제거된 깃털:** %d &nbsp;|&nbsp; "
            "**비대칭도:** %.3f\n"
            "**판정:** %s\n"
            "**제거된 깃털 번호:** %s\n"
            "**스커트 개방 비율:** %s &nbsp;|&nbsp; **추정 공극률:** %s\n"
            "**유효 스커트 폭:** %s &nbsp;|&nbsp; **유효 투영 면적:** %s\n"
            "*깃털 제거는 형상만 바꿉니다. 실험·문헌 대응 관계가 없으면 "
            "Cd·Cl·Cm 자체는 변하지 않습니다.*"
            % (gs["remaining_feathers"], gs["N_feathers"], gs["removed_feathers"], asym,
               judge, gs["removed_ids"] or "없음",
               f(gs["skirt_open_fraction"]), f(gs["estimated_porosity"]),
               f(gs["effective_skirt_width"], " m"),
               f(gs["effective_projected_area"], " m²")))

    def _redraw_damage(self):
        self.damage_info.value = _as_html(self.geometry_text())
        with self.damage_out:
            clear_output(wait=True)
            display(self.feather_figure())
        self._refresh_status()

    def _removal_ids(self, n, pattern):
        total = self.skirt.n_feathers
        n = max(0, min(int(round(float(n or 0))), total))
        ids = list(range(1, total + 1))
        if n == 0:
            return []
        if pattern == "한쪽 집중":
            return ids[:n]
        if pattern == "좌우 대칭":
            removed, i = [], 0
            while len(removed) < n:
                for c in (ids[i % total], ids[(i + total // 2) % total]):
                    if c not in removed and len(removed) < n:
                        removed.append(c)
                i += 1
            return sorted(removed)
        step = total / n
        removed = sorted({ids[int(round(k * step)) % total] for k in range(n)})
        k = 0
        while len(removed) < n and k < total:
            if ids[k] not in removed:
                removed.append(ids[k])
            k += 1
        return sorted(removed)

    def apply_damage_numeric(self, *_):
        removed = self._removal_ids(self.n_remove.value, self.damage_pattern.value)
        self.skirt.restore_all()
        for fid in removed:
            self.skirt.remove(fid)
        self._clear_warnings()
        if removed and len(removed) == self.skirt.n_feathers:
            self._push("깃털을 전부 제거했습니다: 물리적으로 불가능한 형상이라 "
                       "결과에 의미가 없습니다.")
        self._redraw_damage()

    def apply_damage_preset(self, *_):
        name = self.damage_preset.value
        if name == "직접 지정":
            return
        removed = self.S.DAMAGE_PRESETS.get(name)
        if removed is None:
            return
        self.skirt.restore_all()
        for fid in removed:
            self.skirt.remove(fid)
        self.n_remove.value = len(removed)
        self._redraw_damage()

    def restore_all(self, *_):
        self.skirt.restore_all()
        self.n_remove.value = 0
        self.damage_preset.value = "Normal"
        self._clear_warnings()
        self._redraw_damage()

    # ---- 발사 프리셋 · 자세 ----------------------------------------------
    def apply_launch_preset(self, *_):
        p = self.S.PRESETS.get(self.preset.value)
        if not p:
            return
        self.V0.value = float(p["V0"])
        self.launch_angle.value = float(p["launch_angle_deg"])
        self.height.value = float(p["height"])
        self.alpha_eq.value = float(p["alpha_eq_deg"])
        self.elev3d.value = float(p["launch_angle_deg"])
        om = p.get("omega0")
        if om is None:                       # 라켓 타격 프리셋: 충격 스핀 추정
            om = (self.S.IMPACT_SPIN_FACTOR * abs(float(p["V0"]))
                  / self.S.SHUTTLE_LENGTH_M)
        self.omega0.value = float(om)
        a0 = p.get("alpha0_deg")
        if a0:
            self.alpha0.value = float(a0)
            self.orient_preset.value = "Cork Forward"
        self._sync_orientation()

    def _sync_orientation(self, *_):
        """자세 프리셋이 α₀ 의 단일 기준이 되게 한다 (Gradio 판과 같은 규칙)."""
        a = self.S.preset_alpha_deg(self.orient_preset.value)
        if a is not None:
            self.alpha0.value = float(a)
        off = self.S.ORIENTATION_PRESETS.get(self.orient_preset.value)
        if off is not None:
            self.orient_off_3d.value = float(off)
        else:
            self.orient_off_3d.value = float(self.custom_orient.value)
        self._preview_params()

    def _preview_params(self, *_):
        try:
            model, _ = self._build_model(int(self.level.value))
        except Exception as exc:                       # noqa: BLE001 - 미리보기 전용
            self.param_preview.value = _as_html("미리보기 실패: %s" % exc)
            return
        t = model.turnover
        ell = t.aero_length(float(self.rho.value), model.geometry.S)
        self.param_preview.value = _as_html(
            "**Cd:** %.3f &nbsp;|&nbsp; **S:** %.5f m² &nbsp;|&nbsp; **m:** %.4f kg\n"
            "**회전 방정식:** %s &nbsp;|&nbsp; **공기역학 길이 ℓ:** %s &nbsp;|&nbsp; "
            "**Tw:** %s &nbsp;|&nbsp; **ζ:** %s"
            % (model.Cd_model.cd(), model.geometry.S, model.m, t.mode,
               self._fmt(ell, " m"), self._fmt(t.Tw, " s"), self._fmt(t.zeta)))
        self._refresh_status()

    # ---- 수치 검증 · 저장 ------------------------------------------------
    def run_validation(self, *_):
        S = self.S
        with self.verify_out:
            clear_output(wait=True)
            print("실행 중…")
            res = S.ValidationEngine().run_all()
            passed = sum(1 for r in res.values() if r.get("passed"))
            clear_output(wait=True)
            display(W.HTML(_as_html("### ValidationEngine %d/%d 통과"
                                    % (passed, len(res)))))
            for name, r in res.items():
                detail = ", ".join("%s=%s" % (k, self._fmt(v))
                                   for k, v in r.items() if k != "passed")
                display(W.HTML("<div>%s <b>%s</b> — <span style='color:#64748b'>%s"
                               "</span></div>"
                               % ("✅" if r.get("passed") else "❌", name, detail)))
            df = S.physics_verification_suite(self.db)
            n = {k: int((df["결과"] == k).sum()) for k in ("PASS", "WARNING", "FAIL")}
            display(W.HTML(_as_html("### 자동 물리 검증 — PASS %d · WARNING %d · FAIL %d"
                                    % (n["PASS"], n["WARNING"], n["FAIL"]))))
            with pd.option_context("display.max_colwidth", 200):
                display(W.HTML(df.to_html(index=False, escape=False)))

    def run_export(self, *_):
        S, res = self.S, self.state["result"]
        with self.export_out:
            clear_output(wait=True)
            if res is None or not res.valid or len(res.df) == 0:
                print("저장할 결과가 없습니다 — 먼저 시뮬레이션을 실행하세요.")
                return
            stem = "shuttlecock_%s" % time.strftime("%Y%m%d_%H%M%S")
            csv = S.ExportEngine.to_csv(res.df, stem + ".csv")
            meta = dict(version=APP_VERSION, ui=UI_VARIANT, kind=self.state["kind"],
                        summary=S.summarize_result(res),
                        geometry=self.skirt.geometry_state(),
                        launch=dict(V0=self.V0.value, angle_deg=self.launch_angle.value,
                                    height_m=self.height.value,
                                    orientation=self.orient_preset.value),
                        warnings=list(self.warnings))
            js = S.ExportEngine.to_json(meta, stem + ".json")
            html = S.ExportEngine.to_html_plot(self._trajectory_fig(res.df),
                                               stem + "_trajectory.html")
            print("저장 완료:\n  %s\n  %s\n  %s" % (csv, js, html))

    # ---- 버튼 핸들러 ------------------------------------------------------
    def _run_2d_click(self, level):
        def handler(_b=None):
            with self.plot_out:
                clear_output(wait=True)
                print("계산 중…")
            try:
                self.run_2d(level)
            except Exception as exc:                   # noqa: BLE001
                self._push("실행 실패: %s" % exc)
                with self.plot_out:
                    clear_output(wait=True)
                    traceback.print_exc()
                self._refresh_status()
                return
            self.show_result()
        return handler

    def _run_3d_click(self, _b=None):
        with self.plot_out:
            clear_output(wait=True)
            print("계산 중…")
        try:
            self.run_3d()
        except Exception as exc:                       # noqa: BLE001
            self._push("실행 실패: %s" % exc)
            with self.plot_out:
                clear_output(wait=True)
                traceback.print_exc()
            self._refresh_status()
            return
        self.show_result()

    # ---- 탭 구성 (Gradio 판과 같은 순서) ---------------------------------
    def _tab_shuttlecock(self):
        return W.VBox([
            _section("1. 셔틀콕 설정"),
            _note("프로파일을 고르면 질량·Cd·Tw·ζ 가 문헌값으로 채워집니다. "
                  "아래 직접 지정 칸은 비워 두면 프로파일 값을 그대로 씁니다."),
            self.profile, self.skirt_d,
            self.cd_override, self.tw_override, self.zeta_override,
            _section("공기 · 바람"), self.rho, self.u_air_x,
            _section("Turnover(뒤집힘) 모델"),
            _note("aero_pendulum = Cohen 2015 공기역학 진자(권장), "
                  "linear = Tw·ζ 로 표현한 선형 2차계."),
            self.turnover_mode, self.l_gc, self.damping_scale, self.cd_cross,
            self.orient_cd, self.aero_spin,
            _section("현재 모델 파라미터"), self.param_preview,
        ])

    def _tab_damage(self):
        apply_btn = W.Button(description="파손 적용", button_style="warning")
        restore_btn = W.Button(description="전체 복원")
        apply_btn.on_click(self.apply_damage_numeric)
        restore_btn.on_click(self.restore_all)
        self.damage_preset.observe(self.apply_damage_preset, names="value")
        return W.VBox([
            _section("2. 셔틀콕 파손 설정"),
            _note("깃털을 몇 개, 어떤 모양으로 뺄지 정합니다. 한쪽 집중 = 비대칭 파손, "
                  "좌우 대칭 = 대칭 파손. 남은 깃털이 25% 미만이면 모델 적용 범위를 "
                  "벗어나므로 실행이 거부됩니다."),
            self.damage_preset, self.n_remove, self.damage_pattern,
            W.HBox([apply_btn, restore_btn]),
            self.damage_out, self.damage_info,
        ])

    def _tab_launch(self):
        self.preset.observe(self.apply_launch_preset, names="value")
        self.orient_preset.observe(self._sync_orientation, names="value")
        self.custom_orient.observe(self._sync_orientation, names="value")
        for w in (self.V0, self.launch_angle):
            w.observe(lambda _c: self._refresh_status(), names="value")
        return W.VBox([
            _section("3. 발사 조건"),
            _note("프리셋을 고르면 아래 값이 한 번에 채워집니다. "
                  "‘자동 발사장치’ 프리셋은 코르크가 앞을 향한 채 각속도 0 으로 나가므로 "
                  "플립이 일어나지 않습니다."),
            self.preset, self.V0, self.launch_angle, self.height,
            _section("발사 순간의 자세"),
            _note("자세 프리셋이 초기 받음각 α₀ 의 기준입니다. Custom 을 고르면 "
                  "아래 자세각 입력이 쓰입니다."),
            self.orient_preset, self.custom_orient, self.alpha0, self.alpha_eq,
            self.omega0, self.spin0,
            _section("속도 성분"), self.comp_view,
        ])

    def _tab_sim(self):
        b1 = W.Button(description="Level 1 실행 (점질량)")
        b2 = W.Button(description="Level 2 실행 (결합)", button_style="primary")
        b1.on_click(self._run_2d_click(1))
        b2.on_click(self._run_2d_click(2))
        return W.VBox([
            _section("4. 물리 모델 · 시뮬레이션 (2차원)"),
            _note("Level 1 = 항력만 받는 점질량, Level 2 = 병진 + Turnover 회전 결합."),
            self.level, self.solver, self.dt, self.t_max,
            W.HBox([b1, b2]),
        ])

    def _tab_3d(self):
        b3 = W.Button(description="3차원 시뮬레이션 실행", button_style="primary")
        b3.on_click(self._run_3d_click)
        return W.VBox([
            _section("5. 3차원 운동 예측 (6-DOF)"),
            _note("쿼터니언 기반 6자유도. Cd 는 0 보다 커야 합니다 — 0 을 넣으면 "
                  "공기가 없는 포물선이 되어 50 m 를 날아갑니다."),
            self.elev3d, self.azim3d, self.z0_3d, self.orient_off_3d,
            self.cd_3d, self.cl_mode_3d, self.geom_inertia,
            b3,
        ])

    def _tab_result(self):
        return W.VBox([
            _section("6. 결과 분석"),
            self.result_cards, self.plot_out,
        ])

    def _tab_verify(self):
        b = W.Button(description="전체 검증 실행", button_style="info")
        b.on_click(self.run_validation)
        return W.VBox([
            _section("7. 수치 검증"),
            _note("ValidationEngine(해석해 대조 · 수렴성 · 방향성)과 "
                  "물리 검증 스위트를 그대로 실행합니다. 1 분 정도 걸립니다."),
            b, self.verify_out,
        ])

    def _tab_export(self):
        b = W.Button(description="CSV · JSON · HTML 저장")
        b.on_click(self.run_export)
        return W.VBox([
            _section("8. 결과 저장"),
            _note("마지막 실행 결과를 현재 폴더에 저장합니다."),
            b, self.export_out,
        ])

    # ---- 조립 ------------------------------------------------------------
    def build(self):
        tabs = W.Tab()
        children = [self._tab_shuttlecock(), self._tab_damage(), self._tab_launch(),
                    self._tab_sim(), self._tab_3d(), self._tab_result(),
                    self._tab_verify(), self._tab_export()]
        titles = ["1. 셔틀콕", "2. 파손", "3. 발사 조건", "4. 시뮬레이션",
                  "5. 3차원", "6. 결과 분석", "7. 수치 검증", "8. 결과 저장"]
        tabs.children = children
        for i, t in enumerate(titles):
            tabs.set_title(i, t)
        header = W.HTML(
            "<div style='padding:10px 14px;border-radius:10px;"
            "background:linear-gradient(90deg,#1e3a8a,#2563eb);color:#fff'>"
            "<div style='font-size:1.15em;font-weight:700'>"
            "셔틀콕 궤적 · Turnover 시뮬레이터 v%s</div>"
            "<div style='opacity:.85;font-size:.9em'>ipywidgets 판 — Gradio 서버 없이 "
            "노트북 커널에서 바로 동작합니다. 물리 계산은 %s 를 그대로 씁니다.</div>"
            "</div>" % (APP_VERSION, PHYSICS_FILENAME))
        self._redraw_damage()
        self._sync_orientation()
        self._refresh_status()
        return W.VBox([header, self.status, self.warn_box, tabs])


def launch(physics_path=None):
    """노트북에서 호출: ``app = launch()``.

    포트도, 터널도, 공유 링크도 쓰지 않는다. 위젯은 커널의 comm 채널로만
    오가므로 Colab 의 커널 프록시가 SSE 스트림을 못 넘겨서 생기는
    '연결이 끊어졌습니다' 문제가 원리적으로 생기지 않는다.
    """
    app = ShuttlecockApp(physics_path)
    display(app.build())
    return app


if __name__ == "__main__":
    print("이 파일은 Jupyter/Colab 노트북에서 사용합니다:")
    print("    from %s import launch" % Path(__file__).stem)
    print("    app = launch()")
    import contextlib
    import io

    a = ShuttlecockApp()
    with contextlib.redirect_stdout(io.StringIO()):   # 노트북 밖에서는 Output 이 stdout 으로 샌다
        a.build()
        r = a.run_2d(2)
    print("자체 점검 — Level 2 최대 도달 거리: %.2f m"
          % a.S.summarize_result(r)["max_range_m"])
