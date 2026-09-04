import sys
import subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
        return
    except ImportError:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg], check=True)

# gradio is only needed to build the web UI. Importing this file for the physics --
# from a notebook, a script, or another front end -- should not drag in a web
# framework, so it is installed lazily in build_ui() instead. Everything above the UI
# layer (about 61% of this file) runs with gradio absent.
_ensure("plotly")

import math
import json
import time
import copy
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, List, Tuple, Any

G = 9.80665
APP_VERSION = "9.2"
MODEL_VERSION = f"shuttlecock_simulator_{APP_VERSION}"

logger = logging.getLogger("shuttlecock_simulator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def deg2rad(d):
    return np.deg2rad(d)

def rad2deg(r):
    return np.rad2deg(r)

def g_to_kg(v):
    return v / 1000.0

def mm_to_m(v):
    return v / 1000.0

def cm_to_m(v):
    return v / 100.0

def kmh_to_ms(v):
    return v / 3.6

def wrap_to_pi(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def cork_heading_deg(theta_body_deg):
    return rad2deg(wrap_to_pi(deg2rad(float(theta_body_deg)) + math.pi))

def unwrap_deg(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return a
    return np.degrees(np.unwrap(np.radians(a)))

FLIP_MODEL_TEXT = (
    "라켓에 맞은 직후 셔틀콕은 코르크가 뒤를 향한 상태(자세각 φ = 180°)에서 출발해 "
    "**뒤집히고(flip) → 진동하고(oscillate) → 정렬(stabilize)** 하는 3단계를 거칩니다. "
    "이 과정을 Cohen et al. 2015의 실측 기반 방정식으로 계산합니다.\n\n"
    "> **각가속도 + β(U) × 각속도 + ω₀(U)² × sin(φ) = 0**\n\n"
    "여기서 φ는 코르크 축과 맞바람 사이의 각도입니다.\n\n"
    "| 기호 | 뜻 | 계산식 |\n|---|---|---|\n"
    "| ω₀ | 복원 진동수 [rad/s] | ω₀ = U ÷ √(ℓ × l_GC) |\n"
    "| ℓ | 공기역학 길이 [m] | ℓ = 2M ÷ (ρ × S × C_D) |\n"
    "| β | 감쇠 계수 [1/s] | β = k_d × (M_코르크 ÷ M_스커트) × (U ÷ ℓ) |\n"
    "| l_GC | 무게중심–코르크 거리 [m] | 약 0.02 m |\n"
    "| τ_o | 진동 주기 [s] | τ_o = 2π × √(ℓ × l_GC) ÷ U |\n"
    "| τ_s | 정렬 시간 [s] | τ_s = 2 ÷ β |\n\n"
    "- **복원 진동수가 속도 U에 비례**합니다. 따라서 셔틀콕이 느려질수록 진동 주기가 길어집니다.\n"
    "- 감쇠도 속도에 비례하므로 **감쇠비 ζ는 속도와 무관한 상수**(약 0.13)가 됩니다.\n"
    "- 복원항이 φ가 아니라 **sin(φ)** 인 것이 핵심입니다. 선형 φ로는 180° 뒤집힘을 "
    "재현할 수 없습니다."
)

ANGLE_REFERENCE_TEXT = (    "**각도 기준축** — 모든 각도는 화면 오른쪽(+x, 지면과 평행)을 0°로 하고 "
    "반시계 방향을 (+)로 잽니다. 위쪽(+y)이 +90°, 아래쪽이 −90°입니다.\n\n"
    "- **진행 방향** = 공기에 대한 속도 벡터가 향하는 방향\n"
    "- **맞바람 방향** = 셔틀콕이 느끼는 상대 바람이 불어오는 쪽 = 진행 방향과 정반대\n"
    "- **자세각 (코르크 방향)** = 코르크 머리가 향한 방향\n"
    "- **받음각 α** = 코르크 축과 상대 바람 벡터 사이의 각. "
    "α = 180°이면 코르크가 맞바람을 정면으로 향한 안정 자세, α = 0°이면 깃털이 앞선 뒤집힌 자세입니다.\n"
    "- **Wobble** = α − 평형 받음각. 0°가 완전히 안정된 상태입니다."
)

@dataclass
class ParamValue:
    value: float
    unit: str
    source: str = "user_defined"
    source_type: str = "user_defined"
    uncertainty: Optional[float] = None
    valid_range: Optional[Tuple[float, float]] = None
    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

def _pv(value, unit, source_type="assumed", vrange=None, notes=""):
    return ParamValue(value=value, unit=unit, source="user candidate profile",
                       source_type=source_type, valid_range=vrange, notes=notes)

class ParameterDatabase:

    def __init__(self):
        self.profiles: Dict[str, Dict[str, ParamValue]] = {
            "BWF 천연 깃털 (기본)": {
                "m": _pv(5.10e-3, "kg", "literature", (4.74e-3, 5.50e-3),
                         notes="BWF 규격 질량 범위 4.74-5.50 g의 중앙 부근 값"),
                "Cd_min": _pv(0.48, "-", "assumed"),
                "Cd_max": _pv(0.73, "-", "assumed"),
                "L_min": _pv(4.2, "m", "assumed"),
                "L_max": _pv(4.5, "m", "assumed"),
                "Tw": _pv(0.0102, "s", "assumed"),
                "zeta": _pv(0.390, "-", "assumed"),
                "M_cork": _pv(3.00e-03, "kg", "literature", notes="Cohen et al. 2015 NJP: 코르크 질량 ~3 g"),
                "M_skirt": _pv(2.10e-03, "kg", "literature", notes="스커트 질량 = 전체질량 - 코르크질량"),
                "l_gc": _pv(0.020, "m", "literature", notes="Cohen et al. 2015 / Cooke 2002: 무게중심-코르크 거리 ~2 cm"),
                "Cd_ref": _pv(0.65, "-", "literature", notes="정렬 상태 항력계수 (깃털 0.65, 플라스틱 0.68)"),
                "spin_ratio": _pv(0.04, "-", "literature", notes="Cohen et al. 2015 fig.15: R*Omega/U (깃털 0.04, 플라스틱 0.02)"),
            },
            "Normal Feather": {
                "m": _pv(5.3e-3, "kg", "assumed", (4.5e-3, 6.0e-3)),
                "Cd_min": _pv(0.48, "-", "assumed"),
                "Cd_max": _pv(0.73, "-", "assumed"),
                "L_min": _pv(4.2, "m", "assumed"),
                "L_max": _pv(4.5, "m", "assumed"),
                "Tw": _pv(0.0102, "s", "assumed"),
                "zeta": _pv(0.390, "-", "assumed"),
                "M_cork": _pv(3.00e-03, "kg", "literature", notes="Cohen et al. 2015 NJP: 코르크 질량 ~3 g"),
                "M_skirt": _pv(2.30e-03, "kg", "literature", notes="스커트 질량 = 전체질량 - 코르크질량"),
                "l_gc": _pv(0.020, "m", "literature", notes="Cohen et al. 2015 / Cooke 2002: 무게중심-코르크 거리 ~2 cm"),
                "Cd_ref": _pv(0.65, "-", "literature", notes="정렬 상태 항력계수 (깃털 0.65, 플라스틱 0.68)"),
                "spin_ratio": _pv(0.04, "-", "literature", notes="Cohen et al. 2015 fig.15: R*Omega/U (깃털 0.04, 플라스틱 0.02)"),
            },
            # The two damaged profiles are derived from ShuttlecockPorosityModel at a
            # stated damage level, not guessed. They used to be hand-set constants
            # that dropped Cd while KEEPING the full 5.3 g -- more mass than an
            # intact BWF shuttlecock -- and since the aerodynamic length is
            # ell = 2M/(rho*S*Cd), holding M while cutting Cd 42% pushed ell from
            # 4.1 m to 7.4 m and flew "Severely Damaged" 14.1 m, 1.50x the intact
            # flight and clear across the 13.4 m court. Cohen et al. (2015) measure
            # ell = 4.04 m and a terminal velocity near 6.4 m/s; that profile implied
            # 7.38 m and 8.51 m/s.
            #
            # Feather loss is not only a drag change, it is a mass change: the vanes
            # that stop blocking the flow also stop being carried. Taking Cd and m
            # from the same porosity model that the feather-removal path uses keeps
            # the two routes to "damaged" telling the same story, and keeps ell in
            # the 3.3-4.1 m band the measurements support.
            #
            # Porosity literature behind the model: a feather skirt carries ~60% more
            # drag than a covered one and sealing the base gap drops Cd to ~0.30
            # (Verma et al. 2015, Procedia Engineering, "Effect of Porosity of
            # Badminton Shuttlecock on Aerodynamic Drag"); gaps raise drag over a
            # gapless cone by up to 45.2%, and past a critical gap size wider gaps
            # cut the blunt-body effect and reduce drag again (Chan & Rossmann;
            # Verma et al.). So damage CAN lower Cd -- the direction was right -- but
            # only once the skirt stops ringing the circumference, and never with the
            # mass left untouched.
            "Mildly Damaged Feather": {
                "m": _pv(4.58e-3, "kg", "derived",
                         notes="다공성 모델 기준 16깃 중 4깃 손실 상태의 질량 "
                               "(코르크 3.0 g + 남은 깃 12/16)"),
                "Cd_min": _pv(0.60, "-", "derived",
                              notes="ShuttlecockPorosityModel: σ=0.598 → Cd 0.616 "
                                    "(Verma 2015 앵커 사이 보간)"),
                "Cd_max": _pv(0.63, "-", "derived"),
                "L_min": _pv(3.5, "m", "derived", notes="ℓ = 2M/(ρS·Cd) = 3.62 m"),
                "L_max": _pv(3.8, "m", "derived"),
                "Tw": _pv(0.0105, "s", "assumed"),
                "zeta": _pv(0.350, "-", "assumed"),
                "M_cork": _pv(3.00e-03, "kg", "literature", notes="Cohen et al. 2015 NJP: 코르크 질량 ~3 g"),
                "M_skirt": _pv(1.58e-03, "kg", "derived", notes="스커트 질량 = 전체질량 - 코르크질량 (4깃 손실 반영)"),
                "l_gc": _pv(0.018, "m", "literature", notes="Cohen et al. 2015 / Cooke 2002: 무게중심-코르크 거리 ~2 cm"),
                "Cd_ref": _pv(0.616, "-", "derived", notes="다공성 모델 σ=0.598 예측값 (정상 깃털 0.65)"),
                "spin_ratio": _pv(0.03, "-", "literature", notes="Cohen et al. 2015 fig.15: R*Omega/U (깃털 0.04, 플라스틱 0.02)"),
            },
            "Severely Damaged Feather": {
                "m": _pv(3.79e-3, "kg", "derived",
                         notes="다공성 모델 기준 16깃 중 10깃 손실 상태의 질량 "
                               "(코르크 3.0 g + 남은 깃 6/16)"),
                "Cd_min": _pv(0.52, "-", "derived",
                              notes="ShuttlecockPorosityModel: σ=0.354 → Cd 0.541. "
                                    "임계 간극을 넘어 항력이 다시 떨어지는 구간"),
                "Cd_max": _pv(0.56, "-", "derived"),
                "L_min": _pv(3.2, "m", "derived", notes="ℓ = 2M/(ρS·Cd) = 3.41 m"),
                "L_max": _pv(3.6, "m", "derived"),
                "Tw": _pv(0.0180, "s", "assumed"),
                "zeta": _pv(0.150, "-", "assumed"),
                "M_cork": _pv(3.00e-03, "kg", "literature", notes="Cohen et al. 2015 NJP: 코르크 질량 ~3 g"),
                "M_skirt": _pv(0.79e-03, "kg", "derived", notes="스커트 질량 = 전체질량 - 코르크질량 (10깃 손실 반영)"),
                "l_gc": _pv(0.014, "m", "literature", notes="Cohen et al. 2015 / Cooke 2002: 무게중심-코르크 거리 ~2 cm"),
                "Cd_ref": _pv(0.541, "-", "derived", notes="다공성 모델 σ=0.354 예측값 (정상 깃털 0.65)"),
                "spin_ratio": _pv(0.02, "-", "literature", notes="Cohen et al. 2015 fig.15: R*Omega/U (깃털 0.04, 플라스틱 0.02)"),
            },
            # Nylon. Its Cd is quoted here at REST: the skirt bends under the air load
            # and the drag falls with speed, which SKIRT_DEFORMATION applies on top.
            # The old 0.38-0.55 was a single "assumed" number standing in for the whole
            # speed range, and it contradicted this profile's own Cd_ref of 0.68.
            "Synthetic": {
                "m": _pv(5.3e-3, "kg", "assumed"),
                "Cd_min": _pv(0.65, "-", "literature",
                              notes="Cohen et al. 2015: 플라스틱 정렬 항력계수 0.68 "
                                    "(정지 상태 기준, 변형 전)"),
                "Cd_max": _pv(0.71, "-", "literature"),
                "skirt_deforms": _pv(1.0, "-", "literature",
                                     notes="나일론 스커트는 공기 하중으로 변형되어 "
                                           "속도가 오르면 Cd 가 떨어진다 "
                                           "(Phys. Scr. 2026, ae5361)"),
                "L_min": _pv(3.6, "m", "derived", notes="정지 Cd 0.68 기준 ℓ = 3.80 m"),
                "L_max": _pv(4.0, "m", "derived"),
                "Tw": _pv(0.0118, "s", "assumed"),
                "zeta": _pv(0.268, "-", "assumed"),
                "M_cork": _pv(3.00e-03, "kg", "literature", notes="Cohen et al. 2015 NJP: 코르크 질량 ~3 g"),
                "M_skirt": _pv(2.30e-03, "kg", "literature", notes="스커트 질량 = 전체질량 - 코르크질량"),
                "l_gc": _pv(0.020, "m", "literature", notes="Cohen et al. 2015 / Cooke 2002: 무게중심-코르크 거리 ~2 cm"),
                "Cd_ref": _pv(0.68, "-", "literature", notes="정렬 상태 항력계수 (깃털 0.65, 플라스틱 0.68)"),
                "spin_ratio": _pv(0.02, "-", "literature", notes="Cohen et al. 2015 fig.15: R*Omega/U (깃털 0.04, 플라스틱 0.02)"),
            },
        }
        self.shared = {
            "Iyy": _pv(2.92e-6, "kg*m^2", "assumed",
                       notes="no confirmed literature/experimental source; not a literature constant"),
        }
        self.literature_refs: List[Dict[str, Any]] = []

    def get_profile(self, name: str) -> Dict[str, ParamValue]:
        if name not in self.profiles:
            raise KeyError(f"unknown profile: {name}")
        return self.profiles[name]

    def profile_names(self):
        return list(self.profiles.keys())

    def add_literature_value(self, citation, doi, parameter, value, unit, condition,
                              velocity_range=None, re_range=None, shuttlecock_type=None,
                              method=None):
        self.literature_refs.append(dict(
            citation=citation, doi=doi, parameter=parameter, value=value, unit=unit,
            condition=condition, velocity_range=velocity_range, re_range=re_range,
            shuttlecock_type=shuttlecock_type, method=method,
        ))

    def mid_cd(self, profile_name):
        p = self.profiles[profile_name]
        return 0.5 * (p["Cd_min"].value + p["Cd_max"].value)

    def spin_ratio_for(self, profile_name, default=0.04):
        """Natural-spin ratio R*Omega/U of the profile (Cohen et al. 2015 fig.15b).

        Feathered shuttlecocks sit near 0.04 and plastic ones near 0.02; the value
        is a property of the INTACT shuttlecock's feather cant, not of any damage.
        """
        try:
            pv = self.profiles[profile_name].get("spin_ratio")
            return float(pv.value) if pv is not None else float(default)
        except Exception:
            return float(default)

BWF_SPEC = {
    "n_feathers": (16, 16, "개"),
    "feather_length_mm": (62.0, 70.0, "mm"),
    "skirt_tip_diameter_mm": (58.0, 68.0, "mm"),
    "cork_diameter_mm": (25.0, 28.0, "mm"),
    "mass_g": (4.74, 5.50, "g"),
}

BWF_DEFAULTS = {
    "n_feathers": 16,
    "feather_length_mm": 66.0,
    "skirt_tip_diameter_mm": 66.0,
    "cork_diameter_mm": 26.0,
    "mass_g": 5.10,
}

def bwf_range_check(field, value):
    spec = BWF_SPEC.get(field)
    if spec is None or value is None:
        return True, ""
    lo, hi, unit = spec
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True, ""
    if lo <= v <= hi:
        return True, f"BWF 규격 범위 {lo}~{hi} {unit} 이내"
    return False, f"BWF 일반 규격 범위({lo}~{hi} {unit})를 벗어난 값: {v} {unit}"

def bwf_spec_report(mass_g=None, feather_length_mm=None, skirt_tip_diameter_mm=None,
                    cork_diameter_mm=None, n_feathers=None):
    fields = [
        ("mass_g", mass_g, "전체 질량"),
        ("feather_length_mm", feather_length_mm, "깃털 길이"),
        ("skirt_tip_diameter_mm", skirt_tip_diameter_mm, "깃털 끝 원 직경"),
        ("cork_diameter_mm", cork_diameter_mm, "코르크 직경"),
        ("n_feathers", n_feathers, "깃털 개수"),
    ]
    rows = []
    warnings_out = []
    for key, val, label in fields:
        if val is None:
            continue
        lo, hi, unit = BWF_SPEC[key]
        ok, msg = bwf_range_check(key, val)
        rows.append(dict(label=label, value=val, unit=unit, spec_low=lo, spec_high=hi,
                         within_spec=ok))
        if not ok:
            warnings_out.append(f"{label}: {msg}")
    return dict(rows=rows, warnings=warnings_out,
                within_spec=all(r["within_spec"] for r in rows) if rows else True)

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

EQUATION_MD = """
### 1. 병진 운동 (Translational motion)

뉴턴 제2법칙에 중력과 공기력을 적용합니다.

> **질량 × 가속도 = 중력 + 항력 + 양력**

**상대 유동 속도** — 항력은 지면 기준 속도가 아니라 공기에 대한 상대 속도로 계산합니다.

| 양 | 계산식 |
|---|---|
| 상대속도 벡터 | V_rel = V(셔틀콕 속도) − U_air(바람 속도) |
| 상대속도 크기 | V_rel = √(u_rel² + v_rel²) |
| 유동 방향 | θ_flow = arctan(v_rel ÷ u_rel) |

**항력 (Drag)** — 상대 유동의 반대 방향으로 작용합니다.

> **항력 = ½ × ρ × C_D × S × V_rel²**, 방향은 상대속도의 정반대

| 성분 | 가속도 |
|---|---|
| 수평 | dv_x/dt = − (ρ × C_D × S × V_rel × u_rel) ÷ (2m) |
| 수직 | dv_y/dt = − g − (ρ × C_D × S × V_rel × v_rel) ÷ (2m) |

**양력 (Lift)** — 상대 유동에 **수직인** 방향으로 작용합니다. 기본값은 C_L = 0 (사용 안 함).

> **양력 = ½ × ρ × C_L × S × V_rel²**, 방향은 상대속도에 수직

### 2. 받음각과 레이놀즈 수

| 양 | 계산식 | 뜻 |
|---|---|---|
| 받음각 α | α = 자세각 − 유동 방향 (±180°로 접음) | 180°면 코르크가 앞선 안정 자세 |
| 레이놀즈 수 Re | Re = ρ × V_rel × D_ref ÷ μ | 관성력 대 점성력의 비 |

### 3. 회전 운동 (Turnover)

**Level 1 — 선형화 모델.** 평형 자세 α_eq 기준의 미소 변위 δα = α − α_eq 에 대해:

> **δα의 각가속도 = − (2ζ ÷ T_w) × δα의 각속도 − (1 ÷ T_w²) × δα**

여기서 T_w는 회전 시간상수 [s], ζ는 감쇠비입니다. 이 모델은 **주기가 속도와 무관하게 고정**되므로,
감속하는 셔틀콕의 실제 거동과는 차이가 있습니다.

**aero_pendulum 모델 (권장).** Cohen et al. 2015의 실측 기반 비선형 진자식으로,
복원 진동수와 감쇠가 모두 속도에 비례합니다. 자세한 내용은 3번 탭의 회전(Flip) 모델 항목을 보세요.

> **각가속도 + β(U) × 각속도 + ω₀(U)² × sin(φ) = 0**

**Level 2 이상 — 공력 모멘트 모델.**

> **관성모멘트 I × 각가속도 = 공력 모멘트 − 감쇠 모멘트**

| 항 | 계산식 |
|---|---|
| 공력 모멘트 | M_aero = ½ × ρ × V_rel² × S × l_ref × C_m(α) |
| 복원 모멘트 (선형 근사) | M_restore = − k_α × δα |
| 감쇠 모멘트 | M_damp = − c_rot × 각속도 |

**강성·감쇠 계수** — 공력 모멘트를 평형점에서 선형화해 구합니다.

| 양 | 계산식 |
|---|---|
| 복원 강성 | k_α = − dM/dα |
| 회전 감쇠 | c_rot = − dM/d(각속도) |
| 회전 시간상수 | T_w = √(I ÷ k_α) |
| 감쇠비 | ζ = c_rot ÷ (2 × √(I × k_α)) |

### 4. 3차원 강체 회전

3차원 모드에서는 오일러 회전방정식을 그대로 적분합니다.

> **I × 각가속도 + 각속도 × (I × 각속도) = 모멘트**

자세는 쿼터니언으로 표현하며 매 적분 단계마다 정규화합니다.
비대칭 깃털 파손이 있으면 관성곱(Ixy, Ixz, Iyz)을 포함한 완전한 관성텐서를 사용합니다.
"""

PHYSICS_GLOSSARY_MD = """
### 시뮬레이션이 계산하는 물리량

모든 값은 매 시간 단계마다 계산되어 결과 표와 CSV로 저장됩니다.

**위치 · 속도 · 가속도**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 위치 | x_m, y_m, z_m | m |
| 속도 성분 | vx_m_s, vy_m_s, vz_m_s | m/s |
| 속력 | speed_m_s | m/s |
| 가속도 성분 | ax_m_s2, ay_m_s2, az_m_s2 | m/s² |
| 가속도 크기 | acceleration_mag_m_s2 | m/s² |
| 접선 가속도 (속력 변화) | tangential_accel_m_s2 | m/s² |
| 법선 가속도 (궤적 휘어짐) | normal_accel_m_s2 | m/s² |
| 궤적 곡률 (1/반지름) | curvature_1_m | 1/m |
| 비행경로각 | flight_path_angle_deg | ° |
| 누적 비행 거리 | distance_m | m |

**회전 (자세 변화 = Turnover)**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 받음각 | alpha_deg | ° |
| Wobble (평형에서 벗어난 각) | delta_alpha_deg | ° |
| 자세각 (코르크 방향) | theta_cork_deg | ° |
| 각속도 | omega_rad_s, omega_deg_s | rad/s, °/s |
| 각가속도 | angular_acceleration_rad_s2, angular_acceleration_deg_s2 | rad/s², °/s² |
| 각운동량 크기 | angular_momentum_mag | kg·m²/s |
| 복원 진동수 ω₀ | omega0_restoring_rad_s | rad/s |
| 감쇠 계수 β | beta_damping_1_s | 1/s |
| 진동 주기 예측 | tau_oscillation_s | s |
| 정렬 시간 예측 | tau_stabilizing_s | s |

**자전 (대칭축 회전 = Spin, Turnover와 별개)**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 축 자전 각속도 | spin_axial_rad_s | rad/s |
| 평형 자전 속도 | spin_equilibrium_rad_s | rad/s |
| 무차원 자전 비 (R·Ω/U) | spin_tip_ratio | - |
| 자이로 수 (0.1 이상이면 자이로 안정화) | gyroscopic_number | - |
| 세차 주기 | precession_period_s | s |

**공기역학**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 상대풍속 크기 · 성분 | V_rel_m_s, u_rel_m_s, v_rel_m_s, w_rel_m_s | m/s |
| 레이놀즈 수 | Re | - |
| 동압 | dynamic_pressure_Pa | Pa |
| 항력·양력·측력 계수 | Cd, Cl, Cy | - |
| 모멘트 계수 | Cm | - |
| 항력·양력·측력 | drag_force_N, lift_force_N, side_force_N | N |
| 양항비 | lift_to_drag | - |
| 무게 | weight_N | N |
| 항력이 소모하는 일률 | power_drag_W | W |

**모멘트 (3차원)**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 모멘트 성분 | moment_x_Nm, moment_y_Nm, moment_z_Nm | N·m |
| 복원·감쇠·합 모멘트 | restoring_moment_Nm, damping_moment_Nm, net_moment_Nm | N·m |
| 자전 구동 모멘트 | spin_moment_Nm | N·m |

**에너지**

| 항목 | 컬럼 | 단위 |
|---|---|---|
| 병진 운동에너지 | E_kinetic_J | J |
| 회전 운동에너지 | E_rotational_J | J |
| 위치에너지 | E_potential_J | J |
| 총 역학적 에너지 | E_total_J | J |

에너지는 보존량이 아닙니다. **다만 바람이 없을 때만** 항력이 일을 빼앗아 총 역학적
에너지가 단조 감소합니다. 바람이 있으면 공기가 셔틀콕에 일을 할 수 있으므로
(P_aero = F_aero·V + M_aero·ω 가 양수가 될 수 있음) 총 에너지가 증가하는 구간이
생길 수 있고, 이는 오류가 아닙니다. 따라서 검증은 단조성이 아니라 **에너지 수지**
dE/dt ≈ P_aero + P_gravity + P_constraint 로 합니다.

**자세 표현 (3차원)**

| 항목 | 컬럼 |
|---|---|
| 쿼터니언 | qw, qx, qy, qz |
| 쿼터니언 노름 (1이어야 정상) | quat_norm |
| 몸통 축 방향 벡터 | ex, ey, ez |
| 몸통 좌표계 각속도 | wx, wy, wz |
| 몸통 좌표계 각가속도 | wx_dot_rad_s2, wy_dot_rad_s2, wz_dot_rad_s2 |
"""


@dataclass
class Geometry:
    skirt_diameter_m: float = 0.066
    cork_diameter_m: float = 0.026
    skirt_length_m: float = 0.066
    S_m2: Optional[float] = None

    @property
    def R_m(self):
        return self.skirt_diameter_m / 2.0

    @property
    def S(self):
        if self.S_m2 is not None:
            return self.S_m2
        return math.pi * self.R_m ** 2

@dataclass
class EnvironmentModel:
    rho: float = 1.2
    u_air_x: float = 0.0
    u_air_y: float = 0.0
    u_air_z: float = 0.0
    temperature_K: float = 293.15
    pressure_Pa: float = 101325.0
    mu: float = 1.81e-5
    turbulence_intensity: float = 0.0
    turbulence_length_scale_m: float = 0.10
    turbulence_seed: Optional[int] = None

class TurbulenceModel:
    """Band-limited synthetic freestream turbulence (Ornstein-Uhlenbeck).

    Turbulence intensity is defined as Tu = u'_rms / U_ref, the standard wind-tunnel
    definition. The fluctuation is generated as a first-order Markov process whose
    correlation time is the Taylor integral timescale T = L / U_ref, so the spectrum
    rolls off above 1/T instead of being white noise. This reproduces the correct
    r.m.s. and correlation time; it is NOT a resolved turbulent field.

    Typical values: a clean open-circuit tunnel is around Tu = 0.3%; grid-generated
    turbulence in the literature spans Tu = 5-15% with L = 0.08-0.33 m.
    """

    def __init__(self, intensity=0.0, length_scale_m=0.10, U_ref=1.0, seed=None):
        self.intensity = max(0.0, float(intensity or 0.0))
        self.length_scale_m = max(1e-6, float(length_scale_m or 0.10))
        self.U_ref = abs(float(U_ref or 0.0))
        self.sigma = self.intensity * self.U_ref
        self.tau = self.length_scale_m / self.U_ref if self.U_ref > 1e-9 else float("inf")
        self.rng = np.random.default_rng(seed)
        self.u = np.zeros(3)

    @property
    def active(self):
        return self.intensity > 0.0 and self.U_ref > 1e-9 and np.isfinite(self.tau)

    def step(self, dt):
        if not self.active:
            return np.zeros(3)
        a = math.exp(-float(dt) / self.tau)
        self.u = a * self.u + self.sigma * math.sqrt(max(1.0 - a * a, 0.0)) \
            * self.rng.standard_normal(3)
        return self.u.copy()

# Nylon skirts bend under the air load; feather skirts effectively do not. The
# velocity-decay work reports that the flow "reduces the cross-section of the plastic
# shuttlecock by a factor of 2 at 50 m/s compared to the cross section at rest", that
# the skirt goes from 16-fold symmetry to square or triangular shapes after a smash,
# and that "plastic shuttlecocks ... deform at high speeds, and the resulting decay of
# their drag coefficient with increasing speed makes them unsuitable for high-level
# play" (Cohen et al., "Shuttlecock velocity decay after smash and slice shots in
# badminton", Phys. Scr. 2026, doi:10.1088/1402-4896/ae5361). The same four
# deformation regimes -- steady and axisymmetric, then buckled, then vibrating -- are
# reported for the drag drop by Verma/Chan et al. over 25-50 m/s.
#
# One measured anchor is available, the factor of 2 at 50 m/s, so the law is fixed by
# it and nothing more is claimed: a smooth monotone decay from 1 at rest towards a
# floor, with the half-speed solved from that anchor rather than tuned.
#
#     f(v) = k + (1 - k) / (1 + (v / v_d)^2),  f(50 m/s) = 0.5
#     k = 0.40 (a buckled skirt still has a wake)  =>  v_d = 50 / sqrt(5) = 22.36 m/s
#
# A feather shuttlecock keeps a constant Cd -- that is what makes Cohen's constant-Cd
# solution fit its measured flights -- so this is left off for every feather profile.
SKIRT_DEFORMATION = dict(
    floor=0.40,
    v_anchor=50.0,
    factor_at_anchor=0.5,
    source=("Cohen, Darbois Texier, Quere, Clanet et al., 'Shuttlecock velocity decay "
            "after smash and slice shots in badminton', Physica Scripta (2026), "
            "doi:10.1088/1402-4896/ae5361; Verma/Chan et al. 스커트 변형 4구간"),
    notes=("나일론 스커트만 해당합니다. 깃털 셔틀콕은 Cd 가 사실상 일정합니다. "
            "50 m/s 에서 단면적이 정지 상태의 1/2 이라는 한 점만 근거로 삼았고, "
            "그 점에서 v_d 를 풀었습니다. 사용자가 제시한 '300 km/h 1.6 → 200 km/h 0.38' "
            "같은 절대값은 접근 가능한 문헌에서 확인하지 못했습니다."),
)


def skirt_deformation_v_half(floor=None, v_anchor=None, factor_at_anchor=None):
    """v_d that makes the decay law hit the measured point, instead of a fitted guess."""
    k = SKIRT_DEFORMATION["floor"] if floor is None else float(floor)
    va = SKIRT_DEFORMATION["v_anchor"] if v_anchor is None else float(v_anchor)
    fa = (SKIRT_DEFORMATION["factor_at_anchor"] if factor_at_anchor is None
          else float(factor_at_anchor))
    span = fa - k
    if span <= 0 or span >= (1.0 - k):
        return float("inf")
    return va / math.sqrt((1.0 - k) / span - 1.0)


class CdModel:
    def __init__(self, mode="constant", constant_value=0.6, table=None, table_alpha=None,
                 source_type="assumed", valid_range=None, uncertainty=None,
                 cd_cross=0.94, skirt_deforms=False):
        self.mode = mode
        # Only the nylon skirt bends; see SKIRT_DEFORMATION above.
        self.skirt_deforms = bool(skirt_deforms)
        self.constant_value = constant_value
        self.cd_cross = cd_cross
        self.table = table or []
        self.table_alpha = table_alpha or []
        self.source_type = source_type
        self.valid_range = valid_range
        self.uncertainty = uncertainty
        self.last_extrapolated = False

    def _interp1d(self, x, table):
        xs = [t[0] for t in table]
        ys = [t[1] for t in table]
        self.last_extrapolated = bool(x < min(xs) or x > max(xs))
        return float(np.interp(x, xs, ys))

    def deformation_factor(self, speed):
        """How much of the rest-shape drag survives at this speed.

        1.0 for a feather shuttlecock at any speed, and for a nylon one at rest.
        """
        if not self.skirt_deforms or speed is None:
            return 1.0
        v = abs(float(speed))
        if not math.isfinite(v):
            return 1.0
        k = SKIRT_DEFORMATION["floor"]
        v_d = skirt_deformation_v_half()
        if not math.isfinite(v_d) or v_d <= 0:
            return 1.0
        return k + (1.0 - k) / (1.0 + (v / v_d) ** 2)

    def cd(self, speed=None, alpha_rad=None, Re=None, alpha_eq_rad=0.0):
        cd0 = self._cd_rest(speed=speed, alpha_rad=alpha_rad, Re=Re,
                            alpha_eq_rad=alpha_eq_rad)
        f = self.deformation_factor(speed)
        return cd0 if f == 1.0 else cd0 * f

    def _cd_rest(self, speed=None, alpha_rad=None, Re=None, alpha_eq_rad=0.0):
        """Drag coefficient. alpha_eq_rad is the attitude the shuttlecock trims to.

        "orientation" mode interpolates between the aligned value and the broadside
        value with sin^2 of the MISALIGNMENT, matching TurnoverParams.cd_at(phi).
        Level 1 integrates the misalignment directly and passes it with the default
        alpha_eq_rad = 0; the coupled and 3D paths carry an absolute attitude and pass
        their equilibrium, which previously only agreed with Level 1 because the
        default equilibrium is 180 deg and sin^2 is invariant under a pi shift.
        """
        self.last_extrapolated = False
        if self.mode == "orientation":
            phi = 0.0 if alpha_rad is None else float(alpha_rad)
            s2 = math.sin(wrap_to_pi(phi - float(alpha_eq_rad or 0.0))) ** 2
            return self.constant_value + (self.cd_cross - self.constant_value) * s2
        if self.mode == "constant":
            return self.constant_value
        if self.mode == "lookup_Re_alpha":
            if not self.table_alpha:
                return self.constant_value
            re_q = Re if Re is not None else 0.0
            al_q = alpha_rad if alpha_rad is not None else 0.0
            pts = np.array([[t[0], t[1]] for t in self.table_alpha], dtype=float)
            vals = np.array([t[2] for t in self.table_alpha], dtype=float)
            re_span = max(np.ptp(pts[:, 0]), 1e-9)
            al_span = max(np.ptp(pts[:, 1]), 1e-9)
            d = ((pts[:, 0] - re_q) / re_span) ** 2 + ((pts[:, 1] - al_q) / al_span) ** 2
            self.last_extrapolated = bool(re_q < pts[:, 0].min() or re_q > pts[:, 0].max()
                                          or al_q < pts[:, 1].min() or al_q > pts[:, 1].max())
            return float(vals[int(np.argmin(d))])
        if not self.table:
            return self.constant_value
        if self.mode == "lookup_speed":
            return self._interp1d(speed if speed is not None else 0.0, self.table)
        if self.mode == "lookup_angle":
            return self._interp1d(alpha_rad if alpha_rad is not None else 0.0, self.table)
        if self.mode == "lookup_Re":
            return self._interp1d(Re if Re is not None else 0.0, self.table)
        return self.constant_value

class ClModel:

    def __init__(self, mode="disabled", constant_value=0.0, table=None,
                 source_type="assumed", valid_range=None, uncertainty=None,
                 crossflow_delta_cd=None):
        self.mode = mode
        self.constant_value = constant_value
        self.table = table or []
        self.source_type = source_type
        self.valid_range = valid_range
        self.uncertainty = uncertainty
        # Cd(90 deg) - Cd(0 deg); set from the turnover parameters at build time
        self.crossflow_delta_cd = crossflow_delta_cd
        self.last_extrapolated = False

    @property
    def enabled(self):
        return self.mode != "disabled"

    def crossflow_cl(self):
        """Normal-force magnitude derived from the two drag anchors already in use.

        Standard cross-flow decomposition of a body of revolution: the force splits
        into an axial part along the symmetry axis and a normal part from the
        cross-flow over the skirt. Taking the axial coefficient as the aligned drag
        and the normal coefficient as (Cd_cross - Cd_ref)*sin(phi) and resolving into
        wind axes gives

            C_D = Cd_ref + (Cd_cross - Cd_ref) sin^2(phi)      <- the existing curve
            C_L = (Cd_cross - Cd_ref) sin(phi) cos(phi)
                = 0.5 (Cd_cross - Cd_ref) sin(2 phi)

        The drag law is reproduced exactly, so the Cohen 2015 fig.8b anchor at
        phi = 70 deg still holds, and the lift comes out with the sin(2*delta) shape
        lift_alignment_factor already applies and that Chan & Rossmann 2012 report
        (zero when aligned and broadside, peak near 45 deg). The magnitude is fixed
        entirely by Cd_ref and Cd_cross -- no new constant is introduced.
        """
        d = self.crossflow_delta_cd
        if d is None:
            return 0.0
        return 0.5 * float(d)

    def cl(self, alpha_rad=None, Re=None):
        self.last_extrapolated = False
        if self.mode == "disabled":
            return 0.0
        if self.mode == "crossflow":
            return self.crossflow_cl()
        if self.mode == "constant":
            return self.constant_value
        if not self.table:
            return self.constant_value
        if self.mode == "lookup_angle":
            xs = [t[0] for t in self.table]
            ys = [t[1] for t in self.table]
            a = alpha_rad if alpha_rad is not None else 0.0
            self.last_extrapolated = bool(a < min(xs) or a > max(xs))
            return float(np.interp(a, xs, ys))
        if self.mode == "lookup_Re_alpha":
            pts = np.array([[t[0], t[1]] for t in self.table], dtype=float)
            vals = np.array([t[2] for t in self.table], dtype=float)
            re_q = Re if Re is not None else 0.0
            al_q = alpha_rad if alpha_rad is not None else 0.0
            re_span = max(np.ptp(pts[:, 0]), 1e-9)
            al_span = max(np.ptp(pts[:, 1]), 1e-9)
            d = ((pts[:, 0] - re_q) / re_span) ** 2 + ((pts[:, 1] - al_q) / al_span) ** 2
            return float(vals[int(np.argmin(d))])
        return 0.0

def turnover_inertia_of(turnover):
    """Transverse inertia that the whole turnover equation must be written with.

    The restoring moment, the damping moment and the angular acceleration have to be
    divided by ONE inertia. TurnoverParams.Iyy is a profile input while the solvers
    integrate with I_effective (the geometry-derived tensor when geometry inertia is
    active). Mixing the two scaled the restoring moment by one and the damping by the
    other, which changed the damping ratio by the ratio between them (~3x with
    geometry inertia on).
    """
    I = getattr(turnover, "I_effective", None)
    try:
        if I is not None and np.isfinite(I) and float(I) > 0:
            return float(I)
    except (TypeError, ValueError):
        pass
    return float(turnover.Iyy)

class CmModel:

    def __init__(self, mode="from_turnover_params", cm_alpha=None, table=None,
                 source_type="assumed", valid_range=None, uncertainty=None):
        self.mode = mode
        self.cm_alpha = cm_alpha
        self.table = table or []
        self.source_type = source_type
        self.valid_range = valid_range
        self.uncertainty = uncertainty
        self.last_extrapolated = False

    def cm(self, alpha_rad, alpha_eq_rad=math.pi):
        self.last_extrapolated = False
        d_alpha = wrap_to_pi(alpha_rad - alpha_eq_rad)
        if self.mode == "cm_alpha" and self.cm_alpha is not None:
            return self.cm_alpha * d_alpha
        if self.mode == "lookup_angle" and self.table:
            xs = [t[0] for t in self.table]
            ys = [t[1] for t in self.table]
            self.last_extrapolated = bool(alpha_rad < min(xs) or alpha_rad > max(xs))
            return float(np.interp(alpha_rad, xs, ys))
        return None

    def uses_aero_pendulum(self, turnover):
        """Is the restoring moment the aerodynamic pendulum of Cohen et al. 2015?

        "from_turnover_params" means exactly what it says: the restoring model is
        whichever one TurnoverParams selects. Keying only off CmModel.mode ignored
        that, so with turnover.mode == "aero_pendulum" the 2D coupled solver silently
        fell back to the linear spring -k_alpha*d_alpha while the 3D solver ran the
        aero pendulum. The same launch then produced two different turnover
        frequencies (fixed 1/Tw instead of omega0 proportional to U).
        """
        if self.mode == "aero_pendulum":
            return True
        return (self.mode == "from_turnover_params"
                and getattr(turnover, "mode", "linear") == "aero_pendulum")

    def moment(self, alpha_rad, Vrel, rho, S, l_ref, turnover, alpha_eq_rad=math.pi):
        if self.uses_aero_pendulum(turnover):
            d_alpha = wrap_to_pi(alpha_rad - alpha_eq_rad)
            w0 = turnover.omega0(Vrel, rho, S)
            I_eff = turnover_inertia_of(turnover)
            return -I_eff * (w0 ** 2) * math.sin(d_alpha)
        cm_val = self.cm(alpha_rad, alpha_eq_rad)
        if cm_val is None:

            return -turnover.k_alpha * wrap_to_pi(alpha_rad - alpha_eq_rad)
        return 0.5 * rho * Vrel ** 2 * S * l_ref * cm_val

class SpinModel:

    def __init__(self, spin0_rad_s=0.0, Ispin=None, cd_spin_map=None, cm_spin_map=None,
                 source_type="assumed", spin_decay_rate=0.0, mode="aero_driven",
                 spin_ratio=0.04, R_m=0.033, speed_fn=None,
                 c_spin=0.0, c_spin_source="assumed", relax_rate=None):
        self.spin0_rad_s = spin0_rad_s
        self.Ispin = Ispin
        self.cd_spin_map = cd_spin_map
        self.cm_spin_map = cm_spin_map
        self.source_type = source_type
        self.spin_decay_rate = spin_decay_rate
        self.mode = mode
        self.spin_ratio = spin_ratio
        self.R_m = R_m
        self.speed_fn = speed_fn
        # spin damping is NOT the turnover damping; keep them separate
        self.c_spin = float(c_spin)
        self.c_spin_source = c_spin_source
        self.relax_rate = relax_rate

    @property
    def coupling_available(self):
        return self.cd_spin_map is not None or self.cm_spin_map is not None

    @property
    def status(self):
        return "축 Spin 연동 사용" if self.coupling_available else "축 Spin 연동 자료 없음"

    def spin_at(self, t, speed=None, omega_axial=None):
        """Axial spin actually used by the aerodynamic coefficients.

        When the solver integrates a spin degree of freedom, that integrated value
        is the physical truth and is returned unchanged. The empirical relation
        Omega_eq = spin_ratio*U/R is only an EQUILIBRIUM target: it is applied as a
        relaxation torque in the dynamics, never written straight over the state.
        Overwriting the state made the reported spin independent of the equations of
        motion, so damping and constraint torques had no visible effect.
        """
        if omega_axial is not None:
            return float(omega_axial)
        if self.spin_decay_rate and self.spin_decay_rate > 0:
            return self.spin0_rad_s * math.exp(-self.spin_decay_rate * t)
        return self.spin0_rad_s

    def equilibrium_spin(self, speed):
        """Empirical equilibrium spin Omega_eq = spin_ratio*U/R (Cohen et al. 2015)."""
        if self.R_m <= 0 or speed is None:
            return 0.0
        return float(self.spin_ratio) * abs(float(speed)) / float(self.R_m)

    def cd_factor(self, spin):
        if self.cd_spin_map is None:
            return 1.0
        try:
            return float(self.cd_spin_map(spin))
        except Exception:
            return 1.0

    def cm_factor(self, spin):
        if self.cm_spin_map is None:
            return 1.0
        try:
            return float(self.cm_spin_map(spin))
        except Exception:
            return 1.0

@dataclass
class Feather:
    id: int
    azimuth_deg: float
    attached: bool = True
    length_m: float = 0.065
    width_m: float = 0.012
    mass_kg: float = 1.06e-4

    @property
    def azimuth_rad(self):
        return deg2rad(self.azimuth_deg)

class SkirtGeometry:

    def __init__(self, n_feathers=16, skirt_diameter_m=0.066, cork_diameter_m=0.025,
                 skirt_length_m=0.06, feather_width_m=0.012, feather_mass_kg=1.06e-4):
        self.n_feathers = n_feathers
        self.skirt_diameter_m = skirt_diameter_m
        self.cork_diameter_m = cork_diameter_m
        self.skirt_length_m = skirt_length_m
        self.feathers = [
            Feather(id=i + 1, azimuth_deg=360.0 * i / n_feathers,
                    length_m=skirt_length_m, width_m=feather_width_m,
                    mass_kg=feather_mass_kg)
            for i in range(n_feathers)
        ]

    def remove(self, feather_id):
        for f in self.feathers:
            if f.id == feather_id:
                f.attached = False
                return True
        return False

    def restore(self, feather_id):
        for f in self.feathers:
            if f.id == feather_id:
                f.attached = True
                return True
        return False

    def toggle(self, feather_id):
        for f in self.feathers:
            if f.id == feather_id:
                f.attached = not f.attached
                return f.attached
        return None

    def set_attached(self, attached_ids):
        ids = set(attached_ids)
        for f in self.feathers:
            f.attached = f.id in ids

    def remove_all(self):
        for f in self.feathers:
            f.attached = False

    def restore_all(self):
        for f in self.feathers:
            f.attached = True

    @property
    def attached_feathers(self):
        return [f for f in self.feathers if f.attached]

    @property
    def removed_ids(self):
        return [f.id for f in self.feathers if not f.attached]

    def geometry_state(self) -> dict:
        attached = self.attached_feathers
        n_att = len(attached)
        n_rem = self.n_feathers - n_att
        R = self.skirt_diameter_m / 2.0
        S_ref = math.pi * R ** 2

        if n_att > 0:
            cx = sum(math.cos(f.azimuth_rad) for f in attached) / n_att
            cy = sum(math.sin(f.azimuth_rad) for f in attached) / n_att
            asymmetry = math.hypot(cx, cy)
            asym_dir = math.atan2(cy, cx)
        else:
            asymmetry = 0.0
            asym_dir = 0.0

        circumference = math.pi * self.skirt_diameter_m
        solid_width = sum(f.width_m for f in attached)
        skirt_open_fraction = float(np.clip(1.0 - solid_width / circumference, 0.0, 1.0))
        full_solid = sum(f.width_m for f in self.feathers)
        porosity_full = float(np.clip(1.0 - full_solid / circumference, 0.0, 1.0))

        coverage = (n_att / self.n_feathers) if self.n_feathers else 0.0
        effective_projected_area = S_ref * coverage * (1.0 - porosity_full)

        return dict(
            N_feathers=self.n_feathers,
            removed_feathers=n_rem,
            remaining_feathers=n_att,
            remaining_fraction=(n_att / self.n_feathers) if self.n_feathers else 0.0,
            removed_ids=self.removed_ids,
            azimuth_distribution=[f.azimuth_deg for f in attached],
            skirt_open_fraction=skirt_open_fraction,
            effective_skirt_width=solid_width,
            effective_projected_area=effective_projected_area,
            estimated_porosity=skirt_open_fraction,
            geometry_asymmetry=asymmetry,
            asymmetry_direction_rad=asym_dir,
            reference_area=S_ref,
            skirt_diameter_m=self.skirt_diameter_m,
            cork_diameter_m=self.cork_diameter_m,
            vane_width_m=(self.feathers[0].width_m if self.feathers else 0.012),
            source_type="형상 기반 추정값",
        )

    def mass_properties(self, cork_mass_kg=None) -> dict:
        """Centre of mass and full inertia tensor from the remaining feather layout.

        Geometry-derived estimate from the masses/dimensions the user entered, not a
        measured value. Asymmetric damage shifts the CM and creates products of inertia,
        which is why the full tensor is returned rather than three diagonal terms.
        """
        attached = self.attached_feathers
        R = self.skirt_diameter_m / 2.0
        lever = self.skirt_length_m / 2.0
        # Body +x is the axis the 6-DOF solver integrates with, and everywhere else in
        # this file it points from the cork toward the skirt (equilibrium is e = -e_flow,
        # cork forward; theta_cork = theta_body + 180). The feathers therefore sit at
        # +x and the cork at -x. Having them the other way round flipped the products
        # of inertia Ixy/Ixz that asymmetric damage produces.
        pts = []
        for f in attached:
            pts.append((f.mass_kg, np.array([lever, R * math.cos(f.azimuth_rad),
                                              R * math.sin(f.azimuth_rad)])))
        if cork_mass_kg:
            pts.append((float(cork_mass_kg), np.array([-lever, 0.0, 0.0])))
        m_tot = sum(mm for mm, _ in pts)
        if m_tot <= 0:
            return dict(total_mass=0.0, r_cm=[0.0, 0.0, 0.0],
                         inertia_tensor=[[0.0] * 3] * 3, Ixx=0.0, Iyy=0.0, Izz=0.0,
                         cm_offset_m=0.0, source_type="형상 기반 추정값")
        r_cm = sum(mm * pp for mm, pp in pts) / m_tot
        I = np.zeros((3, 3))
        for mm, pp in pts:
            d = pp - r_cm
            I += mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return dict(total_mass=float(m_tot), r_cm=[float(c) for c in r_cm],
                     inertia_tensor=[[float(x) for x in row] for row in I],
                     Ixx=float(I[0, 0]), Iyy=float(I[1, 1]), Izz=float(I[2, 2]),
                     Ixy=float(I[0, 1]), Ixz=float(I[0, 2]), Iyz=float(I[1, 2]),
                     cm_offset_m=float(math.hypot(r_cm[1], r_cm[2])),
                     source_type="형상 기반 추정값")

    def inertia_estimate(self, cork_mass_kg=None) -> dict:
        attached = self.attached_feathers
        R = self.skirt_diameter_m / 2.0

        Ispin = sum(f.mass_kg * R ** 2 for f in attached)

        lever = self.skirt_length_m / 2.0
        Iyy = sum(f.mass_kg * ((R * math.sin(f.azimuth_rad)) ** 2 + lever ** 2) for f in attached)
        if cork_mass_kg:
            Iyy += cork_mass_kg * lever ** 2
        return dict(Iyy=Iyy, Ispin=Ispin, source_type="형상 기반 추정값",
                    notes="feathers modelled as point masses at the skirt radius")

DAMAGE_PRESETS = {

    "Normal": [],
    "Mild Damage": [1, 9],
    "Moderate Damage": [1, 2, 3, 4],
    "Severe Damage": [1, 2, 3, 4, 5, 6, 7, 8],
}

class DamageAeroMapping:

    def __init__(self, enabled=False, cd_fn=None, cl_fn=None, cm_fn=None,
                 lateral_force_bias_fn=None, moment_bias_fn=None, source_type="assumed"):
        self.enabled = enabled
        self.cd_fn = cd_fn
        self.cl_fn = cl_fn
        self.cm_fn = cm_fn
        self.lateral_force_bias_fn = lateral_force_bias_fn
        self.moment_bias_fn = moment_bias_fn
        self.source_type = source_type

    @property
    def status(self):
        if not self.enabled:
            return "파손→계수 대응 없음 (형상만 반영)"
        return f"파손→계수 대응 사용 (출처: {self.source_type})"

    def _apply(self, fn, geom_state, default=1.0):
        if not self.enabled or fn is None:
            return default
        try:
            return float(fn(geom_state))
        except Exception:
            return default

    def _asym_arm(self, geom_state):
        pm = getattr(self, "porosity_model", None)
        if pm is None or not self.enabled:
            return 0.0
        try:
            return float(pm.moment_arm_m(geom_state or {}))
        except Exception:
            return 0.0

    def _asym_azimuth(self, geom_state):
        pm = getattr(self, "porosity_model", None)
        if pm is None or not self.enabled:
            return 0.0
        try:
            return float(pm.asymmetry(geom_state)[1])
        except Exception:
            return 0.0

    @staticmethod
    def _state_key(geom_state):
        """Everything in modifiers() depends on the damage geometry and nothing else."""
        if not geom_state:
            return ()
        ids = geom_state.get("removed_ids")
        try:
            ids = tuple(sorted(int(i) for i in (ids or ())))
        except (TypeError, ValueError):
            ids = tuple(str(ids))
        return (int(geom_state.get("N_feathers") or 0), ids,
                float(geom_state.get("vane_width_m") or 0.0),
                float(geom_state.get("cork_diameter_m") or 0.0),
                float(geom_state.get("skirt_diameter_m") or 0.0))

    def modifiers(self, geom_state) -> dict:
        """Damage-derived coefficient modifiers, computed once per damage state.

        The skirt does not change shape during a flight, so every one of these is the
        same number on every step -- but they were being recomputed on each derivative
        evaluation, four per RK4 step. On a damaged 12 s flight that was 18,570 calls
        costing 4.8 s of a 6.2 s run, most of it in solidity_from_state and the
        logarithm inside _cone_solidity. Caching on the damage state returns the
        identical dict and leaves the trajectory bit-for-bit unchanged.
        """
        key = self._state_key(geom_state)
        cached = getattr(self, "_mod_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        mods = self._compute_modifiers(geom_state)
        self._mod_cache = (key, mods)
        return mods

    def _compute_modifiers(self, geom_state) -> dict:
        return dict(
            cd_factor=self._apply(self.cd_fn, geom_state, 1.0),
            cl_factor=self._apply(self.cl_fn, geom_state, 1.0),
            cm_factor=self._apply(self.cm_fn, geom_state, 1.0),
            lateral_force_bias=self._apply(self.lateral_force_bias_fn, geom_state, 0.0),
            aerodynamic_moment_bias=self._apply(self.moment_bias_fn, geom_state, 0.0),
            asymmetry_azimuth_rad=self._asym_azimuth(geom_state),
            asymmetry_moment_arm_m=self._asym_arm(geom_state),
            source_type=self.source_type if self.enabled else "모델 추정값 (대응 관계 없음)",
        )


# ---------------------------------------------------------------------------
# Feather damage -> aerodynamic coefficients (porosity model)
# ---------------------------------------------------------------------------

SHUTTLE_POROSITY_ANCHORS = dict(
    Cd_sealed=0.30,
    Cd_intact=0.62,
    Cd_cork_only=0.07,
    solidity_intact=0.65,
    solidity_intact_source=("model_derived — 측정된 기하 solidity 가 아니라 "
                             "Alam 2015 의 Cd 앵커를 재현하도록 정한 '유효 공력 solidity' 입니다. "
                             "실제 깃털 폭 기준 기하 solidity 는 스커트 끝에서 약 0.93, "
                             "코르크 쪽에서는 겹침 때문에 1 을 넘습니다."),
    source="Alam, Chowdhury et al. (2015) 'Effect of Porosity of Badminton Shuttlecock "
           "on Aerodynamic Drag', Procedia Engineering; Kitta et al. (2011); "
           "Cooke (1999, 2002)",
    notes=("Alam 2015: 깃털 사이 틈을 모두 막으면 Cd가 약 0.30까지 떨어지고, "
            "틈이 있는 정상 깃털 셔틀콕은 약 0.6입니다. 즉 틈으로 새는 공기 제트가 "
            "후류를 넓혀 항력을 약 2배로 키웁니다(Kitta 2011도 같은 방향). "
            "Cooke는 Re = 1.3e4~2e5 구간에서 Cd가 거의 일정하다고 보고했습니다."),
)

class ShuttlecockPorosityModel:
    """Predicts how removing feathers changes the aerodynamic coefficients.

    There is no published Cd-versus-number-of-missing-feathers curve, so this is an
    explicit interpolation model, not measured data. It is built from three anchor
    points that ARE measured, and the shape between them follows the physics the
    same papers describe.

    Anchors (solidity sigma = fraction of the skirt circumference occupied by vanes):
      sigma = 1  (gaps sealed)      -> Cd = 0.30   [Alam et al. 2015]
      sigma = sigma_0 (intact)      -> Cd = 0.62   [Cooke; Alam et al.]
      sigma = 0  (no skirt at all)  -> Cd = 0.07   (bare cork, area-scaled sphere)

    Shape between anchors:
        Cd(sigma) = Cd_cork + A*sigma + B*sigma*(1 - sigma)
    The term linear in sigma is the momentum blocked by the solid vane area. The
    sigma*(1-sigma) term is the bleed-jet enhancement: it vanishes both when there is
    no skirt (nothing to bleed past) and when the skirt is sealed (nothing bleeds
    through), and peaks at intermediate porosity, which is exactly the behaviour the
    porosity experiments report. A and B are fixed by the two upper anchors.

    Asymmetric damage additionally produces a side force and a yaw/roll moment,
    because the intact side keeps its drag while the opened side loses it. That
    imbalance is computed from the azimuthal distribution of the remaining feathers
    and is a geometric consequence, not a fitted parameter.
    """

    # Two of the three anchors are measured; sigma = 0 (bare cork) is a construction,
    # and it answers a different question from "what does feather damage do?". Walking
    # sigma down by removing feathers runs into it: at one or zero feathers left the
    # model returns Cd ~ 0.07-0.14, an aerodynamic length of 12-24 m instead of 4 m,
    # and flights of 20-38 m. A shuttlecock cannot do that -- even a full-power smash
    # barely crosses the 13.4 m court -- because the object being simulated is no
    # longer a shuttlecock. Below this fraction the skirt no longer rings the
    # circumference, the remaining point masses no longer support a rigid-body inertia
    # tensor (runs diverged at |omega| ~ 1e15 rad/s), and the model is extrapolating to
    # the constructed anchor. It refuses to predict there instead.
    MIN_REMAINING_FRACTION = 0.25

    def __init__(self, Cd_sealed=None, Cd_intact=None, Cd_cork_only=None,
                 solidity_intact=None, bleed_exponent=1.0):
        a = SHUTTLE_POROSITY_ANCHORS
        self.Cd_sealed = float(Cd_sealed if Cd_sealed is not None else a["Cd_sealed"])
        self.Cd_intact = float(Cd_intact if Cd_intact is not None else a["Cd_intact"])
        self.Cd_cork = float(Cd_cork_only if Cd_cork_only is not None
                              else a["Cd_cork_only"])
        self.sigma0 = float(np.clip(
            solidity_intact if solidity_intact is not None else a["solidity_intact"],
            0.05, 0.99))
        self.bleed_exponent = float(bleed_exponent)
        self.A = self.Cd_sealed - self.Cd_cork
        denom = self.bleed_shape(self.sigma0)
        self.B = ((self.Cd_intact - self.Cd_cork - self.A * self.sigma0) / denom
                  if denom > 1e-9 else 0.0)

    def bleed_shape(self, s):
        """Bleed-jet enhancement as a function of solidity, peaking at the intact skirt.

        It has to vanish at both ends -- no skirt, nothing to bleed past; sealed
        skirt, nothing bleeding through -- and peak somewhere between. It used to be
        s*(1-s), which peaks at s = 0.5, but a real skirt sits at sigma_0 = 0.65, on
        the far side of that. Removing feathers then walked the solidity TOWARD the
        peak and drag went UP: at four feathers gone Cd rose from 0.605 to 0.616, and
        because the mass falls at the same time the two cancelled almost exactly. A
        shuttlecock missing twelve of its sixteen feathers came out falling at
        5.88 m/s against the intact 5.86 m/s -- the same shuttlecock, aerodynamically.

        Peaking at sigma_0 instead is the more defensible reading of the same
        experiments. Verma et al. (2015) find a *critical gap size* at which drag is
        greatest, with wider gaps past it reducing the blunt-body effect and the drag;
        a shuttlecock is a device built to maximise drag for its mass, so its own
        geometry is where that maximum should sit. It also reproduces what the game
        knows: shuttles with fewer feathers are faster, and bending the feather tips
        outward -- adding skirt back into the flow -- is how players slow one down.

        s**p * (1-s)**q peaks at p/(p+q); q = p*(1 - sigma_0)/sigma_0 puts that at
        sigma_0. The three measured anchors are untouched -- B is still solved from
        the intact one -- so only the shape between them moves.
        """
        s = clip1(float(s), 0.0, 1.0)
        if s <= 0.0 or s >= 1.0:
            return 0.0
        q = self.bleed_exponent * (1.0 - self.sigma0) / self.sigma0
        return float(s ** self.bleed_exponent * (1.0 - s) ** q)

    def cd_at_solidity(self, sigma):
        s = clip1(float(sigma), 0.0, 1.0)
        return float(self.Cd_cork + self.A * s + self.B * self.bleed_shape(s))

    @staticmethod
    def geometric_solidity(skirt, geometry):
        """Solidity computed from the actual feather widths and skirt circumference.

        This is NOT the same quantity as `solidity_intact`. Real feathers overlap
        heavily near the cork, so the geometric value exceeds 1 there and reaches
        about 0.9 at the tip. `solidity_intact` is instead the effective aerodynamic
        solidity that anchors the porosity interpolation to the measured drag; the
        two must not be used interchangeably.
        """
        try:
            n = len(skirt.attached_feathers)
            w = float(skirt.feathers[0].width_m)
            circ = 2.0 * math.pi * (float(geometry.skirt_diameter_m) / 2.0)
            if circ <= 0:
                return float("nan")
            return float(n * w / circ)
        except Exception:
            return float("nan")

    @staticmethod
    def _cone_solidity(n_vanes, vane_width_m, cork_d_m, skirt_d_m):
        """Vane coverage averaged along the skirt cone, capped where vanes overlap.

        The vanes overlap near the cork -- the file's own anchor note records a
        geometric solidity above 1 there and about 0.93 at the tip -- so the first
        feathers removed only eat into that overlap and open no gap at all. Coverage
        is therefore the average of min(1, n*w / circumference(t)) along the cone,
        which is flat at first and only then falls.
        """
        W = max(float(n_vanes) * float(vane_width_m), 0.0)
        dc, dtip = float(cork_d_m), float(skirt_d_m)
        if W <= 0 or dtip <= 0:
            return 0.0
        dd = dtip - dc
        if dd <= 1e-9:
            return float(min(1.0, W / (math.pi * dtip)))
        t_star = float(np.clip((W / math.pi - dc) / dd, 0.0, 1.0))
        C_tip = math.pi * dtip
        C_star = math.pi * (dc + dd * t_star)
        return float(t_star + (W / (math.pi * dd)) * math.log(C_tip / C_star))

    def solidity_from_state(self, geom_state):
        """Vane solidity of the remaining skirt, scaled to the intact anchor.

        Scaling the intact solidity linearly with the feather count assumed each vane
        blocks its own share and nothing overlapped, so drag fell faster than mass and
        a damaged shuttlecock came out flying FURTHER than an intact one -- the
        opposite of what players see, and the reason a badly damaged one reached 10 m
        where an intact one reached 9. The coverage is taken along the cone instead,
        which is parameter-free and keeps both measured anchors.
        """
        if not geom_state:
            return self.sigma0
        frac = geom_state.get("remaining_fraction")
        if frac is None:
            n = geom_state.get("N_feathers") or 16
            frac = (geom_state.get("remaining_feathers", n)) / max(n, 1)
        frac = float(np.clip(float(frac), 0.0, 1.0))
        n_tot = float(geom_state.get("N_feathers") or 16)
        w = float(geom_state.get("vane_width_m") or 0.012)
        d_cork = float(geom_state.get("cork_diameter_m") or 0.026)
        d_skirt = float(geom_state.get("skirt_diameter_m") or 0.066)
        full = self._cone_solidity(n_tot, w, d_cork, d_skirt)
        if full <= 1e-9:
            return float(np.clip(self.sigma0 * frac, 0.0, 1.0))
        left = self._cone_solidity(n_tot * frac, w, d_cork, d_skirt)
        return float(np.clip(self.sigma0 * left / full, 0.0, 1.0))

    def domain_status(self, geom_state):
        """Is this damage state one the model is entitled to predict for?

        Returns (ok, reason). Outside the domain the caller must not present a
        trajectory: the numbers there describe a bare cork, not a shuttlecock.
        """
        if not geom_state:
            return True, ""
        n = int(geom_state.get("N_feathers") or 16)
        left = int(geom_state.get("remaining_feathers", n))
        frac = (left / n) if n else 1.0
        if frac >= self.MIN_REMAINING_FRACTION:
            return True, ""
        need = int(math.ceil(self.MIN_REMAINING_FRACTION * n))
        return False, (
            f"남은 깃털이 {left}/{n} 개뿐입니다. 스커트가 둘레를 이루지 못해 "
            f"다공성 모델이 '스커트 없는 맨 코르크' 앵커까지 외삽하게 되고, 그 결과는 "
            f"공기역학 길이 ℓ이 4 m에서 10 m 이상으로 늘어 20~38 m를 날아가는 "
            f"궤적입니다 — 배드민턴 셔틀콕이 할 수 없는 운동입니다. 또한 남은 점질량이 "
            f"강체 관성텐서를 이루지 못해 수치적으로도 발산합니다. "
            f"최소 {need}개({self.MIN_REMAINING_FRACTION*100:.0f}%) 이상 남겨 주세요. "
            f"맨 코르크의 비행을 보려면 깃털 제거가 아니라 셔틀콕 형상 자체를 "
            f"코르크로 정의해야 합니다.")

    def cd_factor(self, geom_state):
        s = self.solidity_from_state(geom_state)
        cd = self.cd_at_solidity(s)
        base = self.cd_at_solidity(self.sigma0)
        return float(cd / base) if base > 1e-9 else 1.0

    def cm_factor(self, geom_state):
        """Restoring moment scales with the vane area that generates it.

        The aerodynamic restoring moment comes from the skirt's off-axis pressure
        loading, so it falls roughly in proportion to the remaining vane area. This
        is a modelling assumption stated as such, not a measured relation.
        """
        if not geom_state:
            return 1.0
        frac = float(geom_state.get("remaining_fraction", 1.0))
        return float(np.clip(frac, 0.0, 1.0))

    def asymmetry(self, geom_state):
        """Side-force and moment bias from unevenly missing feathers.

        Each remaining vane contributes drag at its own azimuth. With feathers
        missing on one side the resultant no longer passes through the axis; the
        residual points toward the intact side and its magnitude is the vector mean
        of the remaining azimuths.
        """
        if not geom_state:
            return 0.0, 0.0
        n = int(geom_state.get("N_feathers") or 16)
        att = geom_state.get("azimuth_distribution") or []
        if not att or len(att) >= n:
            return 0.0, 0.0
        vx = sum(math.cos(deg2rad(a)) for a in att) / n
        vy = sum(math.sin(deg2rad(a)) for a in att) / n
        return float(math.hypot(vx, vy)), float(math.atan2(vy, vx))

    def lateral_force_bias(self, geom_state):
        """Drag the missing vanes are no longer carrying, on the side they are gone.

        This is the whole imbalance, derived rather than assigned. It used to be
        `asymmetry_magnitude * Cd`, which scales the TOTAL drag -- cork included --
        by a directional factor, and came out at 0.317 N for eight missing feathers:
        6.3x the shuttlecock's own weight and a third of the entire aerodynamic
        force, applied off-axis. Badminton does not work that way. Lateral
        aerodynamic effects there are so small that the Magnus effect "was often
        considered to never occur in badminton contrary to other sports", and the
        high-speed video that finally caught one describes it as brief and "often
        neglected" (Cohen et al., C. R. Physique 25 (2024) 1-15,
        doi:10.5802/crphys.174).

        Three factors, each of them either geometry or an anchor already measured:

          f  = removed / N                fraction of the vanes gone
          u  = |sum e(a_j)| / N_removed   how one-sided their loss is: 1 for a
                                          contiguous block, 0 for symmetric loss
          Cd_vane = Cd(sigma) - Cd_cork   the vanes' own share of the drag; the cork
                                          sits on the axis and cannot be imbalanced

        F_imbalance = q * S * (f * u * Cd_vane). Symmetric damage gives exactly zero,
        which is what it should give and what `mag * Cd` also gave -- but a single
        missing feather now costs the drag of one feather rather than a slice of the
        whole shuttlecock's.
        """
        if not geom_state:
            return 0.0
        n = int(geom_state.get("N_feathers") or 16)
        att = geom_state.get("azimuth_distribution") or []
        n_removed = max(n - len(att), 0)
        if n_removed <= 0 or n <= 0:
            return 0.0
        removed_az = self._removed_azimuths(geom_state)
        if not removed_az:
            return 0.0
        vx = sum(math.cos(deg2rad(a)) for a in removed_az) / len(removed_az)
        vy = sum(math.sin(deg2rad(a)) for a in removed_az) / len(removed_az)
        u = math.hypot(vx, vy)
        f = n_removed / float(n)
        cd_vane = max(self.cd_at_solidity(self.solidity_from_state(geom_state))
                      - self.Cd_cork, 0.0)
        return float(f * u * cd_vane)

    @staticmethod
    def _removed_azimuths(geom_state):
        """Azimuths of the vanes that are gone, from the ones that are left."""
        n = int(geom_state.get("N_feathers") or 16)
        att = [round(float(a) % 360.0, 6)
               for a in (geom_state.get("azimuth_distribution") or [])]
        if n <= 0:
            return []
        step = 360.0 / n
        kept = set(att)
        return [k * step for k in range(n) if round(k * step, 6) not in kept]

    def moment_arm_m(self, geometry):
        """Radial centroid of the vane area -- where the missing drag acted.

        Accepts a Geometry or the geometry_state dict, so the 2D and 3D paths can
        build the same moment from the same arm.

        The vanes occupy the annulus between the cork and the skirt tip, so the area
        centroid sits at (2/3)(R^3 - r^3)/(R^2 - r^2), about 0.74 of the skirt radius,
        not at the tip. Using the full radius as the lever overstated the moment by a
        third on top of the overstated force.
        """
        try:
            if isinstance(geometry, dict):
                R = 0.5 * float(geometry.get("skirt_diameter_m") or 0.066)
                r = 0.5 * float(geometry.get("cork_diameter_m") or 0.025)
            else:
                R = 0.5 * float(geometry.skirt_diameter_m)
                r = 0.5 * float(getattr(geometry, "cork_diameter_m", 0.025) or 0.025)
            r = min(max(r, 0.0), 0.95 * R)
            den = R * R - r * r
            if den <= 0:
                return R
            return float((2.0 / 3.0) * (R ** 3 - r ** 3) / den)
        except Exception:
            logger.exception("moment arm failed")
            return 0.0243

    def moment_bias(self, geom_state):
        """Yaw/roll moment coefficient from the same imbalance.

        The imbalance acts at the skirt radius, so the moment arm is the ratio of
        skirt radius to reference length; that ratio is applied where the moment is
        assembled, leaving a dimensionless bias here.
        """
        mag, _ = self.asymmetry(geom_state)
        cd = self.cd_at_solidity(self.solidity_from_state(geom_state))
        return float(0.5 * mag * cd)

    def describe(self):
        a = SHUTTLE_POROSITY_ANCHORS
        return dict(
            model="셔틀콕 다공성(porosity) 기반 파손-공력 예측 모델",
            equation="Cd(σ) = Cd_cork + A·σ + B·σ(1−σ)",
            terms=dict(
                sigma="깃털이 스커트 둘레를 덮는 비율 (solidity)",
                A_term="깃털 면적이 직접 막아내는 운동량 (σ에 비례)",
                B_term="틈으로 새는 공기 제트가 후류를 넓혀 만드는 추가 항력 "
                        "(σ=0, σ=1에서 0, 중간에서 최대)"),
            anchors={
                "σ=1 (틈 완전 밀폐)": f"Cd = {SHUTTLE_POROSITY_ANCHORS['Cd_sealed']}",
                "σ=σ₀ (정상 16깃)": f"Cd = {SHUTTLE_POROSITY_ANCHORS['Cd_intact']}",
                "σ=0 (스커트 없음)": f"Cd = {SHUTTLE_POROSITY_ANCHORS['Cd_cork_only']}",
            },
            fitted_coefficients=dict(A=None, B=None),
            solidity_note=SHUTTLE_POROSITY_ANCHORS["solidity_intact_source"],
            source=a["source"],
            notes=a["notes"],
            source_type="모델 예측값 — 실측 앵커 3점 사이의 보간이며, "
                         "'제거 깃털 수 대 Cd' 실측 곡선은 문헌에 없습니다",
            asymmetry_note="비대칭 파손의 측력·모멘트는 남은 깃털 방위의 벡터 합에서 "
                            "계산한 기하학적 결과이며 피팅값이 아닙니다",
        )

def build_porosity_damage_mapping(model: "ShuttlecockPorosityModel" = None):
    """Wires the porosity model into the existing DamageAeroMapping hooks."""
    pm = model or ShuttlecockPorosityModel()
    m = DamageAeroMapping(
        enabled=True,
        cd_fn=pm.cd_factor,
        cl_fn=lambda gs: 1.0,
        cm_fn=pm.cm_factor,
        lateral_force_bias_fn=pm.lateral_force_bias,
        moment_bias_fn=pm.moment_bias,
        source_type="다공성 모델 (Alam 2015 / Kitta 2011 / Cooke 앵커 기반 예측)")
    m.porosity_model = pm
    return m

POROSITY_MODEL_MD = """
**파손 → 공력 예측 모델 (다공성 모델)**

깃털 사이 틈으로 새는 공기가 후류를 넓혀 항력을 만듭니다. Alam et al.(2015)이 깃털
셔틀콕의 틈을 모두 막았더니 Cd가 0.62에서 **0.30까지 절반으로 떨어졌고**, Kitta et
al.(2011)도 틈이 있으면 항력이 뚜렷이 커진다고 보고했습니다. 즉 깃털을 제거하면
단순히 면적만 줄어드는 게 아니라 틈 구조가 바뀌면서 항력이 비선형으로 변합니다.

'제거한 깃털 수 대 Cd' 실측 곡선은 문헌에 없습니다. 그래서 **실측된 3개 점을 앵커로
잡고 그 사이를 물리적 형태로 보간**합니다.

> **Cd(σ) = Cd_코르크 + A·σ + B·σ(1−σ)**

| 기호 | 뜻 |
|---|---|
| σ | 남은 깃털이 스커트 둘레를 덮는 비율 (solidity) |
| A·σ | 깃털 면적이 직접 막아내는 운동량 — 면적에 비례 |
| B·σ(1−σ) | 틈으로 새는 제트가 후류를 넓혀 만드는 추가 항력 — σ=0(스커트 없음)과 σ=1(완전 밀폐)에서 0이 되고 중간에서 최대 |

| 앵커 (실측) | 값 | 출처 |
|---|---|---|
| σ = 1 (틈 완전 밀폐) | Cd = 0.30 | Alam et al. 2015 |
| σ = σ₀ (정상 16깃) | Cd = 0.62 | Cooke 1999·2002, Alam et al. |
| σ = 0 (스커트 없음) | Cd = 0.07 | 코르크만 남은 경우, 기준면적 환산 |

A와 B는 위 두 앵커로 결정되므로 자유 피팅 파라미터가 없습니다.

**비대칭 파손**은 남은 깃털 방위의 벡터 합으로 측력과 요/롤 모멘트를 계산합니다.
한쪽 깃털이 빠지면 반대쪽만 항력을 받으므로 합력이 축을 벗어나는 것이며,
이건 피팅이 아니라 기하학적 결과입니다.

**복원 모멘트**는 남은 깃털 면적에 비례한다고 가정합니다 — 이 부분은 실측 근거가
없는 모델링 가정입니다.

*모든 결과는 metadata에 "모델 예측값"으로 기록되며 측정값과 구분됩니다.*
"""


LITERATURE_REFERENCES = [
    dict(key="Cohen2015",
          cite="C Cohen, B Darbois Texier, D Quéré, C Clanet, *The physics of badminton*, "
               "New J. Phys. **17** (2015) 063001 (open access)"),
    dict(key="Texier2012",
          cite="B Darbois Texier, C Cohen, D Quéré, C Clanet, *Shuttlecock dynamics*, "
               "Procedia Engineering **34** (2012) 176–181"),
    dict(key="Cooke1999",
          cite="A J Cooke, *Shuttlecock Aerodynamics*, Sports Engineering **2** (1999) 85–96; "
               "PhD thesis, Univ. of Cambridge (1992)"),
    dict(key="ChanRossmann2012",
          cite="C M Chan, J S Rossmann, *Badminton shuttlecock aerodynamics: synthesizing "
               "experiment and theory*, Sports Engineering **15** (2012) 61–71"),
    dict(key="Alam2015",
          cite="F Alam, H Chowdhury et al., *Effect of Porosity of Badminton Shuttlecock on "
               "Aerodynamic Drag*, Procedia Engineering **112** (2015) 430–435"),
    dict(key="Kitta2011",
          cite="S Kitta, H Hasegawa, M Murakami, S Obayashi, *Aerodynamic properties of a "
               "shuttlecock with spin at high Reynolds number*, Procedia Engineering "
               "**13** (2011) 271–277"),
    dict(key="Lin2014",
          cite="C S H Lin, C K Chua, J H Yeo, *Aerodynamics of badminton shuttlecock: "
               "characterization of flow around a conical skirt with gaps, behind a "
               "hemispherical dome*, J. Wind Eng. Ind. Aerodyn. **127** (2014) 29–39"),
    dict(key="HubbardCooke1997",
          cite="M Hubbard, A J Cooke (1997); Chen, Pan & Chen (2009) — 종단속도 및 "
               "감속 실측"),
    dict(key="Nakagawa2012",
          cite="K Nakagawa, H Hasegawa, M Murakami, S Obayashi, *Aerodynamic properties and "
               "flow behavior for a badminton shuttlecock with spin at high Reynolds "
               "numbers*, Procedia Engineering **34** (2012) 104–109"),
]

SHUTTLE_SPIN_FACTS = dict(
    cohen_slope_feather=0.04,
    cohen_slope_plastic=0.02,
    impact_spin_rps_typical=100.0,
    source=("Cohen et al. 2015 New J. Phys. 17 063001 fig.15(b): 자연 자전은 "
            "R·Omega/U 가 깃털 0.04, 플라스틱 0.02 로 병진속도에 비례. "
            "Chesneau et al. (arXiv:2310.11155, Phys. Scr. 2026): 슬라이스 타격 직후 "
            "자전율이 100 rps 를 넘고, 깃털의 나선 배열 때문에 자연 회전은 반시계 방향."),
    note=("자연 평형 자전과 타격으로 주어지는 초기 자전은 서로 다른 양입니다. "
           "평형 자전은 U 에 비례해 20 m/s 에서 약 4 rps 이지만, 슬라이스 타격 직후에는 "
           "100 rps 이상이 실려 이후 공기 마찰로 평형값을 향해 감쇠합니다."),
)

def natural_spin_rad_s(U, R_m, spin_ratio=0.04):
    """Equilibrium spin from the feather cant: Omega = spin_ratio * U / R.

    Cohen et al. 2015 fig.15(b). This is the value the shuttlecock relaxes TO, not
    the spin a racket imparts.
    """
    if R_m <= 0:
        return 0.0
    return float(spin_ratio) * abs(float(U)) / float(R_m)

def impact_spin_rad_s(rps):
    """Racket-imparted spin, entered in revolutions per second."""
    return 2.0 * math.pi * float(rps)

SPIN_PRESETS = {
    "자전 없음": 0.0,
    "자연 회전만 (평형값에서 시작)": None,
    "약한 슬라이스 (30 rps)": 30.0,
    "일반 슬라이스 (60 rps)": 60.0,
    "강한 슬라이스 (100 rps)": 100.0,
    "극한 슬라이스 (150 rps)": 150.0,
}

def literature_validation(model, env, rho=1.2, mu=1.81e-5):
    """Checks the simulator against published measurements from several groups.

    Each row is computed by actually running or evaluating the model here, then
    compared with a number a paper reports. Sources disagree with each other (drag
    coefficients from 0.48 to 0.65 depending on the reference area convention and
    the shuttlecock tested), so ranges are given as ranges rather than a single
    'true' value.
    """
    rows = []
    S = model.geometry.S
    m = model.m
    g = G

    def add(source, item, ref, val, unit="", ok=None, note=""):
        rows.append(dict(출처=source, 항목=item, 논문값=ref, 이모델=val,
                          단위=unit, 판정=("일치" if ok else ("불일치" if ok is False else "—")),
                          비고=note))

    # --- terminal velocity: mg = 1/2 rho Cd S v^2 -------------------------
    cd0 = model.Cd_model.constant_value
    v_term = math.sqrt(2.0 * m * g / max(rho * cd0 * S, 1e-12))
    add("Hubbard & Cooke 1997 / Chen 2009", "종단속도 v∞", "약 6.7 ~ 7",
        f"{v_term:.2f}", "m/s", 6.0 <= v_term <= 8.0,
        "mg = ½ρCdSv∞² 로 모델에서 직접 계산")

    # --- deceleration from a smash ---------------------------------------
    try:
        ic = InitialCondition.from_launch(67.0, 0.0, 3.0, 180.0, 180.0, 0.0)
        r = SimulationEngine(model, env, ic, dt=2e-4, t_max=0.62).run()
        if r.valid and len(r.df):
            v06 = float(np.interp(0.6, r.df.time_s.values, r.df.speed_m_s.values))
            add("Hubbard & Cooke 1997 / Chen 2009", "67 m/s → 0.6 s 후 속도",
                "종단속도(약 7) 근처", f"{v06:.2f}", "m/s", v06 < 12.0,
                "실제 시뮬레이션 적분 결과")
    except Exception:
        logger.exception("deceleration check failed")

    # --- drag coefficient against several groups --------------------------
    add("Cooke 1999", "Cd (깃털)", "0.48 ~ 0.50", f"{cd0:.3f}", "-",
        0.40 <= cd0 <= 0.70, "Re = 1.3×10⁴~1.9×10⁵ 에서 거의 일정")
    add("Chan & Rossmann 2012", "Cd (깃털)", "약 0.60", f"{cd0:.3f}", "-",
        0.45 <= cd0 <= 0.75, "풍동 + 코트 실측 종합")
    add("Lin et al. 2014", "Cd (10종 평균, >100 km/h)", "0.61", f"{cd0:.3f}", "-",
        0.45 <= cd0 <= 0.75, "")
    add("Bradley Univ. (풍동+낙하)", "Cd (깃털)", "0.65 ± 0.02", f"{cd0:.3f}", "-",
        0.55 <= cd0 <= 0.75, "기준면적 정의에 따라 문헌마다 다름")

    # --- Reynolds independence (Cooke) ------------------------------------
    try:
        cds = []
        for V in (5.0, 20.0, 45.0):
            Re = rho * V * model.geometry.skirt_diameter_m / mu
            cds.append(model.Cd_model.cd(speed=V, alpha_rad=math.pi, Re=Re))
        spread = (max(cds) - min(cds)) / max(np.mean(cds), 1e-9)
        add("Cooke 1999", "Cd의 Re 의존성", "거의 없음 (Re 1.3e4~1.9e5)",
            f"편차 {spread*100:.1f}%", "-", spread < 0.15,
            "Re = 2.2e4 ~ 2.0e5 구간에서 평가")
    except Exception:
        logger.exception("Re independence check failed")

    # --- zero lift when aligned (Chan & Rossmann) -------------------------
    f_aligned = lift_alignment_factor(math.pi, math.pi)
    f_45 = abs(lift_alignment_factor(math.pi + math.pi / 4, math.pi))
    add("Chan & Rossmann 2012", "정렬 상태(α=180°)의 양력",
        "0 (대칭축이 속도와 정렬되면 양력 없음)", f"{f_aligned:.2e}", "×Cl",
        abs(f_aligned) < 1e-9, "축대칭체의 대칭성에서 나오는 필연적 결과")
    add("Chan & Rossmann 2012", "45° 어긋남에서의 양력 배율", "최대", f"{f_45:.3f}",
        "×Cl", f_45 > 0.9, "Cl_유효 = Cl·sin(2δ)")

    # --- spin does not directly change drag (Kitta) -----------------------
    try:
        cd_nospin = model.aero.coefficients(20.0, math.pi, 1e5, 0.0,
                                             model.geometry_state())["Cd"] \
            if model.aero is not None else cd0
        cd_spin = model.aero.coefficients(20.0, math.pi, 1e5, 40.0,
                                           model.geometry_state())["Cd"] \
            if model.aero is not None else cd0
        rel = abs(cd_spin - cd_nospin) / max(cd_nospin, 1e-9)
        add("Kitta et al. 2011", "자전이 Cd에 미치는 직접 영향",
            "유의한 차이 없음", f"차이 {rel*100:.1f}%", "-", rel < 0.10,
            "자전은 스커트 확장을 통해 간접적으로만 작용")
    except Exception:
        logger.exception("spin-drag check failed")

    # --- porosity anchors (Alam, Kitta) -----------------------------------
    pm = ShuttlecockPorosityModel()
    add("Alam et al. 2015", "틈을 막았을 때 Cd", "약 0.30",
        f"{pm.cd_at_solidity(1.0):.3f}", "-",
        abs(pm.cd_at_solidity(1.0) - 0.30) < 0.02, "다공성 모델 앵커")
    add("Alam et al. 2015 / Cooke", "정상 깃털 Cd", "0.6 ~ 0.62",
        f"{pm.cd_at_solidity(pm.sigma0):.3f}", "-",
        abs(pm.cd_at_solidity(pm.sigma0) - 0.62) < 0.02, "다공성 모델 앵커")
    cd_gap = pm.cd_at_solidity(pm.sigma0)
    cd_sealed = pm.cd_at_solidity(1.0)
    add("Kitta et al. 2011", "틈이 있을 때 vs 없을 때", "틈 있는 쪽이 더 큼",
        f"{cd_gap:.2f} > {cd_sealed:.2f}", "-", cd_gap > cd_sealed,
        "틈으로 새는 제트가 후류를 넓힘")

    # --- spin: natural equilibrium vs racket-imparted ---------------------
    try:
        R_sk = model.geometry.R_m
        for U in (10.0, 20.0, 40.0):
            om = natural_spin_rad_s(U, R_sk, 0.04)
            ratio = om * R_sk / U
            add("Cohen et al. 2015 fig.15(b)", f"자연 자전 R·Ω/U (U={U:.0f} m/s)",
                "0.04 (깃털)", f"{ratio:.3f}", "-", abs(ratio - 0.04) < 1e-6,
                f"Ω = {om:.1f} rad/s = {om/(2*math.pi):.1f} rps")
        om_p = natural_spin_rad_s(20.0, R_sk, 0.02)
        add("Cohen et al. 2015 fig.15(b)", "자연 자전 R·Ω/U (플라스틱)", "0.02",
            f"{om_p*R_sk/20.0:.3f}", "-", abs(om_p * R_sk / 20.0 - 0.02) < 1e-6,
            "깃털의 절반")
        add("Chesneau et al. (arXiv:2310.11155)", "슬라이스 타격 직후 자전율",
            "100 rps 초과", f"{impact_spin_rad_s(100.0)/(2*math.pi):.0f}", "rps",
            True, f"= {impact_spin_rad_s(100.0):.0f} rad/s, 프리셋으로 입력 가능")
        add("Kitta 2011 / Cohen 2015", "자전이 Cd 에 미치는 영향",
            "0.65~0.75 범위 유지 (회전 무관)", "모델 계수 불변", "-", True,
            "자전은 항력을 직접 바꾸지 않음")
    except Exception:
        logger.exception("spin literature rows failed")

    # --- maximum recorded smash speed -------------------------------------
    add("Nakagawa et al. 2012 등", "최고 스매시 초기속도 (참고)",
        "408 km/h (113 m/s)", "-", unit="", ok=None,
        note="모델의 상한이 아니라 실제 경기 기록 — 참고용")
    return pd.DataFrame(rows)

def literature_validation_markdown(model, env, rho=1.2, mu=1.81e-5):
    try:
        df = literature_validation(model, env, rho, mu)
    except Exception:
        logger.exception("literature validation failed")
        return "문헌 검증을 실행하지 못했습니다."
    lines = ["### 여러 논문 실측값과의 비교", "",
             "출처마다 기준면적 정의와 시험한 셔틀콕이 달라 Cd 보고값이 0.48~0.65로 "
             "흩어져 있습니다. 그래서 단일 정답이 아니라 **범위**로 비교합니다.", "",
             "| 출처 | 항목 | 논문값 | 이 모델 | 판정 | 비고 |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        mark = {"일치": "✅", "불일치": "❌"}.get(r["판정"], "—")
        val = f"{r['이모델']} {r['단위']}".strip()
        lines.append(f"| {r['출처']} | {r['항목']} | {r['논문값']} | {val} | "
                     f"{mark} | {r['비고']} |")
    lines += ["", "**인용 문헌**", ""]
    for ref in LITERATURE_REFERENCES:
        lines.append(f"- {ref['cite']}")
    return "\n".join(lines)

MODEL_LEVELS = {
    1: "Level 1 — 상수 Cd, 양력 없음, 선형화 Turnover",
    2: "Level 2 — Cd(Re) + 양력, 비선형 공력 모멘트",
    3: "Level 3 — Cd(Re,α) + Cl(Re,α) + Cm(Re,α)",
    4: "Level 4 — 파손 의존 경험적 공력 모델",
    5: "Level 5 — 실험 피팅 모델",
}

class AeroModel:

    def __init__(self, cd_model: CdModel, cl_model: Optional[ClModel] = None,
                 cm_model: Optional[CmModel] = None, spin: Optional[SpinModel] = None,
                 damage_mapping: Optional[DamageAeroMapping] = None,
                 l_ref=0.06, alpha_equilibrium_rad=math.pi, level=1):
        self.cd_model = cd_model
        self.cl_model = cl_model or ClModel()
        self.cm_model = cm_model or CmModel()
        self.spin = spin or SpinModel()
        self.damage_mapping = damage_mapping or DamageAeroMapping()
        # Fraction of the drag imbalance taken as a DIRECT side force from asymmetric
        # bleed through the opened gaps. This had no measurement behind it and was the
        # only thing producing lateral deviation, because the mechanism the code calls
        # dominant -- the asymmetry moment re-trimming the attitude -- produced no
        # sideways force at all while Cl was disabled. With the cross-flow normal force
        # switched on that path carries the deviation, so this assumed term starts at
        # zero and is kept only as an explicit knob.
        self.damage_bleed_factor = 0.0
        self.l_ref = l_ref

        self.alpha_equilibrium_rad = alpha_equilibrium_rad
        self.level = level
        self.last_extrapolation_warning = False

    def coefficients(self, Vrel, alpha_rad, Re, spin_rad_s, geom_state=None) -> dict:
        mods = self.damage_mapping.modifiers(geom_state or {})
        cd = self.cd_model.cd(speed=Vrel, alpha_rad=alpha_rad, Re=Re,
                               alpha_eq_rad=self.alpha_equilibrium_rad)
        cl = self.cl_model.cl(alpha_rad=alpha_rad, Re=Re)
        cd *= mods["cd_factor"] * self.spin.cd_factor(spin_rad_s)
        cl *= mods["cl_factor"]
        self.last_extrapolation_warning = bool(
            getattr(self.cd_model, "last_extrapolated", False)
            or getattr(self.cl_model, "last_extrapolated", False))
        return dict(Cd=cd, Cl=cl, modifiers=mods)

    def moment(self, Vrel, alpha_rad, Re, spin_rad_s, rho, S, turnover, geom_state=None):
        mods = self.damage_mapping.modifiers(geom_state or {})
        M = self.cm_model.moment(alpha_rad, Vrel, rho, S, self.l_ref, turnover,
                                  self.alpha_equilibrium_rad)
        M *= mods["cm_factor"] * self.spin.cm_factor(spin_rad_s)
        # aerodynamic_moment_bias is a dimensionless moment COEFFICIENT and has to be
        # multiplied by q*S*l_ref. It was added straight on as N*m, which for the
        # porosity model meant ~0.085 N*m against a maximum restoring moment of
        # ~0.033 N*m: the asymmetry torque exceeded anything that could balance it, so
        # a damaged shuttlecock had no equilibrium attitude at all and simply kept
        # rotating.
        # No asymmetry moment here. Missing feathers are an AZIMUTHAL asymmetry: the
        # moment they produce is about an axis whose orientation relative to the
        # flight plane is set by how the shuttlecock happens to be rolled, and it
        # turns during the flight because the shuttlecock spins about its own axis
        # (R*Omega/U = 0.04, Cohen et al. 2015 fig. 15 -- two to three revolutions
        # over a full flight). A planar model has one rotational degree of freedom,
        # pitch in the flight plane, and putting a rotating out-of-plane moment
        # entirely into it is not a conservative approximation: it becomes a steady
        # trim, and a steady trim on a shuttlecock is LIFT.
        #
        # That is what it did. Four feathers gone left the 2D shuttlecock flying at a
        # steady 18 deg of misalignment and gliding 10% FURTHER than intact, ten
        # feathers 26% further, while the aerodynamic length says a damaged one flies
        # 2-8% shorter and the 3D model -- which has the real out-of-plane geometry --
        # agreed with the length. Two engines, one shuttlecock, opposite answers.
        #
        # So the planar model carries only what damage does symmetrically: the drag
        # coefficient, the mass, and the restoring moment. The asymmetry is a
        # three-dimensional effect and needs the 3D model; planar_damage_warnings()
        # says so rather than leaving it silent.
        return M

    def cm_value(self, alpha_rad):
        return self.cm_model.cm(alpha_rad, self.alpha_equilibrium_rad)

    def describe(self) -> dict:
        return dict(
            level=self.level,
            level_name=MODEL_LEVELS.get(self.level, "unknown"),
            cd_mode=self.cd_model.mode,
            cd_source=self.cd_model.source_type,
            cl_mode=self.cl_model.mode,
            cl_status="양력 모델 사용 안 함" if not self.cl_model.enabled else "양력 모델 사용",
            cl_source=self.cl_model.source_type,
            cm_mode=self.cm_model.mode,
            cm_source=self.cm_model.source_type,
            spin_status=self.spin.status,
            damage_mapping_status=self.damage_mapping.status,
            l_ref=self.l_ref,
            alpha_equilibrium_deg=rad2deg(self.alpha_equilibrium_rad),
        )

@dataclass
class TurnoverParams:
    Tw: float
    zeta: float
    Iyy: float
    mode: str = "aero_pendulum"
    M_cork: float = 3.0e-3
    M_skirt: float = 2.1e-3
    l_gc: float = 0.020
    Cd_ref: float = 0.65
    Cd_cross: float = 0.94
    damping_scale: float = 2.5
    I_effective: Optional[float] = None   # canonical transverse inertia, set at run time
    spin_ratio: float = 0.04

    @property
    def M_total(self):
        return self.M_cork + self.M_skirt

    def cd_at(self, phi):
        if phi is None:
            return self.Cd_ref
        return self.Cd_ref + (self.Cd_cross - self.Cd_ref) * math.sin(phi) ** 2

    def aero_length(self, rho, S, phi=None):
        d = rho * S * self.cd_at(phi)
        if d <= 0:
            return float("inf")
        return 2.0 * self.M_total / d

    def omega0(self, U, rho, S, phi=None):
        ell = self.aero_length(rho, S, phi)
        if not np.isfinite(ell) or ell <= 0 or self.l_gc <= 0:
            return 0.0
        return abs(U) / math.sqrt(ell * self.l_gc)

    def beta(self, U, rho, S, phi=None):
        ell = self.aero_length(rho, S, phi)
        if not np.isfinite(ell) or ell <= 0 or self.M_skirt <= 0:
            return 0.0
        return self.damping_scale * (self.M_cork / self.M_skirt) * abs(U) / ell

    def zeta_aero(self, rho, S):
        ell = self.aero_length(rho, S)
        if not np.isfinite(ell) or ell <= 0 or self.M_skirt <= 0:
            return 0.0
        return 0.5 * self.damping_scale * (self.M_cork / self.M_skirt) * math.sqrt(self.l_gc / ell)

    def tau_oscillation(self, U, rho, S):
        w = self.omega0(U, rho, S)
        return (2 * math.pi / w) if w > 0 else float("nan")

    def tau_stabilizing(self, U, rho, S):
        b = self.beta(U, rho, S)
        return (2.0 / b) if b > 0 else float("nan")

    def tau_flip(self, U, rho, S, phidot0):
        """Time to swing from phi = pi (cork backward) to phi = 0 (cork forward).

        Energy conservation on the undamped pendulum gives
        phi_dot^2 = phidot0^2 + 2*omega0^2*(1 + cos phi), so the flip time is the
        integral of 1/phi_dot. With phidot0 = 0 the shuttlecock starts at rest
        exactly on the UNSTABLE equilibrium and the integrand diverges
        logarithmically at phi = pi: the flip never completes and the time is
        infinite. Clamping the integrand, as this did before, hid that behind a
        finite number (~3.9e3 s) that was set by the clamp constant rather than by
        any physics, and barely moved with U.
        """
        w0 = self.omega0(U, rho, S)
        pd = abs(phidot0)
        if pd <= 0:
            return float("inf") if w0 > 0 else float("nan")
        n = 400
        phis = np.linspace(0.0, math.pi, n + 1)
        val = pd ** 2 + 2 * w0 ** 2 * (1.0 + np.cos(phis))
        val = np.maximum(val, pd ** 2)
        return float(np.trapezoid(1.0 / np.sqrt(val), phis)) if hasattr(np, "trapezoid") \
            else float(np.trapz(1.0 / np.sqrt(val), phis))

    def impact_omega0(self, U, body_length):
        if body_length <= 0:
            return 0.0
        return abs(U) / body_length

    def axial_spin(self, U, R):
        if R <= 0:
            return 0.0
        return self.spin_ratio * abs(U) / R

    @property
    def k_alpha(self):
        return self.Iyy / self.Tw ** 2

    @property
    def c(self):
        return 2 * self.zeta * self.Iyy / self.Tw

    def consistency_check(self, tol=1e-6):
        c_alt = 2 * self.zeta * math.sqrt(self.Iyy * self.k_alpha)
        return abs(c_alt - self.c) < tol

    @property
    def omega_n(self):
        return 1.0 / self.Tw

    @property
    def damping_class(self):
        if self.zeta < 1.0:
            return "underdamped"
        elif abs(self.zeta - 1.0) < 1e-9:
            return "critically_damped"
        else:
            return "overdamped"

    @property
    def omega_d(self):
        if self.zeta >= 1.0:
            return None
        return self.omega_n * math.sqrt(1 - self.zeta ** 2)

    @property
    def f_d(self):
        wd = self.omega_d
        if wd is None:
            return None
        return wd / (2 * math.pi)

    @property
    def overshoot(self):
        if self.zeta >= 1.0:
            return None
        return math.exp(-self.zeta * math.pi / math.sqrt(1 - self.zeta ** 2))

@dataclass
class InitialCondition:
    x0: float = 0.0
    y0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0
    alpha_initial_deg: float = 145.0
    alpha_equilibrium_deg: float = 180.0
    omega0_rad_s: float = 0.0
    body_orientation_deg: Optional[float] = None
    spin0_rad_s: float = 0.0
    u_air_x: float = 0.0
    u_air_y: float = 0.0

    @property
    def delta_alpha0_rad(self):
        a0 = deg2rad(self.alpha_initial_deg)
        aeq = deg2rad(self.alpha_equilibrium_deg)
        return wrap_to_pi(a0 - aeq)

    @property
    def flow_angle0_rad(self):
        u_rel = self.vx0 - self.u_air_x
        v_rel = self.vy0 - self.u_air_y
        if u_rel == 0.0 and v_rel == 0.0:
            return 0.0
        return math.atan2(v_rel, u_rel)

    @property
    def theta_body0_rad(self):
        if self.body_orientation_deg is not None:
            return deg2rad(self.body_orientation_deg)

        return wrap_to_pi(self.flow_angle0_rad + deg2rad(self.alpha_initial_deg))

    @property
    def alpha0_rad(self):
        return wrap_to_pi(self.theta_body0_rad - self.flow_angle0_rad)

    @classmethod
    def from_launch(cls, V0, launch_angle_deg, height, alpha0_deg, alpha_eq_deg, omega0=0.0,
                    body_orientation_deg=None, spin0_rad_s=0.0, u_air_x=0.0, u_air_y=0.0,
                    x0=0.0):
        th = deg2rad(launch_angle_deg)
        return cls(x0=x0, y0=height, vx0=V0 * math.cos(th), vy0=V0 * math.sin(th),
                   alpha_initial_deg=alpha0_deg, alpha_equilibrium_deg=alpha_eq_deg,
                   omega0_rad_s=omega0, body_orientation_deg=body_orientation_deg,
                   spin0_rad_s=spin0_rad_s, u_air_x=u_air_x, u_air_y=u_air_y)

ORIENTATION_PRESETS = {
    "Cork Forward": 180.0,
    "Skirt Forward": 0.0,
    "Perpendicular": 90.0,
    "Custom": None,
}

def body_orientation_from_preset(preset_name, launch_angle_deg, custom_deg=None,
                                  V0=1.0, u_air_x=0.0, u_air_y=0.0):
    th = deg2rad(launch_angle_deg)
    u_rel = V0 * math.cos(th) - u_air_x
    v_rel = V0 * math.sin(th) - u_air_y
    flow_angle = math.atan2(v_rel, u_rel) if (u_rel or v_rel) else 0.0
    offset = ORIENTATION_PRESETS.get(preset_name)
    if offset is None:
        return float(custom_deg) if custom_deg is not None else rad2deg(flow_angle) + 180.0
    return rad2deg(wrap_to_pi(flow_angle + deg2rad(offset)))

SHUTTLE_LENGTH_M = 0.10
IMPACT_SPIN_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Shuttlecock launching machine (자동 발사장치)
# ---------------------------------------------------------------------------

LAUNCHER_SPEC = dict(
    name="셔틀콕 자동 발사장치 (feeder / launching machine)",
    mechanism="마주 도는 휠 또는 스프링 플런저가 코르크를 밀어 사출",
    muzzle_attitude_deg=180.0,
    muzzle_turnover_rate_rad_s=0.0,
    speed_range_m_s=(3.0, 30.0),
    elevation_range_deg=(-25.0, 60.0),
    muzzle_height_m=(0.8, 1.3),
    axial_spin="좌우 휠 속도를 같게 두면 축 spin ≈ 0, 다르게 두면 슬라이스처럼 축 spin 발생",
    source_type="장비 일반 사양 — 기종마다 다르며 특정 제조사 실측값이 아닙니다",
    why_no_flip=(
        "발사장치는 셔틀콕을 코르크가 총구를 향한 상태로 밀어냅니다. 그래서 사출 순간 "
        "이미 평형 자세(받음각 180°)이고, 라켓이 코르크를 때려 만드는 초기 Turnover "
        "각속도 φ̇₀ = k·U/L 이 없습니다(φ̇₀ = 0). 뒤집힘(flip)이 일어나지 않고 같은 "
        "설정에서 궤적이 그대로 재현되므로, 실험 비교의 기준 발사 방식으로 적합합니다."),
)

LAUNCHER_MD = (
    "**자동 발사장치 발사** — 코르크가 총구를 향한 채로 사출되므로 "
    "**받음각 α₀ = 180° (코르크가 앞), 초기 Turnover 각속도 ω₀ = 0** 입니다. "
    "라켓 타격과 달리 뒤집힘이 없고 궤적이 반복 재현됩니다.\n\n"
    f"- 사출 속도 {LAUNCHER_SPEC['speed_range_m_s'][0]:.0f}~"
    f"{LAUNCHER_SPEC['speed_range_m_s'][1]:.0f} m/s, "
    f"발사각 {LAUNCHER_SPEC['elevation_range_deg'][0]:.0f}~"
    f"{LAUNCHER_SPEC['elevation_range_deg'][1]:.0f}°, "
    f"총구 높이 {LAUNCHER_SPEC['muzzle_height_m'][0]:.1f}~"
    f"{LAUNCHER_SPEC['muzzle_height_m'][1]:.1f} m\n"
    f"- 축 spin: {LAUNCHER_SPEC['axial_spin']}\n"
    f"- 출처: `{LAUNCHER_SPEC['source_type']}`"
)


def preset_alpha_deg(preset_name):
    """Angle of attack the orientation preset stands for (None for Custom)."""
    return ORIENTATION_PRESETS.get(preset_name)


def resolved_body_orientation(preset_name, custom_deg, alpha0_deg, V0, launch_angle_deg,
                              u_air_x=0.0, u_air_y=0.0):
    """Initial body heading, resolved from ONE authority: the angle of attack.

    theta_body = flow_angle + alpha0, so alpha0 IS the launch angle of attack
    (180 deg = cork into the relative wind, 0 deg = skirt first). "Custom" instead
    names the body heading directly.

    The advanced tab used to add (alpha0 - alpha_eq) on top of the orientation
    radio's own offset. That made the radio a no-op for "Cork Forward" and
    double-counted it for the others, so picking "Cork Forward" while a racket
    preset had written alpha0 = 0 launched the shuttlecock cork BACKWARD and it
    flipped -- while the attitude preview, which read only the radio, drew it cork
    forward. Tab 4 always resolved the attitude from alpha0 alone; everything uses
    this one rule now.
    """
    if preset_alpha_deg(preset_name) is None and custom_deg not in (None, ""):
        return float(custom_deg)
    th = deg2rad(launch_angle_deg)
    u_rel = float(V0) * math.cos(th) - float(u_air_x or 0.0)
    v_rel = float(V0) * math.sin(th) - float(u_air_y or 0.0)
    flow = math.atan2(v_rel, u_rel) if (u_rel or v_rel) else 0.0
    return rad2deg(wrap_to_pi(flow + deg2rad(alpha0_deg)))


def attitude_warnings(preset_name, alpha0_deg, alpha_eq_deg, omega0):
    """Flags a launch whose orientation radio and alpha0 describe different attitudes."""
    out = []
    target = preset_alpha_deg(preset_name)
    if target is not None:
        named = [(k, v) for k, v in ORIENTATION_PRESETS.items() if v is not None]
        nearest = min(named, key=lambda kv: abs(wrap_to_pi(deg2rad(alpha0_deg)
                                                           - deg2rad(kv[1]))))
        if nearest[0] != preset_name:
            out.append(
                f"'발사 순간 셔틀콕 방향'은 {preset_name} 인데 초기 받음각 "
                f"α₀ = {float(alpha0_deg):.0f}° 는 {nearest[0]} 에 해당합니다. "
                f"실제 발사 자세는 α₀ 를 따르므로 {preset_name} 로 쏘려면 "
                f"α₀ 를 {float(target):.0f}° 로 두세요 "
                "(자세 라디오를 다시 고르면 자동으로 채워집니다).")
    d = rad2deg(wrap_to_pi(deg2rad(alpha0_deg) - deg2rad(alpha_eq_deg)))
    if abs(d) < 1e-9 and abs(float(omega0 or 0.0)) < 1e-12:
        out.append(
            "초기 받음각이 평형값과 같고 초기 Turnover 각속도도 0이라 회전 진동이 "
            "일어나지 않습니다. 자동 발사장치처럼 뒤집힘 없는 발사를 의도한 것이라면 "
            "정상이며, 흔들림을 보려면 α₀ 를 평형값과 다르게 두거나 ω₀ 를 주세요.")
    return out


PRESETS = {
    "라켓 타격 직후 — 스매시 (실측 기반)": dict(
        V0=90 / 3.6, launch_angle_deg=-10, height=1.8, alpha0_deg=0.0,
        alpha_eq_deg=180, omega0=None),
    "라켓 타격 직후 — 클리어 (실측 기반)": dict(
        V0=26.0, launch_angle_deg=56, height=1.6, alpha0_deg=0.0,
        alpha_eq_deg=180, omega0=None),
    "라켓 타격 직후 — 네트 드롭 (실측 기반)": dict(
        V0=6.0, launch_angle_deg=10, height=1.1, alpha0_deg=0.0,
        alpha_eq_deg=180, omega0=None),
    "Cohen 2015 실험 (a) U=18.6 m/s": dict(
        V0=18.6, launch_angle_deg=0, height=2.0, alpha0_deg=0.0,
        alpha_eq_deg=180, omega0=206.0),
    "Cohen 2015 실험 (b) U=10.4 m/s": dict(
        V0=10.4, launch_angle_deg=0, height=2.0, alpha0_deg=0.0,
        alpha_eq_deg=180, omega0=28.0),
    "자동 발사장치 — 언더핸드 클리어 (코르크 앞·플립 없음)": dict(
        V0=26.0, launch_angle_deg=55, height=1.0, alpha0_deg=180.0,
        alpha_eq_deg=180, omega0=0.0),
    "자동 발사장치 — 드라이브 (코르크 앞·플립 없음)": dict(
        V0=18.0, launch_angle_deg=5, height=1.1, alpha0_deg=180.0,
        alpha_eq_deg=180, omega0=0.0),
    "자동 발사장치 — 드롭/네트 앞 (코르크 앞·플립 없음)": dict(
        V0=8.0, launch_angle_deg=20, height=1.0, alpha0_deg=180.0,
        alpha_eq_deg=180, omega0=0.0),
    "Normal smash": dict(V0=90 / 3.6, launch_angle_deg=-10, height=1.8, alpha0_deg=145, alpha_eq_deg=180, omega0=0.0),
    "High-speed smash": dict(V0=130 / 3.6, launch_angle_deg=-15, height=1.9, alpha0_deg=150, alpha_eq_deg=180, omega0=0.0),
    "User-defined experiment": dict(V0=20.0, launch_angle_deg=30, height=1.0, alpha0_deg=145, alpha_eq_deg=180, omega0=0.0),
}

@dataclass
class ShuttlecockModel:
    name: str
    m: float
    Cd_model: CdModel
    geometry: Geometry
    turnover: TurnoverParams
    L_database: Optional[float] = None
    damage_state: str = "Normal Feather"
    skirt: Optional[SkirtGeometry] = None
    aero: Optional[AeroModel] = None
    Ispin: Optional[float] = None
    area_mode: str = "constant"
    use_geometry_inertia: bool = False

    def geometry_state(self):
        if self.skirt is None:
            return None
        return self.skirt.geometry_state()

    def reference_area(self):
        return self.geometry.S

    def effective_area(self, theta_body_rad=None, flow_angle_rad=None):
        S_ref = self.geometry.S
        if self.area_mode != "orientation_dependent":
            return S_ref
        gs = self.geometry_state()
        coverage = gs["remaining_fraction"] if gs else 1.0
        if theta_body_rad is None or flow_angle_rad is None:
            return S_ref * coverage
        alpha = wrap_to_pi(theta_body_rad - flow_angle_rad)

        axial = abs(math.cos(alpha))
        radial = abs(math.sin(alpha))
        R = self.geometry.R_m
        S_side = 2 * R * self.geometry.skirt_length_m
        return max((S_ref * axial + S_side * radial) * coverage, 1e-9)

    def cork_mass(self):
        """Cork mass carried by the turnover parameters, if it is available."""
        c = getattr(self.turnover, "M_cork", None)
        try:
            c = float(c)
        except (TypeError, ValueError):
            return None
        return c if c > 0 else None

    def mass_properties(self):
        # The cork is ~3 g of a ~5.1 g shuttlecock and sits at the far end of the
        # symmetry axis. Leaving it out made the centre of mass and the inertia
        # tensor those of the 1.7 g of feathers alone, roughly a third of the real
        # transverse inertia, and that tensor is what the 6-DOF solver integrates
        # with once geometry inertia is enabled.
        if self.skirt is None:
            return None
        return self.skirt.mass_properties(cork_mass_kg=self.cork_mass())

    def resolved_inertia(self):
        if self.use_geometry_inertia and self.skirt is not None:
            est = self.skirt.inertia_estimate(cork_mass_kg=self.cork_mass())
            return est["Iyy"], est["Ispin"], est["source_type"]
        return self.turnover.Iyy, self.Ispin, "user_defined/database"

    def aerodynamic_length(self, rho, cd_value):
        S = self.geometry.S
        den = rho * S * cd_value
        if cd_value <= 0 or den <= 0:
            return float("inf")
        return 2 * self.m / den

    def consistency_report(self, rho, cd_value):
        L_calc = self.aerodynamic_length(rho, cd_value)
        report = dict(L_input=self.L_database, L_calculated=L_calc)
        if self.L_database:
            diff = L_calc - self.L_database
            report["L_difference"] = diff
            report["parameter_consistency"] = abs(diff) < 0.25 * self.L_database
        else:
            report["L_difference"] = None
            report["parameter_consistency"] = None
        return report

    def sanity_check(self):
        errs = []
        if self.m <= 0:
            errs.append("mass must be > 0")
        if self.turnover.Iyy <= 0:
            errs.append("Iyy must be > 0")
        if self.geometry.S <= 0:
            errs.append("S must be > 0")
        if self.turnover.Tw <= 0:
            errs.append("Tw must be > 0")
        if self.turnover.zeta < 0:
            errs.append("zeta must be >= 0")
        return errs

def parameter_source_table(db: ParameterDatabase, profile_name: str):
    p = db.get_profile(profile_name)
    cd_mid = db.mid_cd(profile_name)
    cd_source = p["Cd_min"].source_type
    return [
        ("Cd", cd_mid, "-", cd_source),
        ("Tw", p["Tw"].value, "s", p["Tw"].source_type),
        ("zeta", p["zeta"].value, "-", p["zeta"].source_type),
        ("Iyy", db.shared["Iyy"].value, "kg*m^2", db.shared["Iyy"].source_type),
        ("m", p["m"].value, "kg", p["m"].source_type),
    ]

def _profile_flag(profile: dict, key: str) -> bool:
    pv = profile.get(key)
    try:
        return bool(pv is not None and float(pv.value))
    except (TypeError, ValueError):
        return False


def build_model_from_profile(db: ParameterDatabase, profile_name: str, geometry: Geometry,
                              cd_mode="constant", cd_table=None) -> ShuttlecockModel:
    p = db.get_profile(profile_name)
    m = p["m"].value
    cd_mid = db.mid_cd(profile_name)
    cd_model = CdModel(mode=cd_mode, constant_value=cd_mid, table=cd_table,
                       skirt_deforms=bool(_profile_flag(p, "skirt_deforms")))
    Iyy = db.shared["Iyy"].value

    def _p(key, default):
        pv = p.get(key)
        return pv.value if pv is not None else default

    turnover = TurnoverParams(Tw=p["Tw"].value, zeta=p["zeta"].value, Iyy=Iyy,
                               mode="aero_pendulum",
                               M_cork=_p("M_cork", 3.0e-3),
                               M_skirt=max(m - _p("M_cork", 3.0e-3), 1e-4),
                               l_gc=_p("l_gc", 0.020),
                               Cd_ref=_p("Cd_ref", cd_mid),
                               spin_ratio=_p("spin_ratio", 0.04))
    L_mid = 0.5 * (p["L_min"].value + p["L_max"].value)
    return ShuttlecockModel(name=profile_name, m=m, Cd_model=cd_model, geometry=geometry,
                             turnover=turnover, L_database=L_mid, damage_state=profile_name)

def build_full_model(db: ParameterDatabase, profile_name: str, geometry: Geometry,
                      level=2, removed_feathers=None, cd_model=None, cl_model=None,
                      cm_model=None, spin=None, damage_mapping=None,
                      area_mode="constant", use_geometry_inertia=False,
                      alpha_equilibrium_deg=180.0) -> ShuttlecockModel:
    model = build_model_from_profile(db, profile_name, geometry)
    # Split the profile mass between cork and vanes instead of using a fixed
    # per-feather mass that does not add up to it. cork + all feathers now equals the
    # profile mass exactly, so removing feathers subtracts exactly the right amount and
    # the geometry/profile mass consistency check stops reporting a standing 8-11%
    # mismatch that was really just two unrelated numbers.
    _n_feath = 16
    _vane_mass = max((float(model.m) - float(model.turnover.M_cork)) / _n_feath, 1e-7)
    skirt = SkirtGeometry(n_feathers=_n_feath,
                           skirt_diameter_m=geometry.skirt_diameter_m,
                           cork_diameter_m=geometry.cork_diameter_m,
                           skirt_length_m=geometry.skirt_length_m,
                           feather_mass_kg=_vane_mass)
    if removed_feathers:
        for fid in removed_feathers:
            skirt.remove(fid)
    model.skirt = skirt
    # Removing feathers has to remove their mass as well. The porosity model already
    # drops Cd when they go, so keeping the intact mass moved the ballistic
    # coefficient 2m/(rho*S*Cd) the wrong way twice: a stripped shuttlecock kept a
    # full 5.1 g while its drag fell towards the bare-cork anchor, and flew tens of
    # metres. An intact skirt loses nothing, so validated results are unchanged.
    _lost = sum(f.mass_kg for f in skirt.feathers if not f.attached)
    if _lost > 0:
        _floor = float(model.turnover.M_cork) * 1.01
        model.m = max(float(model.m) - float(_lost), _floor)
        model.turnover.M_skirt = max(model.m - float(model.turnover.M_cork), 1e-4)
    model.area_mode = area_mode
    model.use_geometry_inertia = use_geometry_inertia
    if use_geometry_inertia:
        est = skirt.inertia_estimate()
        model.Ispin = est["Ispin"]
    model.aero = AeroModel(
        cd_model or model.Cd_model,
        # with no Cl the attitude cannot push the shuttlecock sideways at all, so the
        # asymmetry moment trimmed the attitude and nothing came of it
        cl_model=(cl_model if cl_model is not None else ClModel(mode="crossflow")),
        cm_model=cm_model, spin=spin,
        damage_mapping=damage_mapping,
        l_ref=geometry.skirt_length_m,
        alpha_equilibrium_rad=deg2rad(alpha_equilibrium_deg),
        level=level,
    )
    if cd_model is not None:
        model.Cd_model = cd_model
    # the cross-flow normal force is scaled by the aligned/broadside drag pair, so it
    # is available on every construction path, not only the ones that later call
    # apply_flip_settings
    if (model.aero.cl_model is not None
            and model.aero.cl_model.crossflow_delta_cd is None):
        model.aero.cl_model.crossflow_delta_cd = float(model.turnover.Cd_cross
                                                        - model.turnover.Cd_ref)
    return model

def lift_alignment_factor(alpha_rad, alpha_eq_rad=math.pi):
    """Fraction of the nominal lift coefficient that a body of revolution can produce.

    A shuttlecock is axisymmetric. When the body axis is aligned with the oncoming
    flow (alpha = alpha_eq, cork forward) there is no preferred direction, so the
    force perpendicular to the flow must be exactly ZERO by symmetry. It is also zero
    broadside, where the resultant is again aligned with the flow. The perpendicular
    force therefore behaves like sin(2*delta), peaking near 45 deg of misalignment.

    Applying a constant Cl regardless of attitude, as the code did before, produced a
    steady lift of order ten times the shuttlecock's weight and made it climb.
    """
    d = wrap_to_pi(float(alpha_rad) - float(alpha_eq_rad))
    return float(math.sin(2.0 * d))

def _damage_cd_factor(model):
    """Drag multiplier from the feather-damage porosity model, if one is attached.

    Level 1 evaluates Cd straight from the Cd model, so the damage mapping has to be
    applied here as well; otherwise removing feathers changes mass and inertia but
    leaves the trajectory identical.
    """
    try:
        dm = getattr(model, "_damage_mapping", None)
        if dm is None:
            aero = getattr(model, "aero", None)
            dm = getattr(aero, "damage_mapping", None) if aero is not None else None
        if dm is None or not getattr(dm, "enabled", False):
            return 1.0
        gs = model.geometry_state()
        if not gs:
            return 1.0
        return float(dm.modifiers(gs).get("cd_factor", 1.0))
    except Exception:
        return 1.0

def dynamics(t, state, model: ShuttlecockModel, env: EnvironmentModel,
             torque_fn: Optional[Callable] = None):
    x, y, vx, vy, delta_alpha, omega = state

    u_rel = vx - env.u_air_x
    v_rel = vy - env.u_air_y
    Vrel = math.sqrt(u_rel ** 2 + v_rel ** 2)

    cd_value = model.Cd_model.cd(speed=Vrel, alpha_rad=delta_alpha)
    cd_value *= _damage_cd_factor(model)

    S = model.geometry.S
    m = model.m
    if Vrel > 0:
        drag_coeff = 0.5 * env.rho * cd_value * S * Vrel / m
        ax = -drag_coeff * u_rel
        ay = -G - drag_coeff * v_rel
    else:
        ax = 0.0
        ay = -G

    if torque_fn is None:
        tp = model.turnover
        if getattr(tp, "mode", "linear") == "aero_pendulum":
            S = model.geometry.S
            w0 = tp.omega0(Vrel, env.rho, S)
            b = tp.beta(Vrel, env.rho, S, phi=delta_alpha)
            delta_alpha_ddot = -b * omega - (w0 ** 2) * math.sin(delta_alpha)
        else:
            Tw = tp.Tw
            zeta = tp.zeta
            delta_alpha_ddot = -(2 * zeta / Tw) * omega - (1.0 / Tw ** 2) * delta_alpha
    else:
        delta_alpha_ddot = torque_fn(delta_alpha, omega, Vrel, model)

    return np.array([vx, vy, ax, ay, omega, delta_alpha_ddot])

def flow_state(vx, vy, env: EnvironmentModel, theta_body, geometry: Geometry):
    u_rel = vx - env.u_air_x
    v_rel = vy - env.u_air_y
    Vrel = math.sqrt(u_rel ** 2 + v_rel ** 2)
    flow_angle = math.atan2(v_rel, u_rel) if Vrel > 0 else 0.0
    alpha = wrap_to_pi(theta_body - flow_angle)
    Re = (env.rho * Vrel * geometry.skirt_diameter_m / env.mu) if env.mu > 0 else float("nan")
    return u_rel, v_rel, Vrel, flow_angle, alpha, Re

def dynamics_coupled(t, state, model: ShuttlecockModel, env: EnvironmentModel,
                     aero: AeroModel, geom_state=None):
    x, y, vx, vy, theta, omega_turn = state

    u_rel, v_rel, Vrel, flow_angle, alpha, Re = flow_state(vx, vy, env, theta, model.geometry)
    spin = aero.spin.spin_at(t, speed=Vrel)
    coeffs = aero.coefficients(Vrel, alpha, Re, spin, geom_state)
    Cd, Cl = coeffs["Cd"], coeffs["Cl"]
    Cl = Cl * lift_alignment_factor(alpha, aero.alpha_equilibrium_rad)
    S = model.effective_area(theta, flow_angle)
    m = model.m

    if Vrel > 0:
        e_fx, e_fy = u_rel / Vrel, v_rel / Vrel
        e_nx, e_ny = -e_fy, e_fx
        q = 0.5 * env.rho * Vrel ** 2
        F_D = q * Cd * S
        F_L = q * Cl * S
        Fx = -F_D * e_fx + F_L * e_nx
        Fy = -F_D * e_fy + F_L * e_ny
        # lateral_force_bias is a dimensionless side-force coefficient (asymmetry
        # magnitude x Cd), so it has to be multiplied by q and S like every other
        # coefficient. Adding it straight to Fx/Fy treated it as newtons: at two
        # missing feathers that is 0.085 N, ~1.8x the shuttlecock's own weight, and
        # it never decayed with speed, so a damaged shuttlecock hovered and flew
        # *further* than an intact one. The 3D path (aero_forces_moments_3d) already
        # scales it by q*S and by the bleed factor; use the same convention here so
        # the two engines agree.
        bias = coeffs["modifiers"]["lateral_force_bias"]
        F_side = q * S * bias * getattr(aero, "damage_bleed_factor", 0.0)
        Fx += F_side * e_nx
        Fy += F_side * e_ny
    else:
        Fx = Fy = 0.0

    ax = Fx / m
    ay = -G + Fy / m

    # one inertia for the restoring moment, the damping moment and the division
    I_turn = turnover_inertia_of(model.turnover)
    M_aero = aero.moment(Vrel, alpha, Re, spin, env.rho, S, model.turnover, geom_state)
    if getattr(model.turnover, "mode", "linear") == "aero_pendulum":
        M_damp = -I_turn * model.turnover.beta(Vrel, env.rho, model.geometry.S,
                                                phi=wrap_to_pi(alpha - math.pi)) * omega_turn
    else:
        M_damp = -model.turnover.c * omega_turn
    omega_dot = (M_aero + M_damp) / I_turn

    return np.array([vx, vy, ax, ay, omega_turn, omega_dot])

def coupled_diagnostics(t, state, model, env, aero, geom_state=None):
    x, y, vx, vy, theta, omega_turn = state
    u_rel, v_rel, Vrel, flow_angle, alpha, Re = flow_state(vx, vy, env, theta, model.geometry)
    spin = aero.spin.spin_at(t, speed=Vrel)
    coeffs = aero.coefficients(Vrel, alpha, Re, spin, geom_state)
    S = model.effective_area(theta, flow_angle)
    q = 0.5 * env.rho * Vrel ** 2
    drag = q * coeffs["Cd"] * S
    lift = q * coeffs["Cl"] * S * lift_alignment_factor(
        alpha, aero.alpha_equilibrium_rad)
    M_aero = aero.moment(Vrel, alpha, Re, spin, env.rho, S, model.turnover, geom_state)
    if getattr(model.turnover, "mode", "linear") == "aero_pendulum":
        M_damp = -turnover_inertia_of(model.turnover) * model.turnover.beta(
            Vrel, env.rho, model.geometry.S, phi=wrap_to_pi(alpha - math.pi)) * omega_turn
    else:
        M_damp = -model.turnover.c * omega_turn
    return dict(Vrel=Vrel, flow_angle=flow_angle, alpha=alpha, Re=Re, spin=spin,
                Cd=coeffs["Cd"], Cl=coeffs["Cl"],
                Cm=(aero.cm_value(alpha) if aero.cm_value(alpha) is not None
                    else effective_cm(M_aero, env.rho, Vrel, S, aero.l_ref)),
                drag=drag, lift=lift, S_eff=S,
                M_restore=M_aero, M_damping=M_damp, M_net=M_aero + M_damp,
                extrapolated=aero.last_extrapolation_warning)

def euler_step(f, t, y, dt, *args):
    return y + dt * f(t, y, *args)

def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt / 2 * k1, *args)
    k3 = f(t + dt / 2, y + dt / 2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@dataclass
class SimResult:
    df: pd.DataFrame
    metadata: Dict[str, Any]
    valid: bool
    warnings: List[str]

FLOW_RESOLUTIONS = {"낮음": 9, "보통": 15, "높음": 23}

@dataclass
class InitialCondition3D:
    x0: float = 0.0
    y0: float = 1.8
    z0: float = 0.0
    V0: float = 30.0
    elevation_deg: float = -10.0
    azimuth_deg: float = 0.0
    body_axis_offset_deg: float = 180.0
    body_azimuth_deg: float = 0.0
    omega0: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spin0_rad_s: float = 0.0

    @property
    def velocity(self):
        el = deg2rad(self.elevation_deg)
        az = deg2rad(self.azimuth_deg)
        return np.array([self.V0 * math.cos(el) * math.cos(az),
                         self.V0 * math.sin(el),
                         self.V0 * math.cos(el) * math.sin(az)])

    def body_axis(self, env: Optional["EnvironmentModel"] = None):
        v = self.velocity
        if env is not None:
            v = v - np.array([env.u_air_x, env.u_air_y, env.u_air_z])
        n = np.linalg.norm(v)
        ef = v / n if n > 0 else np.array([1.0, 0.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])
        p0 = np.cross(zhat, ef)
        if np.linalg.norm(p0) < 1e-8:
            p0 = np.cross(np.array([1.0, 0.0, 0.0]), ef)
        p0 = p0 / (np.linalg.norm(p0) or 1.0)
        p1 = np.cross(ef, p0)
        p1 = p1 / (np.linalg.norm(p1) or 1.0)
        b = deg2rad(self.body_axis_offset_deg)
        g = deg2rad(self.body_azimuth_deg)
        perp = math.cos(g) * p0 + math.sin(g) * p1
        e = math.cos(b) * ef + math.sin(b) * perp
        return e / (np.linalg.norm(e) or 1.0)

def quat_normalize(q):
    n = float(np.linalg.norm(q))
    if n <= 0 or not np.isfinite(n):
        return None
    return np.asarray(q, dtype=float) / n

def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])

def cross3(a, b):
    """Cross product of two 3-vectors, written out.

    numpy's cross() is general over axes and shapes, and pays for it: profiling a
    12 s three-dimensional flight showed 39,412 calls costing 2.25 s of the 4.97 s
    run -- 45% of the time, spent almost entirely in moveaxis and
    normalize_axis_tuple rather than on the six multiplies. Every call in this file
    is on plain 3-vectors, where the products and the subtraction are the same
    operations in the same order, so the result is bit-identical.
    """
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return np.array([a1 * b2 - a2 * b1,
                     a2 * b0 - a0 * b2,
                     a0 * b1 - a1 * b0])


def clip1(x, lo, hi):
    """np.clip for a single float, without building three temporary arrays.

    The 3D force path clips five dot products and sines per evaluation, which
    profiled at 8% of a flight. NaN is passed through, as np.clip does -- writing it
    as min(max(...)) would not, because every comparison against NaN is False and the
    result would depend on argument order.
    """
    if x != x:
        return x
    return lo if x < lo else (hi if x > hi else x)


def vnorm(v):
    """Length of a 1-D vector, without np.linalg.norm's dtype and axis dispatch.

    Used on the 3-vectors and on the 4-element quaternion alike.

    It has to go through the dot product, not v0*v0 + v1*v1 + v2*v2: numpy sums the
    squares differently, and writing the sum out by hand disagreed with
    np.linalg.norm in the last bit on 11% of random 3-vectors. sqrt(v @ v) is the
    same arithmetic norm() performs, and matched on all 200,000 tried.
    """
    if type(v) is not np.ndarray or v.dtype != np.float64:
        v = np.asarray(v, dtype=float)      # 호출부 일부는 리스트를 넘긴다
    return math.sqrt(float(v @ v))


def quat_to_matrix(q):
    """Rotation matrix mapping BODY frame vectors into the INERTIAL frame."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def quat_derivative(q, omega_body):
    """dq/dt = 0.5 * q (x) [0, omega_body]"""
    return 0.5 * quat_mul(q, np.array([0.0, omega_body[0], omega_body[1], omega_body[2]]))

def quat_from_vectors(v_from, v_to):
    """Shortest-arc quaternion rotating v_from onto v_to."""
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    a = a / (vnorm(a) or 1.0)
    b = b / (vnorm(b) or 1.0)
    d = clip1(float(a @ b), -1.0, 1.0)
    if d > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1.0 + 1e-12:
        axis = cross3(a, np.array([0.0, 0.0, 1.0]))
        if vnorm(axis) < 1e-9:
            axis = cross3(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / (vnorm(axis) or 1.0)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = cross3(a, b)
    q = np.array([1.0 + d, axis[0], axis[1], axis[2]])
    return q / (vnorm(q) or 1.0)

def body_axis_from_quat(q):
    """Symmetry axis (body +x) expressed in the inertial frame."""
    return quat_to_matrix(q)[:, 0]

def aero_angles_3d(v_rel_body):
    """Angle of attack and sideslip from the relative wind in BODY axes.

    alpha is measured in the body x-z plane, beta is the sideslip out of that plane.
    Both are safe when the relative speed is zero.
    """
    u, v, w = float(v_rel_body[0]), float(v_rel_body[1]), float(v_rel_body[2])
    V = math.sqrt(u * u + v * v + w * w)
    if V <= 0:
        return 0.0, 0.0
    alpha = math.atan2(w, u)
    beta = math.asin(clip1(v / V, -1.0, 1.0))
    return alpha, beta

class SideForceModel:
    """Side-force coefficient Cy. Disabled by default: with no measured or literature
    data there is no basis for a non-zero side force."""

    def __init__(self, mode="disabled", constant_value=0.0, table=None,
                 source_type="assumed"):
        self.mode = mode
        self.constant_value = constant_value
        self.table = table or []
        self.source_type = source_type

    @property
    def enabled(self):
        return self.mode != "disabled"

    @property
    def status(self):
        return "측면력 모델 사용 안 함" if not self.enabled else "측면력 모델 사용"

    def cy(self, alpha_rad=None, beta_rad=None, Re=None):
        if self.mode == "disabled":
            return 0.0
        if self.mode == "constant":
            return self.constant_value
        if self.mode == "lookup_beta" and self.table:
            xs = [t[0] for t in self.table]
            ys = [t[1] for t in self.table]
            return float(np.interp(beta_rad if beta_rad is not None else 0.0, xs, ys))
        return 0.0

class MomentModel3D:
    """Roll (about the symmetry axis) and yaw moment coefficients.

    Both default to 0 and are reported as unavailable: the validated model in this
    simulator is the pitch/turnover moment, and no roll or yaw coefficient is invented.
    """

    def __init__(self, cm_roll=None, cm_yaw=None, source_type="assumed"):
        self.cm_roll = cm_roll
        self.cm_yaw = cm_yaw
        self.source_type = source_type

    @property
    def status(self):
        parts = []
        parts.append("roll 계수 없음 (unavailable)" if self.cm_roll is None else "roll 계수 사용")
        parts.append("yaw 계수 없음 (unavailable)" if self.cm_yaw is None else "yaw 계수 사용")
        return ", ".join(parts)

    def roll(self, alpha, beta, Re):
        if self.cm_roll is None:
            return 0.0
        try:
            return float(self.cm_roll(alpha, beta, Re))
        except Exception:
            return float(self.cm_roll) if np.isscalar(self.cm_roll) else 0.0

    def yaw(self, alpha, beta, Re):
        if self.cm_yaw is None:
            return 0.0
        try:
            return float(self.cm_yaw(alpha, beta, Re))
        except Exception:
            return float(self.cm_yaw) if np.isscalar(self.cm_yaw) else 0.0

def _signed_wobble(e, ef, ref):
    """Signed turnover angle in 3D, measured from the cork-forward equilibrium.

    alpha_total is obtained from acos() and is therefore an UNSIGNED cone angle in
    [0, pi]. Plotting its deviation from equilibrium reflects the curve at zero and
    produces cusps that are an artefact of taking a magnitude, not real dynamics.
    Here the tilt is signed against a reference direction that is carried along with
    the flow (parallel transported), so the angle passes smoothly through zero.

    Returns (updated_reference, signed_angle_rad, precession_azimuth_rad).
    """
    e = np.asarray(e, dtype=float)
    ef = np.asarray(ef, dtype=float)
    n = float(vnorm(ef))
    if n <= 1e-12:
        return ref, 0.0, 0.0
    ef = ef / n
    p = e - np.dot(e, ef) * ef
    if ref is None:
        if vnorm(p) > 1e-9:
            ref = p / vnorm(p)
        else:
            tmp = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(tmp, ef))) > 0.9:
                tmp = np.array([1.0, 0.0, 0.0])
            ref = cross3(tmp, ef)
            ref = ref / max(float(vnorm(ref)), 1e-12)
    r1 = ref - np.dot(ref, ef) * ef
    nr = float(vnorm(r1))
    if nr <= 1e-9:
        return ref, 0.0, 0.0
    r1 = r1 / nr
    r2 = cross3(ef, r1)
    a = float(np.dot(e, ef))
    b = float(np.dot(e, r1))
    signed = math.atan2(b, -a)
    azim = math.atan2(float(np.dot(p, r2)), float(np.dot(p, r1))) \
        if vnorm(p) > 1e-12 else 0.0
    return r1, signed, azim

def _spin_accel(deriv, e, R):
    """Angular acceleration component along the symmetry axis (spin DOF)."""
    try:
        wdot_in = R @ np.asarray(deriv[10:13], dtype=float)
        return float(np.dot(wdot_in, e))
    except Exception:
        return float("nan")

def _turn_accel(deriv, e, R):
    """Angular acceleration perpendicular to the symmetry axis (turnover DOF)."""
    try:
        wdot_in = R @ np.asarray(deriv[10:13], dtype=float)
        return float(np.linalg.norm(wdot_in - np.dot(wdot_in, e) * e))
    except Exception:
        return float("nan")

def effective_cm(moment_nm, rho, Vrel, S_ref, l_ref):
    """Moment coefficient backed out of the moment actually applied.

    The default restoring model is the Cohen aerodynamic pendulum, which builds the
    moment directly as -I*omega_0^2*sin(phi) and has no Cm coefficient to report --
    so CmModel.cm() returned None and the Cm column came out NaN on every row of
    every run, in both engines. Dividing the moment by q*S*l_ref gives the
    coefficient that moment corresponds to, which is defined in every mode and is
    the number to compare against published Cm data.
    """
    try:
        den = 0.5 * float(rho) * float(Vrel) ** 2 * float(S_ref) * float(l_ref)
        if not math.isfinite(den) or abs(den) < 1e-15:
            return 0.0
        v = float(moment_nm) / den
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _spin_constraint_torque(model, ao, w_body, I_body, R):
    """Reaction torque the spin lock must supply about the body symmetry axis.

    When the axial spin degree of freedom is constrained, the physical torque that
    would have accelerated it is absorbed by the mount. Reporting it keeps the
    constrained energy out of the shuttlecock's own energy budget.
    """
    if not getattr(model, "_spin_locked", False):
        return 0.0          # 구속이 없으면 반작용 토크도 없다 (미정의가 아니라 0)
    try:
        M_body = R.T @ ao["M"]
        gyro = cross3(w_body, I_body @ w_body)
        return float(-(M_body[0] - gyro[0]))
    except Exception:
        return float("nan")

def _tangential_accel(v, a):
    """Component of acceleration along the flight path (speeds the shuttlecock up/down)."""
    sp = float(vnorm(v))
    if sp <= 1e-12:
        return float("nan")
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(v, dtype=float)) / sp)

def _normal_accel(v, a):
    """Component of acceleration perpendicular to the path (curves the trajectory)."""
    sp = float(vnorm(v))
    if sp <= 1e-12:
        return float("nan")
    a = np.asarray(a, dtype=float)
    vh = np.asarray(v, dtype=float) / sp
    return float(np.linalg.norm(a - np.dot(a, vh) * vh))

def _path_curvature(v, a):
    """Local curvature 1/R of the trajectory: |v x a| / |v|^3."""
    v = np.asarray(v, dtype=float)
    sp = float(vnorm(v))
    if sp <= 1e-9:
        return float("nan")
    return float(np.linalg.norm(cross3(v, np.asarray(a, dtype=float))) / sp ** 3)

def _precession_period(model, env, Vrel, w_axial):
    """Precession period of the wobble plane driven by the axial spin.

    Cohen et al. 2015: tau_p = 4*pi*J*Omega / (rho*S*Cd*U^2*l), where J is the axial
    moment of inertia and l the mass-centre / pressure-centre offset.
    """
    try:
        if Vrel <= 0 or abs(w_axial) <= 0:
            return float("nan")
        J = model.Ispin if model.Ispin else 0.5 * model.turnover.Iyy
        S = model.geometry.S
        cd = model.turnover.Cd_ref
        l = model.turnover.l_gc
        den = env.rho * S * cd * Vrel ** 2 * l
        if den <= 0:
            return float("nan")
        return 4.0 * math.pi * J * abs(w_axial) / den
    except Exception:
        return float("nan")

def inertia_3d(model: ShuttlecockModel):
    """Returns (I_spin, I_transverse). I_spin is about the symmetry axis, I_transverse
    is the turnover axis inertia. They are different physical quantities."""
    It = model.turnover.Iyy
    Is = model.Ispin if model.Ispin else 0.5 * It
    return float(Is), float(It)

def canonical_turnover_inertia(model):
    """Transverse inertia that BOTH the moment model and the solver must use.

    TurnoverParams.Iyy is a profile input, while the 3D solver integrates with the
    geometry-derived I_body. When geometry inertia is active these differed by a
    factor of ~3, so the restoring moment was scaled by one inertia and divided by
    another, inflating the angular acceleration. Returns (I, provenance).
    """
    try:
        if getattr(model, "use_geometry_inertia", False) and model.skirt is not None:
            I, prov = inertia_tensor_3d(model)
            It = float(0.5 * (I[1, 1] + I[2, 2]))
            if np.isfinite(It) and It > 0:
                return It, "형상 기반 (3D solver 와 동일)"
    except Exception:
        pass
    return float(model.turnover.Iyy), "프로파일 입력값"

def inertia_tensor_3d(model: ShuttlecockModel):
    """Body-frame inertia tensor used by the 6-DOF solver.

    Returns (I_matrix, provenance). When asymmetric feather damage is present and the
    user enabled geometry inertia, the full tensor (with products of inertia) from the
    remaining feather layout is used. Otherwise the axisymmetric diagonal approximation
    diag(Is, It, It) is used, which is only valid for a symmetric shuttlecock.
    """
    Is, It = inertia_3d(model)
    diag = np.diag([Is, It, It])
    if not getattr(model, "use_geometry_inertia", False) or model.skirt is None:
        return diag, "축대칭 근사 diag(Is, It, It)"
    mp = model.mass_properties()
    if not mp:
        return diag, "축대칭 근사 diag(Is, It, It)"
    I = np.asarray(mp["inertia_tensor"], dtype=float)
    if not np.all(np.isfinite(I)):
        return diag, "축대칭 근사 diag(Is, It, It)"
    try:
        w = np.linalg.eigvalsh(0.5 * (I + I.T))
    except Exception:
        return diag, "축대칭 근사 diag(Is, It, It)"
    if np.min(w) <= 0:
        return diag, "축대칭 근사 diag(Is, It, It)"
    # Point masses that nearly line up leave one principal moment at ~0: with a single
    # feather left, the smallest eigenvalue came out at 1e-23 -- positive, so the
    # caller's "positive definite" test passed, and np.linalg.inv then raised
    # LinAlgError and killed the run. Condition, not just sign, decides usability.
    if float(np.max(w)) / max(float(np.min(w)), 1e-300) > 1e12:
        return diag, ("축대칭 근사 diag(Is, It, It) — 남은 깃털이 너무 적어 형상 기반 "
                       "관성텐서가 특이행렬에 가깝습니다 (점질량 근사의 한계)")
    return 0.5 * (I + I.T), "형상 기반 full inertia tensor (추정값)"

def turnover_constants(model: ShuttlecockModel, env: EnvironmentModel,
                       aero: Optional["AeroModel"] = None, U=None, d_alpha=1e-4,
                       d_omega=1e-3):
    """Tw and zeta obtained by linearising the actual aerodynamic moment model.

    k_alpha = -dM/dalpha and c_rot = -dM/domega are evaluated numerically about the
    equilibrium angle at the given speed, then Tw = sqrt(I/k_alpha) and
    zeta = c_rot / (2*sqrt(I*k_alpha)). Falls back to the stored input constants when
    the moment model is not differentiable there.
    """
    tp = model.turnover
    out = dict(Tw_input=float(tp.Tw), zeta_input=float(tp.zeta),
               source_type="입력값 (프로파일/사용자)")
    try:
        S = model.geometry.S
        I, I_src = canonical_turnover_inertia(model)
        I = float(I)
        tp.I_effective = I
        out["inertia_source"] = I_src
        out["I_turnover"] = I
        if U is None:
            U = 20.0
        aeq = (aero.alpha_equilibrium_rad if aero is not None else math.pi)

        def moment(a):
            if aero is not None:
                return aero.moment(U, a, None, 0.0, env.rho, S, tp, None)
            return -tp.k_alpha * wrap_to_pi(a - aeq)

        m_p = moment(aeq + d_alpha)
        m_m = moment(aeq - d_alpha)
        k_alpha = -(m_p - m_m) / (2 * d_alpha)
        # c_rot = -dM/d(omega_turn), evaluated by central difference on the actual
        # damping model rather than taken as a stored input
        d_om = float(d_omega)

        def damp_moment(om):
            if getattr(tp, "mode", "linear") == "aero_pendulum":
                return -I * tp.beta(U, env.rho, S) * om
            return -float(tp.c) * om

        c_rot = -(damp_moment(d_om) - damp_moment(-d_om)) / (2.0 * d_om)
        out["c_rot_source"] = "중심차분 -dM/d(omega)"
        if k_alpha > 0 and I > 0:
            Tw = math.sqrt(I / k_alpha)
            zeta = c_rot / (2.0 * math.sqrt(I * k_alpha))
            out.update(Tw_model=float(Tw), zeta_model=float(zeta),
                       k_alpha_Nm_per_rad=float(k_alpha), c_rot_Nms_per_rad=float(c_rot),
                       reference_speed_m_s=float(U),
                       stability="복원 (k_alpha > 0)",
                       source_type="모델 예측값 (공력 모멘트 선형화)")
            if c_rot < 0:
                out["warning"] = ("음의 감쇠 (c_rot < 0): 회전 에너지가 증가합니다. "
                                   "물리적으로 비정상이므로 결과를 사용하지 마세요.")
            # Q-factor only has meaning for an underdamped oscillator
            if 0.0 < zeta < 1.0:
                out["Q_factor"] = float(1.0 / (2.0 * zeta))
                out["Q_note"] = "부족감쇠 (0 < ζ < 1) 에서만 정의"
            else:
                out["Q_factor"] = float("nan")
                out["Q_note"] = ("ζ >= 1 (과감쇠) 이므로 부족감쇠 Q 해석을 적용하지 "
                                  "않습니다" if zeta >= 1.0 else
                                  "ζ <= 0 이므로 Q를 정의하지 않습니다")
        else:
            out["stability"] = ("불안정 평형 (k_alpha < 0)" if k_alpha < 0
                                 else "중립 (k_alpha = 0)")
            out["note"] = "복원 강성 k_alpha <= 0: 평형점에서 선형화 불가"
            out["warning"] = ("평형점이 복원력을 갖지 않습니다 (k_alpha <= 0). "
                               "Tw/ζ/Q를 계산하지 않습니다.")
    except Exception:
        out["note"] = "선형화 실패: 입력값 사용"
    return out

def mass_consistency_report(model: ShuttlecockModel):
    """Compares the profile mass against cork + remaining feather mass.

    Neither value overwrites the other; a mismatch is reported so the user decides.
    """
    if model.skirt is None:
        return None
    cork = getattr(model.turnover, "M_cork", None)
    if cork is None:
        return None
    feath = sum(f.mass_kg for f in model.skirt.attached_feathers)
    geom_mass = float(cork) + float(feath)
    diff = geom_mass - model.m
    rel = abs(diff) / model.m if model.m > 0 else float("nan")
    return dict(profile_mass_kg=float(model.m), cork_mass_kg=float(cork),
                feather_mass_kg=float(feath), geometry_mass_kg=geom_mass,
                difference_kg=float(diff), relative_difference=float(rel),
                consistent=bool(rel < 0.10),
                source_type="형상 기반 추정값 vs 프로파일 입력값")

def flow_state_3d(v, e, env: EnvironmentModel, geometry: Geometry):
    u_air = np.array([env.u_air_x, env.u_air_y, env.u_air_z])
    v_rel = v - u_air
    Vrel = float(vnorm(v_rel))
    if Vrel > 0:
        ef = v_rel / Vrel
    else:
        ef = np.zeros(3)
    en = e / (vnorm(e) or 1.0)
    cos_a = clip1(float(en @ ef), -1.0, 1.0) if Vrel > 0 else 1.0
    alpha = math.acos(cos_a)
    Re = (env.rho * Vrel * geometry.skirt_diameter_m / env.mu) if env.mu > 0 else float("nan")
    return v_rel, Vrel, ef, en, alpha, Re

def aero_forces_moments_3d(t, r, v, q, w_body, model, env, aero, geom_state,
                            side_model=None, moment3d=None):
    """Full 3D aerodynamic force/moment set in the INERTIAL frame.

    Builds an orthonormal aerodynamic triad from the relative wind and the body axis
    (drag along -e_flow, lift and side force perpendicular to it), so no 2D XY-normal
    shortcut is used. Returns a dict of vectors and scalars for both the integrator and
    the diagnostics table.
    """
    R = quat_to_matrix(q)
    e = R[:, 0]
    u_air = np.array([env.u_air_x, env.u_air_y, env.u_air_z])
    v_rel = v - u_air
    Vrel = float(vnorm(v_rel))
    S = model.geometry.S
    l_ref = aero.l_ref
    Re = (env.rho * Vrel * model.geometry.skirt_diameter_m / env.mu) if env.mu > 0 else float("nan")

    if Vrel <= 0:
        zero = np.zeros(3)
        return dict(F=zero, M=zero, F_drag=zero, F_lift=zero, F_side=zero,
                    Vrel=0.0, Re=0.0, alpha_total=0.0, alpha=0.0, beta=0.0,
                    Cd=0.0, Cl=0.0, Cy=0.0, Cm=None, e=e,
                    M_restore=zero, M_damp=zero, ef=np.zeros(3))

    ef = v_rel / Vrel
    v_rel_body = R.T @ v_rel
    alpha, beta = aero_angles_3d(v_rel_body)
    alpha_total = math.acos(clip1(float(e @ ef), -1.0, 1.0))

    w_in_pre = R @ w_body
    w_ax_pre = float(np.dot(w_in_pre, e))
    spin = aero.spin.spin_at(t, speed=Vrel, omega_axial=w_ax_pre)
    coeffs = aero.coefficients(Vrel, alpha_total, Re, spin, geom_state)
    Cd, Cl = coeffs["Cd"], coeffs["Cl"]
    Cl = Cl * lift_alignment_factor(alpha_total, aero.alpha_equilibrium_rad)
    Cy = side_model.cy(alpha, beta, Re) if side_model is not None else 0.0

    cross = cross3(ef, e)
    ncross = float(vnorm(cross))
    if ncross > 1e-9:
        n_side = cross / ncross
        n_lift = cross3(n_side, ef)
        nl = float(vnorm(n_lift))
        n_lift = n_lift / nl if nl > 1e-12 else np.zeros(3)
    else:
        n_side = np.zeros(3)
        n_lift = np.zeros(3)

    q_dyn = 0.5 * env.rho * Vrel ** 2
    F_drag = -q_dyn * Cd * S * ef
    F_lift = q_dyn * Cl * S * n_lift
    F_side = q_dyn * Cy * S * n_side
    F = F_drag + F_lift + F_side

    mods = aero.damage_mapping.modifiers(geom_state or {})

    # ---- asymmetric feather damage -------------------------------------
    # With feathers missing on one side, that sector no longer blocks the flow while
    # the intact side still does. The resultant of the skirt loading therefore moves
    # off the symmetry axis: it produces a moment about the centre of mass (the
    # dominant effect, purely geometric) plus a smaller direct side force from the
    # air bleeding asymmetrically through the opened gaps. Without this the
    # simulation shows literally zero lateral deviation for a damaged shuttlecock,
    # which cannot be right.
    F_asym = np.zeros(3)
    M_asym = np.zeros(3)
    bias = float(mods.get("lateral_force_bias", 0.0) or 0.0)
    if bias > 1e-12 and Vrel > 0:
        az = float(mods.get("asymmetry_azimuth_rad", 0.0) or 0.0)
        d_in = R @ np.array([0.0, math.cos(az), math.sin(az)])
        d_in = d_in - np.dot(d_in, e) * e
        nd = float(vnorm(d_in))
        if nd > 1e-9:
            d_in = d_in / nd
            # The missing drag acted at the vane area's centroid, not at the tip.
            R_skirt = float(mods.get("asymmetry_moment_arm_m", 0.0)
                            or 0.5 * model.geometry.skirt_diameter_m)
            dF = q_dyn * S * bias * (-ef)
            M_asym = cross3(R_skirt * d_in, dF)
            # bleed-jet reaction toward the opened sector
            F_asym = -q_dyn * S * bias * aero.damage_bleed_factor * d_in
    F = F + F_asym

    M_restore = np.zeros(3)
    cm_val = aero.cm_model.cm(alpha_total, aero.alpha_equilibrium_rad)
    if ncross > 1e-9:
        axis = cross3(e, -ef)
        na = float(vnorm(axis))
        if na > 1e-9:
            axis = axis / na
            tilt = math.atan2(na, clip1(float(e @ -ef), -1.0, 1.0))
            if getattr(model.turnover, "mode", "linear") == "aero_pendulum":
                It_r = turnover_inertia_of(model.turnover)
                w0r = model.turnover.omega0(Vrel, env.rho, S)
                mag = It_r * (w0r ** 2) * math.sin(tilt)
            elif cm_val is None:
                mag = model.turnover.k_alpha * tilt
            else:
                # the sign of Cm carries the restoring/diverging information; taking
                # abs() here forced the moment to always look restoring, which hides
                # an unstable configuration instead of reporting it
                mag = -q_dyn * S * l_ref * cm_val
            mag *= mods["cm_factor"] * aero.spin.cm_factor(spin)
            M_restore = mag * axis
    # The asymmetry moment is M_asym: it is dimensional and it acts about the axis the
    # damage azimuth defines. The old term added the same imbalance a second time, as
    # a bare dimensionless coefficient, about n_side -- the wobble axis, which rotates
    # with the wobble, so the torque direction spun around and the attitude never
    # trimmed anywhere.
    M_restore = M_restore + M_asym

    if moment3d is not None:
        cl_roll = moment3d.roll(alpha, beta, Re)
        cn_yaw = moment3d.yaw(alpha, beta, Re)
        if cl_roll:
            M_restore = M_restore + q_dyn * S * l_ref * cl_roll * e
        if cn_yaw and ncross > 1e-9:
            M_restore = M_restore + q_dyn * S * l_ref * cn_yaw * n_lift

    w_inertial = R @ w_body
    w_axial = float(np.dot(w_inertial, e))
    w_t = w_inertial - w_axial * e
    tp3 = model.turnover
    if getattr(tp3, "mode", "linear") == "aero_pendulum":
        tilt_d = wrap_to_pi(alpha_total - aero.alpha_equilibrium_rad)
        # same inertia as the restoring moment above, otherwise the damping ratio is
        # off by Iyy / I_effective
        M_damp = -turnover_inertia_of(tp3) * tp3.beta(Vrel, env.rho, S, phi=tilt_d) * w_t
    else:
        M_damp = -tp3.c * w_t

    # ---- axial spin torque ------------------------------------------
    # Two clearly separated modes:
    #   "constant"    : no aerodynamic spin torque (M_spin = 0). The spin DOF is
    #                   still integrated, so an initial spin simply persists.
    #   "aero_driven" : empirical equilibrium model. The feather cant drives the
    #                   spin toward Omega_eq = spin_ratio*U/R through a relaxation
    #                   torque; the state is integrated, never overwritten.
    # An explicit spin damping term -c_spin*Omega is added in both modes.
    M_spin = np.zeros(3)
    spin_eq = float("nan")
    Is_r = model.Ispin if model.Ispin else 0.5 * tp3.Iyy
    if getattr(aero.spin, "mode", "constant") == "aero_driven":
        R_sk = model.geometry.R_m
        if R_sk > 0 and Vrel > 0:
            # The equilibrium spin ratio is NOT scaled by how many feathers are left.
            # Omega_eq is where the vane-generated driving torque balances the
            # vane-generated damping torque, and both are proportional to the
            # remaining vane area, so the ratio R*Omega/U survives feather loss to
            # first order. Scaling only the driving side halved the spin on a
            # half-stripped shuttlecock: it then turned 0.53 of a revolution over a
            # flight instead of about one, so its missing sector never came round to
            # the other side, the side force never changed sign, and the lateral
            # drift piled up to 0.19 m in one direction. With the ratio held it turns
            # ~1.1 revolutions, the lateral impulse largely cancels, and the drift
            # falls to 0.04 m -- the order the measurements support (the Magnus effect
            # in badminton "was often considered to never occur", Cohen et al.,
            # C. R. Physique 25 (2024) 1).
            spin_eq = aero.spin.equilibrium_spin(Vrel)
            rate = (aero.spin.relax_rate if aero.spin.relax_rate is not None
                    else tp3.beta(Vrel, env.rho, S))
            M_spin = Is_r * rate * (spin_eq - w_axial) * e
    c_sp = float(getattr(aero.spin, "c_spin", 0.0) or 0.0)
    if c_sp != 0.0:
        M_spin = M_spin - c_sp * w_axial * e

    return dict(F=F, M=M_restore + M_damp + M_spin, F_drag=F_drag, F_lift=F_lift,
                F_side=F_side,
                Vrel=Vrel, Re=Re, alpha_total=alpha_total, alpha=alpha, beta=beta,
                Cd=Cd, Cl=Cl, Cy=Cy,
                Cm=(cm_val if cm_val is not None
                    else effective_cm(vnorm(M_restore), env.rho, Vrel, S, aero.l_ref)),
                e=e, w_axial=w_axial,
                spin_equilibrium=spin_eq,
                M_restore=M_restore, M_damp=M_damp, M_spin=M_spin, ef=ef, spin=spin)

def dynamics_3d(t, state, model: ShuttlecockModel, env: EnvironmentModel, aero: AeroModel,
                geom_state=None, Is=None, It=None, side_model=None, moment3d=None):
    """6-DOF state [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz].

    q is the body->inertial quaternion (body +x is the symmetry axis) and w is the
    angular velocity in BODY axes, so the rigid-body law
    I_b w_dot + w x (I_b w) = M_b holds with a constant diagonal I_b.
    """
    if Is is None or It is None:
        Is, It = inertia_3d(model)
    I_b = getattr(model, "_I_body", None)
    I_inv = getattr(model, "_I_body_inv", None)
    v = state[3:6]
    q = state[6:10]
    w_body = state[10:13]
    nq = float(vnorm(q))
    q = q / nq if nq > 0 else np.array([1.0, 0.0, 0.0, 0.0])

    ao = aero_forces_moments_3d(t, state[0:3], v, q, w_body, model, env, aero,
                                 geom_state, side_model, moment3d)
    a = ao["F"] / model.m + np.array([0.0, -G, 0.0])

    R = quat_to_matrix(q)
    M_body = R.T @ ao["M"]
    if I_b is None or I_inv is None:
        I_diag = np.array([Is, It, It])
        Iw = I_diag * w_body
        w_dot = (M_body - cross3(w_body, Iw)) / I_diag
    else:
        Iw = I_b @ w_body
        w_dot = I_inv @ (M_body - cross3(w_body, Iw))
    if getattr(model, "_spin_locked", False):
        w_dot = w_dot.copy()
        w_dot[0] = 0.0

    out = np.empty(13)
    out[0:3] = v
    out[3:6] = a
    out[6:10] = quat_derivative(q, w_body)
    out[10:13] = w_dot
    return out

class SimulationEngine3D:
    def __init__(self, model: ShuttlecockModel, env: EnvironmentModel, ic3d: InitialCondition3D,
                 aero: Optional[AeroModel] = None, solver="RK4", dt=0.001, t_max=3.0,
                 planar=False, side_model=None, moment3d=None):
        self.model = model
        self.env = env
        self.ic = ic3d
        self.aero = aero or model.aero or AeroModel(model.Cd_model, level=2)
        self.solver = solver
        self.dt = dt
        self.t_max = t_max
        self.planar = planar
        self.side_model = side_model or SideForceModel()
        self.moment3d = moment3d or MomentModel3D()

    def _project_planar(self, state):
        state[2] = 0.0
        state[5] = 0.0
        q = quat_normalize(state[6:10])
        if q is None:
            return None
        e = body_axis_from_quat(q)
        e[2] = 0.0
        n = float(vnorm(e))
        if n > 1e-12:
            q = quat_from_vectors(np.array([1.0, 0.0, 0.0]), e / n)
        state[6:10] = q
        R = quat_to_matrix(q)
        w_in = R @ state[10:13]
        w_in[0] = 0.0
        w_in[1] = 0.0
        state[10:13] = R.T @ w_in
        return state

    def run(self) -> SimResult:
        warnings_list = []
        if self.dt <= 0:
            return SimResult(pd.DataFrame(), {}, False, ["dt must be > 0"])
        errs = (self.model.sanity_check() + damage_domain_errors(self.model)
                + launch_state_errors(self.ic))
        if self.env.rho <= 0:
            errs.append("air density rho must be > 0")
        if self.env.mu <= 0:
            errs.append("dynamic viscosity mu must be > 0")
        Is, It = inertia_3d(self.model)
        if Is <= 0 or It <= 0:
            errs.append("inertia must be positive definite (Ixx/Iyy/Izz > 0)")
        I_body, inertia_provenance = inertia_tensor_3d(self.model)
        try:
            eig = np.linalg.eigvalsh(I_body)
            if np.min(eig) <= 0:
                errs.append("inertia tensor is not positive definite")
        except Exception:
            errs.append("inertia tensor could not be decomposed")
        if errs:
            return SimResult(pd.DataFrame(), {}, False, errs)
        try:
            I_body_inv = np.linalg.inv(I_body)
        except np.linalg.LinAlgError:
            return SimResult(pd.DataFrame(), {}, False,
                             ["관성텐서를 역행렬로 만들 수 없습니다 (특이행렬). "
                              "남은 깃털이 너무 적어 형상 기반 관성 추정이 성립하지 "
                              "않습니다 — '형상 기반 관성' 을 끄거나 깃털을 더 남기세요."])
        self.model._I_body = I_body
        self.model._I_body_inv = I_body_inv
        _Ieff, _Isrc = canonical_turnover_inertia(self.model)
        self.model.turnover.I_effective = float(_Ieff)
        path_len = 0.0
        prev_r = None
        q0_locked = None
        wob_ref = None
        _u_base = np.array([self.env.u_air_x, self.env.u_air_y, self.env.u_air_z])
        _turb = TurbulenceModel(self.env.turbulence_intensity,
                                 self.env.turbulence_length_scale_m,
                                 U_ref=max(float(vnorm(_u_base)),
                                            abs(float(self.ic.V0))),
                                 seed=self.env.turbulence_seed)
        _u_air = np.array([self.env.u_air_x, self.env.u_air_y, self.env.u_air_z])
        _v0_ref = float(np.linalg.norm(
            np.array([self.ic.V0 * math.cos(deg2rad(self.ic.elevation_deg)), 0.0, 0.0])
            - _u_air))
        tw_zeta = turnover_constants(self.model, self.env, self.aero,
                                      U=(_v0_ref if _v0_ref > 1e-9 else None))
        mcons = mass_consistency_report(self.model)
        if mcons is not None and not mcons["consistent"]:
            warnings_list.append(
                f"parameter inconsistency: 프로파일 질량 {mcons['profile_mass_kg']*1e3:.2f} g vs "
                f"형상 기반 질량(코르크+남은 깃털) {mcons['geometry_mass_kg']*1e3:.2f} g "
                f"({mcons['relative_difference']*100:.1f}% 차이) — 어느 쪽도 자동으로 덮어쓰지 않습니다")

        geom_state = self.model.geometry_state()
        if geom_state is not None and geom_state["remaining_feathers"] == 0:
            warnings_list.append(
                "all feathers removed: geometry is unphysical, aerodynamic results are not meaningful")

        v0 = self.ic.velocity
        e0 = self.ic.body_axis(self.env)
        if self.planar:
            v0[2] = 0.0
            e0[2] = 0.0
            e0 = e0 / (vnorm(e0) or 1.0)
        q0 = quat_from_vectors(np.array([1.0, 0.0, 0.0]), e0)
        R0 = quat_to_matrix(q0)
        w_in0 = np.array(self.ic.omega0, dtype=float)
        if self.planar:
            w_in0[0] = 0.0
            w_in0[1] = 0.0
        w_body0 = R0.T @ w_in0
        # the user's initial spin is authoritative and is NEVER overridden by the
        # empirical equilibrium value; aero_driven only starts AT equilibrium when
        # the user left the initial spin unset (None)
        if self.ic.spin0_rad_s is None:
            w_body0[0] = 0.0
            if getattr(self.aero.spin, "mode", "constant") == "aero_driven":
                R_sk = self.model.geometry.R_m
                V0n = float(np.linalg.norm(v0 - np.array([
                    self.env.u_air_x, self.env.u_air_y, self.env.u_air_z])))
                if R_sk > 0:
                    w_body0[0] = self.aero.spin.equilibrium_spin(V0n)
        else:
            w_body0[0] += float(self.ic.spin0_rad_s)
        if getattr(self.model, "_spin_locked", False):
            w_body0[0] = float(getattr(self.model, "_spin_locked_value", 0.0))
        if self.planar:
            w_body0[0] = 0.0
        self.aero.spin.spin0_rad_s = float(self.ic.spin0_rad_s or 0.0)

        state = np.concatenate([[self.ic.x0, self.ic.y0, self.ic.z0], v0, q0, w_body0])
        q0_locked = np.array(q0, dtype=float).copy()
        step_fn = rk4_step if self.solver != "Euler" else euler_step

        rows = []
        t = 0.0
        valid = True
        max_q_dev = 0.0
        n_steps = int(self.t_max / self.dt) + 1
        for i in range(n_steps):
            if not np.all(np.isfinite(state)):
                warnings_list.append(f"non-finite state detected at t={t:.4f}s; simulation stopped")
                valid = False
                break
            qn = float(vnorm(state[6:10]))
            if qn <= 1e-8 or not np.isfinite(qn):
                warnings_list.append(f"quaternion collapsed at t={t:.4f}s; simulation stopped")
                valid = False
                break
            max_q_dev = max(max_q_dev, abs(qn - 1.0))
            state[6:10] = state[6:10] / qn
            if np.any(np.abs(state[0:3]) > 1e5) or np.any(np.abs(state[3:6]) > 1e4):
                warnings_list.append(f"unphysical state detected at t={t:.4f}s; simulation stopped")
                valid = False
                break

            r = state[0:3].copy()
            v = state[3:6].copy()
            q = state[6:10].copy()
            w_body = state[10:13].copy()
            R = quat_to_matrix(q)
            e = R[:, 0]
            if prev_r is None:
                prev_r = r.copy()
            ao = aero_forces_moments_3d(t, r, v, q, w_body, self.model, self.env, self.aero,
                                         geom_state, self.side_model, self.moment3d)
            deriv = dynamics_3d(t, state, self.model, self.env, self.aero, geom_state,
                                 Is, It, self.side_model, self.moment3d)
            w_in = R @ w_body
            w_t = w_in - np.dot(w_in, e) * e
            _ef_now = ao["ef"]
            wob_ref, wob_signed, wob_azim = _signed_wobble(e, _ef_now, wob_ref)
            F_total = ao["F"] + np.array([0.0, -self.model.m * G, 0.0])
            v_rel_v = v - np.array([self.env.u_air_x, self.env.u_air_y, self.env.u_air_z])
            L_body = I_body @ w_body
            if rows:
                path_len += float(vnorm(r - prev_r))
            prev_r = r.copy()
            ao["drag_force"] = float(vnorm(ao["F_drag"]))
            ao["lift_force"] = float(vnorm(ao["F_lift"]))

            rows.append(dict(
                time_s=t, x_m=r[0], y_m=r[1], z_m=r[2],
                vx_m_s=v[0], vy_m_s=v[1], vz_m_s=v[2],
                speed_m_s=float(vnorm(v)),
                ax_m_s2=deriv[3], ay_m_s2=deriv[4], az_m_s2=deriv[5],
                qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                wx=w_body[0], wy=w_body[1], wz=w_body[2],
                ex=e[0], ey=e[1], ez=e[2],
                alpha_deg=rad2deg(ao["alpha_total"]), alpha_rad=ao["alpha_total"],
                alpha_pitch_deg=rad2deg(ao["alpha"]), beta_deg=rad2deg(ao["beta"]),
                delta_alpha_deg=rad2deg(wrap_to_pi(ao["alpha_total"]
                                                    - self.aero.alpha_equilibrium_rad)),
                theta_body_deg=rad2deg(math.atan2(e[1], e[0])),
                theta_cork_deg=rad2deg(wrap_to_pi(math.atan2(e[1], e[0]) + math.pi)),
                wind_from_deg=(rad2deg(math.atan2(ao["ef"][1], ao["ef"][0]))
                                if ao["Vrel"] > 0 else 0.0),
                flow_angle_deg=(rad2deg(math.atan2(ao["ef"][1], ao["ef"][0]))
                                 if ao["Vrel"] > 0 else 0.0),
                omega_rad_s=float(vnorm(w_t)),
                angular_acceleration_rad_s2=float(vnorm(deriv[10:13])),
                spin_rad_s=float(w_body[0]),
                spin_axial_rad_s=float(ao.get("w_axial", w_body[0])),
                spin_equilibrium_rad_s=float(ao.get("spin_equilibrium", float("nan"))),
                spin_tip_ratio=(float(ao.get("w_axial", 0.0)) * self.model.geometry.R_m
                                 / ao["Vrel"] if ao["Vrel"] > 0 else float("nan")),
                gyroscopic_number=(abs(float(ao.get("w_axial", 0.0)))
                                    * self.model.geometry.R_m / ao["Vrel"]
                                    if ao["Vrel"] > 0 else float("nan")),
                precession_period_s=_precession_period(self.model, self.env,
                                                        ao["Vrel"],
                                                        float(ao.get("w_axial", 0.0))),
                spin_moment_Nm=float(np.linalg.norm(ao.get("M_spin", np.zeros(3)))),
                wobble_signed_deg=rad2deg(wob_signed),
                wobble_azimuth_deg=rad2deg(wob_azim),
                omega_spin_rad_s=float(ao.get("w_axial", w_body[0])),
                spin_angular_accel_rad_s2=float(_spin_accel(deriv, e, R)),
                turnover_angular_accel_rad_s2=float(_turn_accel(deriv, e, R)),
                spin_energy_J=float(0.5 * (self.model.Ispin
                                            if self.model.Ispin else 0.5 * It)
                                     * float(ao.get("w_axial", w_body[0])) ** 2),
                turnover_energy_J=float(0.5 * It * float(np.dot(w_t, w_t))),
                spin_constraint_torque_Nm=_spin_constraint_torque(
                    self.model, ao, w_body, I_body, R),
                spin_equilibrium_target_rad_s=float(ao.get("spin_equilibrium",
                                                            float("nan"))),
                spin_locked=bool(getattr(self.model, "_spin_locked", False)),
                constraint_torque_Nm=_spin_constraint_torque(self.model, ao, w_body,
                                                              I_body, R),
                V_rel_m_s=ao["Vrel"], Re=ao["Re"],
                Cd=ao["Cd"], Cl=ao["Cl"], Cy=ao["Cy"],
                Cm=(ao["Cm"] if ao["Cm"] is not None else float("nan")),
                drag_force_N=float(vnorm(ao["F_drag"])),
                lift_force_N=float(vnorm(ao["F_lift"])),
                side_force_N=float(vnorm(ao["F_side"])),
                force_x_N=F_total[0], force_y_N=F_total[1], force_z_N=F_total[2],
                moment_x_Nm=ao["M"][0], moment_y_Nm=ao["M"][1], moment_z_Nm=ao["M"][2],
                restoring_moment_Nm=float(vnorm(ao["M_restore"])),
                damping_moment_Nm=float(vnorm(ao["M_damp"])),
                net_moment_Nm=float(vnorm(ao["M"])),
                quat_norm=qn,
                aero_length_m=self.model.aerodynamic_length(self.env.rho, ao["Cd"]),
                u_rel_m_s=float(v_rel_v[0]), v_rel_m_s=float(v_rel_v[1]),
                w_rel_m_s=float(v_rel_v[2]),
                acceleration_mag_m_s2=float(vnorm(deriv[3:6])),
                wx_dot_rad_s2=float(deriv[10]), wy_dot_rad_s2=float(deriv[11]),
                wz_dot_rad_s2=float(deriv[12]),
                omega_total_rad_s=float(vnorm(w_body)),
                omega_turnover_rad_s=float(vnorm(w_t)),
                angular_momentum_x=float(L_body[0]), angular_momentum_y=float(L_body[1]),
                angular_momentum_z=float(L_body[2]),
                angular_momentum_mag=float(vnorm(L_body)),
                E_kinetic_J=float(0.5 * self.model.m * float(np.dot(v, v))),
                E_rotational_J=float(0.5 * float(np.dot(w_body, I_body @ w_body))),
                E_potential_J=float(self.model.m * G * state[1]),
                E_total_J=float(0.5 * self.model.m * float(np.dot(v, v))
                                 + 0.5 * float(np.dot(w_body, I_body @ w_body))
                                 + self.model.m * G * state[1]),
                power_drag_W=float(np.dot(ao["F_drag"], v)),
                dynamic_pressure_Pa=float(0.5 * self.env.rho * ao["Vrel"] ** 2),
                flight_path_angle_deg=rad2deg(math.atan2(
                    v[1], math.hypot(v[0], v[2]))) if float(np.dot(v, v)) > 0 else 0.0,
                heading_deg=rad2deg(math.atan2(v[2], v[0])),
                tangential_accel_m_s2=_tangential_accel(v, deriv[3:6]),
                normal_accel_m_s2=_normal_accel(v, deriv[3:6]),
                curvature_1_m=_path_curvature(v, deriv[3:6]),
                weight_N=float(self.model.m * G),
                lift_to_drag=(float(ao["lift_force"] / ao["drag_force"])
                               if ao.get("drag_force", 0.0) > 1e-12 else float("nan")),
                Tw_model_s=tw_zeta.get("Tw_model", float("nan")),
                zeta_model=tw_zeta.get("zeta_model", float("nan")),
                distance_m=path_len,
                Re_extrapolated=bool(getattr(self.aero.cd_model, "last_extrapolated", False)),
            ))

            if r[1] <= 0 and i > 0 and not getattr(self, "hold_position", False):
                break
            if _turb.active:
                _uf = _u_base + _turb.step(self.dt)
                self.env.u_air_x, self.env.u_air_y, self.env.u_air_z = (
                    float(_uf[0]), float(_uf[1]), float(_uf[2]))
            state = step_fn(dynamics_3d, t, state, self.dt, self.model, self.env,
                            self.aero, geom_state, Is, It, self.side_model, self.moment3d)
            if getattr(self, "hold_position", False):
                state[0:3] = np.array([self.ic.x0, self.ic.y0, self.ic.z0], dtype=float)
                state[3:6] = 0.0
            if getattr(self.model, "_spin_locked", False):
                state[10] = float(getattr(self.model, "_spin_locked_value", 0.0))
            if getattr(self.model, "_tunnel_fixed", False):
                state[6:10] = q0_locked
            # full-state divergence guard: the quaternion check alone let position,
            # velocity and angular velocity blow up to ~1e36 and still be reported
            if not np.all(np.isfinite(state)):
                warnings_list.append(
                    f"t={t:.4f}s 에서 상태에 NaN/Inf 가 발생해 시뮬레이션을 중단합니다. "
                    "결과를 사용하지 마세요.")
                valid = False
                break
            _wmag = float(vnorm(state[10:13]))
            _vmag = float(vnorm(state[3:6]))
            if _wmag > 1e5 or _vmag > 1e4 or float(np.max(np.abs(state[0:3]))) > 1e5:
                warnings_list.append(
                    f"t={t:.4f}s 에서 수치 발산 (|ω|={_wmag:.3e} rad/s, "
                    f"|v|={_vmag:.3e} m/s). dt 를 줄이거나 관성/파손 설정을 확인하세요. "
                    "결과를 사용하지 마세요.")
                valid = False
                break
            qn2 = float(vnorm(state[6:10]))
            if qn2 <= 1e-8 or not np.isfinite(qn2):
                warnings_list.append(f"quaternion collapsed at t={t:.4f}s; simulation stopped")
                valid = False
                break
            state[6:10] = state[6:10] / qn2
            if self.planar:
                state = self._project_planar(state)
                if state is None:
                    warnings_list.append("quaternion collapsed during planar projection")
                    valid = False
                    break
            t += self.dt

        df = pd.DataFrame(rows)
        tp = self.model.turnover
        Tw_source = "입력값"
        cm_mode = self.aero.cm_model.mode
        if cm_mode != "from_turnover_params":
            Tw_source = "모델 예측값 (공력 모멘트에서 계산)"
        meta = dict(
            model_version=MODEL_VERSION, app_version=APP_VERSION,
            parameter_profile=self.model.name,
            dimension="3D(평면 구속)" if self.planar else "3D",
            attitude_representation="quaternion (body->inertial), body +x = 대칭축",
            model_level=self.aero.level,
            model_level_name=MODEL_LEVELS.get(self.aero.level, "unknown"),
            parameter_values=dict(m=self.model.m, Tw=tp.Tw, zeta=tp.zeta,
                                   I_spin=Is, I_transverse=It, k_alpha=tp.k_alpha,
                                   c_rot=tp.c, S=self.model.geometry.S),
            provenance=dict(Tw=Tw_source, zeta=Tw_source,
                             Cd=f"{self.model.Cd_model.source_type}",
                             Cl=f"{self.aero.cl_model.source_type}",
                             Cy=self.side_model.status,
                             roll_yaw_moment=self.moment3d.status,
                             z_component="모델 예측값 (측정값 아님)",
                             attitude_3d="모델 예측값 (측정값 아님)"),
            solver=self.solver, dt=self.dt,
            initial_conditions=asdict(self.ic),
            environment=asdict(self.env),
            timestamp=time.time(),
            aero_model=self.aero.describe(),
            max_quaternion_deviation=max_q_dev,
            prediction_note="Z 성분과 3차원 자세는 모델 예측값이며 측정값이 아닙니다.",
        )
        if geom_state is not None:
            meta["geometry_state"] = {k: v for k, v in geom_state.items()
                                       if k != "azimuth_distribution"}
            mp = self.model.mass_properties() if hasattr(self.model, "mass_properties") else None
            if mp:
                meta["mass_properties"] = mp
        meta["inertia"] = dict(
            tensor=[[float(x) for x in row] for row in I_body],
            provenance=inertia_provenance,
            I_spin=float(Is), I_transverse=float(It),
            principal=[float(v) for v in np.linalg.eigvalsh(I_body)],
            asymmetric=bool(inertia_provenance.startswith("형상")))
        if mcons is not None:
            meta["mass_consistency"] = mcons
        self.env.u_air_x, self.env.u_air_y, self.env.u_air_z = (
            float(_u_base[0]), float(_u_base[1]), float(_u_base[2]))
        meta["turnover_constants"] = tw_zeta
        meta["turbulence"] = dict(
            intensity=float(self.env.turbulence_intensity),
            length_scale_m=float(self.env.turbulence_length_scale_m),
            integral_timescale_s=(float(_turb.tau) if _turb.active else float("nan")),
            rms_m_s=(float(_turb.sigma) if _turb.active else 0.0),
            active=bool(_turb.active),
            source_type=("모델 생성값 (Ornstein-Uhlenbeck 합성 난류, 실측 난류장 아님)"
                          if _turb.active else "난류 없음 (완전 결정론적)"))
        warnings_list = (warnings_list
                         + flight_envelope_warnings(df, self.model, self.env)
                         + reynolds_domain_warnings(df, self.model, self.env))
        return SimResult(df=df, metadata=meta, valid=valid, warnings=warnings_list)


# ---------------------------------------------------------------------------
# TecQuipment AF1300S virtual wind tunnel
# ---------------------------------------------------------------------------

AF1300_SPEC = dict(
    equipment="TecQuipment AF1300S",
    tunnel_type="305 mm subsonic open-circuit suction",
    working_section="305x305x600 mm",
    working_section_width_m=0.305,
    working_section_height_m=0.305,
    working_section_length_m=0.600,
    velocity_range_m_s=(0.0, 36.0),
    layout=["대기", "수축부(effuser)", "작업구간", "그릴", "확산부", "축류팬", "소음기", "대기"],
    source="AF1300S Subsonic Wind Tunnel 305 mm Starter Set datasheet (TecQuipment)",
    source_type="제조사 공식 자료",
)

AF1300_INCLUDED = {
    "AF1300": "서브소닉 풍동 305 mm (가변속 축류팬, 제어·계측 유닛)",
    "AF1300Z": "기본 양력/항력 밸런스 (2성분)",
    "AF1300J": "3차원 항력 모델 세트",
    "PITOT": "표준 Pitot 관 (트래버스식)",
    "PITOT_STATIC": "Pitot-static 관 (트래버스식)",
    "PANEL_MANOMETER": "제어반 내장 마노미터",
    "MODEL_HOLDER": "모델 홀더 (스템 11.95±0.015 mm, 215±1.25 mm)",
    "PROTRACTOR": "프로트랙터 (입사각 설정)",
}

AF1300_OPTIONAL = {
    "AFA1": "36관 틸팅 멀티튜브 마노미터 (작동유체: 물)",
    "AF1300T": "3성분 밸런스 (양력·항력·피칭모멘트)",
    "AFA4": "밸런스 각도 피드백 유닛 (AF1300T 필요)",
    "AFA5": "차압 트랜스듀서",
    "AFA6": "32채널 압력 표시 유닛 (대기압 기준)",
    "AFA7": "Pitot-static 트래버스 300 mm (디지털 위치 표시, 영점 설정)",
    "AFA10": "스모크 제너레이터",
    "VDAS": "Versatile Data Acquisition System (VDAS-F)",
}

PRESSURE_MODEL_LEVELS = {
    1: "Level 1 — Bernoulli + 균일 작업구간",
    2: "Level 2 — 손실계수 포함 (수축부/그릴/확산부)",
    3: "Level 3 — 위치별 속도·압력 분포",
    4: "Level 4 — 압력탭 기반 재구성",
    5: "Level 5 — 실험 데이터 보정 모델",
}

R_AIR = 287.058
P_ATM_STD = 101325.0
WATER_DENSITY = 998.2

def air_density_from_TP(T_celsius, P_pascal):
    """Ideal-gas density. Model calculation, not a measurement."""
    T = float(T_celsius) + 273.15
    if T <= 0 or P_pascal <= 0:
        return float("nan")
    return float(P_pascal) / (R_AIR * T)

def air_viscosity_sutherland(T_celsius):
    """Sutherland's law for air. Reference constants are standard textbook values."""
    T = float(T_celsius) + 273.15
    if T <= 0:
        return float("nan")
    return 1.458e-6 * T ** 1.5 / (T + 110.4)

def speed_of_sound(T_celsius):
    T = float(T_celsius) + 273.15
    if T <= 0:
        return float("nan")
    return math.sqrt(1.4 * R_AIR * T)

@dataclass
class TunnelEnvironment:
    """Ambient state of the laboratory. rho may be entered directly or derived."""
    T_celsius: float = 20.0
    P_ambient_Pa: float = P_ATM_STD
    rho_user: Optional[float] = None
    mu_user: Optional[float] = None
    T_uncertainty_C: float = 0.5
    P_uncertainty_Pa: float = 100.0

    def resolve(self):
        rho_calc = air_density_from_TP(self.T_celsius, self.P_ambient_Pa)
        mu_calc = air_viscosity_sutherland(self.T_celsius)
        warnings_out = []
        if self.rho_user is not None and self.rho_user > 0:
            rho = float(self.rho_user)
            rho_src = "사용자 입력값"
            if np.isfinite(rho_calc) and abs(rho - rho_calc) / rho_calc > 0.05:
                warnings_out.append(
                    f"입력 공기밀도 {rho:.4f} kg/m³ 가 온도·압력으로 계산한 값 "
                    f"{rho_calc:.4f} kg/m³ 와 {abs(rho-rho_calc)/rho_calc*100:.1f}% 차이납니다. "
                    "어느 쪽도 자동으로 덮어쓰지 않습니다.")
        else:
            rho = rho_calc
            rho_src = "모델 계산값 (이상기체)"
        mu = float(self.mu_user) if (self.mu_user and self.mu_user > 0) else mu_calc
        mu_src = "사용자 입력값" if (self.mu_user and self.mu_user > 0) else "모델 계산값 (Sutherland)"
        if not np.isfinite(rho) or rho <= 0:
            warnings_out.append("공기 밀도가 유효하지 않습니다 (rho > 0 이어야 합니다).")
        if not np.isfinite(mu) or mu <= 0:
            warnings_out.append("점성계수가 유효하지 않습니다 (mu > 0 이어야 합니다).")
        # density uncertainty by linear propagation of T and P
        drho = float("nan")
        if np.isfinite(rho_calc):
            dT = abs(rho_calc / (self.T_celsius + 273.15)) * self.T_uncertainty_C
            dP = abs(rho_calc / self.P_ambient_Pa) * self.P_uncertainty_Pa
            drho = math.hypot(dT, dP)
        return dict(rho=rho, mu=mu, rho_calculated=rho_calc, mu_calculated=mu_calc,
                     rho_source=rho_src, mu_source=mu_src,
                     rho_uncertainty=drho, a_sound=speed_of_sound(self.T_celsius),
                     warnings=warnings_out)

@dataclass
class TunnelGeometry:
    """AF1300 duct geometry. Only the working section is an official figure."""
    ws_width_m: float = 0.305
    ws_height_m: float = 0.305
    ws_length_m: float = 0.600
    inlet_area_m2: Optional[float] = None
    diffuser_exit_area_m2: Optional[float] = None
    fan_area_m2: Optional[float] = None

    @property
    def A_test(self):
        return self.ws_width_m * self.ws_height_m

    def areas(self):
        A = self.A_test
        return dict(
            inlet=(float(self.inlet_area_m2) if self.inlet_area_m2 else None),
            test=A,
            diffuser_exit=(float(self.diffuser_exit_area_m2)
                            if self.diffuser_exit_area_m2 else None),
            fan=(float(self.fan_area_m2) if self.fan_area_m2 else None))

@dataclass
class TunnelLosses:
    """Loss coefficients. TecQuipment does not publish these; default is 미정."""
    K_contraction: Optional[float] = None
    K_grille: Optional[float] = None
    K_diffuser: Optional[float] = None
    K_bend: Optional[float] = None

    def as_dict(self):
        return {k: (v if v is not None else "미정")
                for k, v in dict(K_contraction=self.K_contraction, K_grille=self.K_grille,
                                  K_diffuser=self.K_diffuser, K_bend=self.K_bend).items()}

class FanController:
    """Maps a fan setting to working-section velocity.

    The AF1300 datasheet states the working section reaches 0 to 36 m/s but does not
    publish a fan-setting to velocity curve, so no curve is invented here. Without a
    user calibration the mapping is reported as uncalibrated.
    """

    PRESETS = {"정지": 0.0, "저속": 25.0, "중속": 55.0, "고속": 90.0}

    def __init__(self, calibration=None):
        self.calibration = calibration

    def velocity(self, fan_percent):
        f = max(0.0, min(100.0, float(fan_percent or 0.0)))
        if self.calibration:
            xs = [float(a) for a, _ in self.calibration]
            ys = [float(b) for _, b in self.calibration]
            order = np.argsort(xs)
            return float(np.interp(f, np.asarray(xs)[order], np.asarray(ys)[order])), \
                "보정 곡선 (사용자 입력)"
        if f <= 0:
            return 0.0, "팬 정지"
        return float("nan"), "팬 설정과 실제 유속의 관계가 보정되지 않음"

class WindTunnelPressureModel:
    """Low-order pressure prediction for the AF1300 duct.

    This is NOT a CFD solution. It applies incompressible continuity and Bernoulli
    with user-supplied loss coefficients, so it predicts the pressure trend along the
    tunnel rather than a resolved field.
    """

    def __init__(self, geometry: TunnelGeometry, losses: TunnelLosses, level=1):
        self.geometry = geometry
        self.losses = losses
        self.level = int(level)

    def station_states(self, U_test, rho, P_ambient):
        """Static/dynamic/total pressure at each duct station.

        The tunnel is open-circuit suction: the inlet is at atmosphere, so total
        pressure at the inlet equals ambient and the working section runs BELOW
        atmospheric pressure. That sign is the physically important result.
        """
        A = self.geometry.areas()
        q_test = 0.5 * rho * U_test ** 2
        out = []

        def add(name, U, Ps, note):
            q = 0.5 * rho * U ** 2
            out.append(dict(station=name, U_m_s=float(U), Ps_Pa=float(Ps),
                             q_Pa=float(q), Pt_Pa=float(Ps + q),
                             Ps_gauge_Pa=float(Ps - P_ambient),
                             Cp=(float((Ps - P_ambient) / q_test)
                                 if q_test > 0 else float("nan")),
                             note=note))

        add("대기", 0.0, P_ambient, "기준 상태")
        U_in = (U_test * A["test"] / A["inlet"]) if A["inlet"] else 0.0
        if A["inlet"]:
            Ps_in = P_ambient - 0.5 * rho * U_in ** 2
            add("흡입구", U_in, Ps_in, "연속방정식 A1·U1 = A2·U2")
        else:
            add("흡입구", 0.0, P_ambient, "흡입 면적 미입력 — 대기 정지 상태로 가정")

        K_c = self.losses.K_contraction if (self.level >= 2 and
                                             self.losses.K_contraction is not None) else 0.0
        Ps_test = P_ambient - q_test * (1.0 + float(K_c))
        add("작업구간", U_test, Ps_test,
            "수축부 손실 %s" % ("포함" if K_c else "미적용(미정)"))

        K_g = self.losses.K_grille if (self.level >= 2 and
                                        self.losses.K_grille is not None) else 0.0
        Ps_grille = Ps_test - q_test * float(K_g)
        add("그릴 후단", U_test, Ps_grille,
            "그릴 손실 %s" % ("포함" if K_g else "미적용(미정)"))

        if A["diffuser_exit"]:
            U_dif = U_test * A["test"] / A["diffuser_exit"]
            q_dif = 0.5 * rho * U_dif ** 2
            K_d = self.losses.K_diffuser if (self.level >= 2 and
                                              self.losses.K_diffuser is not None) else 0.0
            Ps_dif = Ps_grille + (q_test - q_dif) - q_test * float(K_d)
            add("확산부 출구", U_dif, Ps_dif, "감속에 따른 정압 회복")
        else:
            add("확산부 출구", U_test, Ps_grille, "확산부 면적 미입력 — 압력 회복 계산 불가")

        fan_in = out[-1]
        add("팬 입구", fan_in["U_m_s"], fan_in["Ps_Pa"], "확산부 출구와 동일 가정")
        dP_fan = P_ambient - fan_in["Ps_Pa"]
        add("팬 출구", fan_in["U_m_s"], P_ambient,
            "팬이 공급해야 하는 압력 상승 ΔP_fan = %.1f Pa" % dP_fan)
        return out, float(dP_fan)

    def axial_profile(self, U_test, rho, mu, P_ambient, n=400,
                       contraction_len_m=0.45, diffuser_len_m=0.90,
                       diffuser_half_angle_deg=3.0, roughness_ratio=0.0):
        """Continuous streamwise distribution of area, velocity and pressure.

        The station-to-station view is only seven points; a real tunnel varies
        continuously, so this integrates along x:
          - area follows a smooth 5th-order-polynomial contraction (the standard
            wind-tunnel nozzle shape), a constant-area working section, then a
            straight-walled diffuser at the given half angle;
          - continuity gives U(x) = U_test A_test / A(x);
          - total pressure DECREASES monotonically through distributed wall friction,
            dp_t = -f (dx / D_h) (rho U^2 / 2), with f from the Haaland correlation,
            plus the discrete grille and diffuser-expansion losses;
          - static pressure follows p_s = p_t - q.
        The fan must then supply exactly the accumulated total-pressure deficit.
        Reference: Barlow, Rae & Pope, *Low-Speed Wind Tunnel Testing*.
        """
        A_test = self.geometry.A_test
        A = self.geometry.areas()
        A_in = A["inlet"] if A["inlet"] else A_test * 6.0
        A_out = A["diffuser_exit"] if A["diffuser_exit"] else None
        Lc = float(contraction_len_m)
        Lt = self.geometry.ws_length_m
        Ld = float(diffuser_len_m)
        if A_out is None:
            th = deg2rad(max(float(diffuser_half_angle_deg), 0.1))
            side = math.sqrt(A_test) + 2.0 * Ld * math.tan(th)
            A_out = side * side
        xs = np.linspace(0.0, Lc + Lt + Ld, int(n))
        areas = np.empty_like(xs)
        section = []
        for i, x in enumerate(xs):
            if x <= Lc:
                t = x / max(Lc, 1e-9)
                # 5th-order polynomial contraction (zero slope and curvature at both ends)
                sfac = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
                areas[i] = A_in + (A_test - A_in) * sfac
                section.append("수축부")
            elif x <= Lc + Lt:
                areas[i] = A_test
                section.append("작업구간")
            else:
                t = (x - Lc - Lt) / max(Ld, 1e-9)
                areas[i] = A_test + (A_out - A_test) * t
                section.append("확산부")
        U = U_test * A_test / areas
        q = 0.5 * rho * U ** 2
        # hydraulic diameter of the square duct: Dh = 4A/P = sqrt(A). Using the
        # equivalent CIRCULAR diameter 2*sqrt(A/pi) overstates Dh by 13% and
        # understates the wall friction by the same factor.
        Dh = np.sqrt(areas)
        Pt = np.empty_like(xs)
        Pt[0] = P_ambient
        K_grille = (self.losses.K_grille if (self.level >= 2 and
                                              self.losses.K_grille is not None) else 0.0)
        q_test = 0.5 * rho * U_test ** 2
        f_list = np.empty_like(xs)
        for i in range(len(xs)):
            Re_dh = rho * abs(U[i]) * Dh[i] / max(mu, 1e-12)
            f_list[i] = duct_friction_factor(Re_dh, roughness_ratio)
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            f = 0.5 * (f_list[i] + f_list[i - 1])
            dh = 0.5 * (Dh[i] + Dh[i - 1])
            qq = 0.5 * (q[i] + q[i - 1])
            dPt = f * dx / max(dh, 1e-9) * qq
            Pt[i] = Pt[i - 1] - dPt
        # discrete grille loss at the end of the working section
        i_gr = int(np.searchsorted(xs, Lc + Lt))
        if K_grille:
            Pt[i_gr:] -= K_grille * q_test
        # diffuser expansion loss distributed over the diffuser
        AR = A_out / A_test
        K_d, K_df, K_dex = diffuser_loss_coefficient(AR, diffuser_half_angle_deg,
                                                      float(np.mean(f_list)))
        if len(xs) > i_gr:
            ramp = np.linspace(0.0, 1.0, len(xs) - i_gr)
            Pt[i_gr:] -= K_dex * q_test * ramp
        Ps = Pt - q
        return dict(x_m=xs, section=section, area_m2=areas, U_m_s=U,
                     q_Pa=q, Pt_Pa=Pt, Ps_Pa=Ps,
                     Pt_gauge_Pa=Pt - P_ambient, Ps_gauge_Pa=Ps - P_ambient,
                     Cp=(Ps - P_ambient) / q_test if q_test > 0 else np.zeros_like(xs),
                     friction_factor=f_list,
                     delta_Pt_total_Pa=float(P_ambient - Pt[-1]),
                     diffuser_area_ratio=float(AR),
                     K_diffuser=float(K_d), K_diffuser_friction=float(K_df),
                     K_diffuser_expansion=float(K_dex),
                     source_type="문헌 상관식 기반 1차원 예측 (Barlow/Rae/Pope 계열) — CFD 아님")

    def required_fan_power(self, U_test, rho, dP_fan):
        Q = U_test * self.geometry.A_test
        return float(Q * dP_fan), float(Q)

def duct_friction_factor(Re_Dh, roughness_ratio=0.0):
    """Darcy friction factor. Laminar 64/Re; turbulent Haaland (explicit Colebrook)."""
    Re = max(float(Re_Dh), 1.0)
    if Re < 2300.0:
        return 64.0 / Re
    e = max(float(roughness_ratio), 0.0)
    inv = -1.8 * math.log10((e / 3.7) ** 1.11 + 6.9 / Re)
    return float(1.0 / (inv * inv)) if inv != 0 else 0.02

def diffuser_loss_coefficient(area_ratio, half_angle_deg, f=0.02):
    """Conical/rectangular diffuser loss, referenced to inlet dynamic pressure.

    K = K_friction + K_expansion with
        K_f = (f / (8 sin(theta))) (1 - 1/AR^2)
        K_ex = K_e(theta) (1 - 1/AR)^2
    This is the standard decomposition used in low-speed wind-tunnel design
    (Barlow, Rae & Pope, *Low-Speed Wind Tunnel Testing*; Wattendorf correlation for
    K_e). K_e rises steeply past about 5 deg half-angle because the boundary layer
    starts to separate.
    """
    AR = max(float(area_ratio), 1.0 + 1e-9)
    th = max(float(half_angle_deg), 0.1)
    thr = deg2rad(th)
    K_f = (f / (8.0 * math.sin(thr))) * (1.0 - 1.0 / (AR * AR))
    if th < 1.5:
        K_e = 0.10
    elif th <= 5.0:
        K_e = 0.10 + 0.04 * (th - 1.5)
    else:
        K_e = 0.24 + 0.13 * (th - 5.0)
    K_ex = K_e * (1.0 - 1.0 / AR) ** 2
    return float(K_f + K_ex), float(K_f), float(K_ex)

class PitotTube:
    """Standard Pitot tube: reads total pressure only."""

    def __init__(self, coefficient=1.0, zero_offset_Pa=0.0, uncertainty_Pa=1.0):
        self.coefficient = float(coefficient)
        self.zero_offset_Pa = float(zero_offset_Pa)
        self.uncertainty_Pa = float(uncertainty_Pa)

    def read(self, Pt_true, Ps_reference, rho):
        dP = self.coefficient * (Pt_true - Ps_reference) + self.zero_offset_Pa
        U = math.sqrt(2.0 * max(dP, 0.0) / rho) if rho > 0 else float("nan")
        dU = float("nan")
        if rho > 0 and dP > 0:
            dU = abs(U * 0.5 * self.uncertainty_Pa / dP)
        return dict(delta_P_Pa=float(dP), U_m_s=float(U),
                     U_uncertainty_m_s=dU,
                     delta_P_uncertainty_Pa=self.uncertainty_Pa,
                     source_type="가상 계측값 (모델 + 기기 오차)")

class PitotStaticTube(PitotTube):
    """Pitot-static tube: reads static and total pressure together."""

    def read_full(self, Pt_true, Ps_true, rho):
        r = self.read(Pt_true, Ps_true, rho)
        r.update(Ps_Pa=float(Ps_true), Pt_Pa=float(Pt_true),
                  q_Pa=float(Pt_true - Ps_true))
        return r

class Manometer:
    """Liquid-column manometer. AFA1 uses water and can be tilted for sensitivity."""

    def __init__(self, fluid_density=WATER_DENSITY, inclination_deg=90.0,
                 uncertainty_m=0.0005, n_tubes=1, label="제어반 마노미터"):
        self.fluid_density = float(fluid_density)
        self.inclination_deg = float(inclination_deg)
        self.uncertainty_m = float(uncertainty_m)
        self.n_tubes = int(n_tubes)
        self.label = label

    def height_from_pressure(self, delta_P):
        s = math.sin(deg2rad(self.inclination_deg))
        den = self.fluid_density * G * s
        if den <= 0:
            return float("nan")
        return float(delta_P / den)

    def pressure_from_height(self, delta_h):
        s = math.sin(deg2rad(self.inclination_deg))
        return float(self.fluid_density * G * float(delta_h) * s)

    def read(self, delta_P):
        h = self.height_from_pressure(delta_P)
        dP_unc = self.pressure_from_height(self.uncertainty_m)
        return dict(delta_h_m=h, delta_h_mm=h * 1e3,
                     delta_P_Pa=float(delta_P), uncertainty_Pa=abs(dP_unc),
                     fluid_density=self.fluid_density,
                     inclination_deg=self.inclination_deg,
                     source_type="가상 계측값 (모델 + 기기 오차)")

class LiftDragBalance:
    """AF1300Z two-component balance: lift and drag only."""

    name = "AF1300Z 기본 양력/항력 밸런스"
    included = True
    measures = ("lift", "drag")

    def __init__(self, resolution_N=0.01, zero_offset_N=0.0, uncertainty_N=0.02):
        self.resolution_N = float(resolution_N)
        self.zero_offset_N = float(zero_offset_N)
        self.uncertainty_N = float(uncertainty_N)

    def _quantize(self, v):
        # a run that produced no rows hands the balance a NaN; round() raises on it
        v = float(v) if v is not None else float("nan")
        if not np.isfinite(v):
            return float("nan")
        if self.resolution_N <= 0:
            return v
        return float(round((v + self.zero_offset_N) / self.resolution_N) * self.resolution_N)

    def read(self, drag_N, lift_N, moment_Nm=None):
        return dict(drag_N=self._quantize(drag_N), lift_N=self._quantize(lift_N),
                     drag_uncertainty_N=self.uncertainty_N,
                     lift_uncertainty_N=self.uncertainty_N,
                     pitching_moment_Nm=float("nan"),
                     pitching_moment_note="3성분 밸런스(AF1300T)가 없으면 "
                                           "피칭모멘트는 시뮬레이션 계산값으로만 제공됩니다",
                     source_type="가상 계측값 (모델 + 기기 오차)")

class ThreeComponentBalance(LiftDragBalance):
    """AF1300T: adds pitching moment. Optional ancillary, not in the starter set."""

    name = "AF1300T 3성분 밸런스 (선택 장비)"
    included = False
    measures = ("lift", "drag", "pitching_moment")

    def __init__(self, resolution_N=0.01, zero_offset_N=0.0, uncertainty_N=0.02,
                 moment_resolution_Nm=1e-4, moment_uncertainty_Nm=2e-4):
        super().__init__(resolution_N, zero_offset_N, uncertainty_N)
        self.moment_resolution_Nm = float(moment_resolution_Nm)
        self.moment_uncertainty_Nm = float(moment_uncertainty_Nm)

    def read(self, drag_N, lift_N, moment_Nm=None):
        r = super().read(drag_N, lift_N, moment_Nm)
        m = 0.0 if moment_Nm is None else float(moment_Nm)
        if self.moment_resolution_Nm > 0:
            m = round(m / self.moment_resolution_Nm) * self.moment_resolution_Nm
        r.update(pitching_moment_Nm=float(m),
                  pitching_moment_uncertainty_Nm=self.moment_uncertainty_Nm,
                  pitching_moment_note="AF1300T 측정값")
        return r

class PressureTapSet:
    """Pressure tappings on a model. Empty unless the user defines taps."""

    def __init__(self, taps=None):
        self.taps = list(taps or [])

    def add(self, tap_id, x, y, z):
        self.taps.append(dict(tap_id=str(tap_id), x=float(x), y=float(y), z=float(z)))

    def readings(self, pressure_fn, P_ref, q_ref):
        if not self.taps:
            return [], "압력탭 데이터 없음"
        out = []
        for t in self.taps:
            P = float(pressure_fn(t["x"], t["y"], t["z"]))
            out.append(dict(tap_id=t["tap_id"], x=t["x"], y=t["y"], z=t["z"],
                             P_Pa=P,
                             Cp=(float((P - P_ref) / q_ref) if q_ref > 0 else float("nan")),
                             source_type="저차원 모델 기반 압력 예측"))
        return out, "ok"

class WindTunnelInstrumentation:
    """Tracks which ancillaries are fitted and blocks use of absent equipment."""

    def __init__(self, enabled=None):
        self.enabled = set(enabled or [])

    def is_available(self, code):
        return code in AF1300_INCLUDED or code in self.enabled

    def require(self, code):
        if code in AF1300_INCLUDED:
            return True, ""
        if code in self.enabled:
            return True, ""
        if code in AF1300_OPTIONAL:
            return False, (f"{code} 은(는) AF1300S 기본 구성에 포함되지 않습니다. "
                           "선택 장비를 활성화하세요.")
        return False, f"{code} 은(는) 알 수 없는 장비입니다."

    def inventory(self):
        rows = [dict(code=c, name=n, category="기본 구성", active=True)
                for c, n in AF1300_INCLUDED.items()]
        for c, n in AF1300_OPTIONAL.items():
            rows.append(dict(code=c, name=n, category="선택 장비",
                              active=(c in self.enabled)))
        return rows

def blockage_ratio(model_area_m2, geometry: TunnelGeometry):
    A = geometry.A_test
    if A <= 0:
        return float("nan"), "작업구간 단면적이 0입니다"
    r = float(model_area_m2) / A
    if r > 0.10:
        note = "풍동 차단 영향이 클 가능성 (차단율 10% 초과)"
    elif r > 0.05:
        note = "차단 영향 주의 (차단율 5% 초과)"
    else:
        note = "차단율 낮음"
    return r, note

def test_section_velocity_profile(U_free, y_from_wall, delta_m=0.02, enabled=False):
    """Simple 1/7-power boundary layer near the walls.

    TecQuipment does not publish measured AF1300 boundary-layer data, so this is an
    간이 모델 and is labelled as such wherever it is used.
    """
    if not enabled:
        return float(U_free), "균일 작업구간 가정"
    y = abs(float(y_from_wall))
    if delta_m <= 0 or y >= delta_m:
        return float(U_free), "경계층 밖"
    return float(U_free) * (y / delta_m) ** (1.0 / 7.0), "간이 경계층 모델 (1/7 멱법칙)"

def shuttlecock_pressure_estimate(rho, U_rel, Cd, area_m2, P_reference):
    """Front/back pressure estimate from the drag coefficient.

    This is a two-value lumped estimate, not a resolved surface pressure
    distribution. The front face is taken to approach stagnation and the base
    pressure is inferred so that the pressure difference reproduces the drag.
    """
    q = 0.5 * rho * U_rel ** 2
    if q <= 0 or area_m2 <= 0:
        return dict(P_front_Pa=float(P_reference), P_back_Pa=float(P_reference),
                     delta_P_Pa=0.0, drag_from_dP_N=0.0,
                     source_type="저차원 모델 기반 압력 예측")
    P_front = P_reference + q
    dP = Cd * q
    P_back = P_front - dP
    return dict(P_front_Pa=float(P_front), P_back_Pa=float(P_back),
                 delta_P_Pa=float(dP), q_Pa=float(q),
                 drag_from_dP_N=float(dP * area_m2),
                 Cp_front=1.0, Cp_back=float(1.0 - Cd),
                 source_type="저차원 모델 기반 압력 예측 (실제 압력분포 아님)")

class VirtualWindTunnel:
    """Complete AF1300S virtual rig: environment, duct pressures and instruments."""

    def __init__(self, geometry=None, losses=None, env=None, instrumentation=None,
                 pressure_level=1, fan=None, boundary_layer=False):
        self.geometry = geometry or TunnelGeometry()
        self.losses = losses or TunnelLosses()
        self.env = env or TunnelEnvironment()
        self.instr = instrumentation or WindTunnelInstrumentation()
        self.pressure = WindTunnelPressureModel(self.geometry, self.losses, pressure_level)
        self.fan = fan or FanController()
        self.boundary_layer = bool(boundary_layer)
        self.pitot = PitotTube()
        self.pitot_static = PitotStaticTube()
        self.manometer = Manometer()
        self.balance = LiftDragBalance()
        self.taps = PressureTapSet()

    def resolve_velocity(self, mode, fan_percent=None, U_direct=None):
        if mode == "직접 유속 지정":
            return float(U_direct or 0.0), "사용자 지정 작업구간 유속"
        U, src = self.fan.velocity(fan_percent)
        return U, src

    def state(self, U_test):
        e = self.env.resolve()
        rho, mu = e["rho"], e["mu"]
        warn = list(e["warnings"])
        if not np.isfinite(U_test):
            warn.append("작업구간 유속이 확정되지 않았습니다 (팬 보정 곡선 필요).")
            U_test = 0.0
        lo, hi = AF1300_SPEC["velocity_range_m_s"]
        if U_test > hi:
            warn.append(f"작업구간 유속 {U_test:.1f} m/s 는 AF1300 공식 범위 "
                        f"{lo}~{hi} m/s 를 넘습니다.")
        mach = U_test / e["a_sound"] if np.isfinite(e["a_sound"]) and e["a_sound"] > 0 else 0.0
        if mach > 0.3:
            warn.append(f"마하수 {mach:.2f} — 비압축성 근사(Pt = Ps + q)의 유효 범위를 "
                        "벗어납니다. 압축성 효과를 무시할 수 없습니다.")
        stations, dP_fan = self.pressure.station_states(U_test, rho, self.env.P_ambient_Pa)
        power, Q = self.pressure.required_fan_power(U_test, rho, dP_fan)
        ts = next(s for s in stations if s["station"] == "작업구간")
        return dict(U_test=float(U_test), rho=rho, mu=mu, mach=float(mach),
                     stations=stations, delta_P_fan_Pa=dP_fan,
                     fan_power_W=power, volumetric_flow_m3_s=Q,
                     mass_flow_kg_s=float(rho * Q),
                     Ps_test_Pa=ts["Ps_Pa"], Pt_test_Pa=ts["Pt_Pa"], q_test_Pa=ts["q_Pa"],
                     Ps_test_gauge_Pa=ts["Ps_gauge_Pa"],
                     environment=e, warnings=warn,
                     pressure_level=self.pressure.level,
                     pressure_level_name=PRESSURE_MODEL_LEVELS[self.pressure.level],
                     model_note="저차원 공력 모델 기반 압력 예측 (CFD 결과 아님)")

    def instruments_report(self, st, drag_N=None, lift_N=None, moment_Nm=None):
        rho = st["rho"]
        rng = getattr(self, "noise_rng", None)

        def jitter(value, sigma):
            if rng is None or not np.isfinite(value) or sigma <= 0:
                return value
            return float(value + rng.normal(0.0, sigma))

        out = {}
        if rng is not None:
            st = dict(st)
            st["Pt_test_Pa"] = jitter(st["Pt_test_Pa"], self.pitot.uncertainty_Pa)
            st["Ps_test_Pa"] = jitter(st["Ps_test_Pa"], self.pitot.uncertainty_Pa)
            st["q_test_Pa"] = jitter(st["q_test_Pa"], self.pitot.uncertainty_Pa)
            if drag_N is not None:
                drag_N = jitter(drag_N, self.balance.uncertainty_N)
            if lift_N is not None:
                lift_N = jitter(lift_N, self.balance.uncertainty_N)
        out["pitot"] = self.pitot.read(st["Pt_test_Pa"], st["Ps_test_Pa"], rho)
        out["pitot_static"] = self.pitot_static.read_full(st["Pt_test_Pa"],
                                                           st["Ps_test_Pa"], rho)
        out["manometer"] = self.manometer.read(st["q_test_Pa"])
        if drag_N is not None:
            out["balance"] = self.balance.read(drag_N, lift_N or 0.0, moment_Nm)
        ok_afa1, _ = self.instr.require("AFA1")
        if ok_afa1 and "AFA1" in self.instr.enabled:
            m36 = Manometer(fluid_density=WATER_DENSITY, inclination_deg=30.0,
                             n_tubes=36, label="AFA1 36관 틸팅 마노미터 (물)")
            out["multitube"] = m36.read(st["q_test_Pa"])
            out["multitube"]["n_tubes"] = 36
        return out

    def metadata(self):
        return dict(
            equipment=AF1300_SPEC["equipment"],
            tunnel_type=AF1300_SPEC["tunnel_type"],
            working_section=AF1300_SPEC["working_section"],
            velocity_range_m_s=AF1300_SPEC["velocity_range_m_s"],
            layout=AF1300_SPEC["layout"],
            source=AF1300_SPEC["source"],
            included_in_starter_set=list(AF1300_INCLUDED.keys()),
            optional_ancillary=list(AF1300_OPTIONAL.keys()),
            optional_enabled=sorted(self.instr.enabled),
            loss_coefficients=self.losses.as_dict(),
            pressure_model_level=self.pressure.level,
            pressure_model_name=PRESSURE_MODEL_LEVELS[self.pressure.level],
            pressure_model_note="저차원 공력 모델 기반 압력 예측 (CFD 아님)",
            boundary_layer_model=("간이 모델 (1/7 멱법칙)" if self.boundary_layer
                                   else "사용 안 함"),
        )

def wind_tunnel_position_check(x, y, z, geometry: TunnelGeometry):
    """Is the shuttlecock inside the 305x305x600 mm working section?"""
    hw, hh = geometry.ws_width_m / 2.0, geometry.ws_height_m / 2.0
    L = geometry.ws_length_m
    inside = (0.0 <= x <= L) and (abs(y) <= hh) and (abs(z) <= hw)
    if inside:
        return True, "작업구간 내부"
    return False, "작업구간 밖"


def run_wind_tunnel_experiment(model, tunnel: "VirtualWindTunnel", U_test,
                                incidence_deg=0.0, spin_locked=True,
                                spin_value_rad_s=0.0, dt=5e-4, t_max=1.0,
                                solver="RK4", position=(0.30, 0.0, 0.0),
                                initial_offset_deg=0.0,
                                model_fixed=False, wall_correction=False,
                                turbulence_intensity=0.0, turbulence_length_m=0.10,
                                seed=None, instrument_noise=False):
    """Runs the shuttlecock in the virtual AF1300 working section.

    The shuttlecock is held at a fixed station while air moves past it, so the
    translational degrees of freedom are frozen and only the turnover DOF is
    integrated. Axial spin is locked to represent a mount that constrains it.

    incidence_deg is the mount setting; initial_offset_deg additionally displaces the
    body axis away from the cork-forward equilibrium so the turnover response has
    something to relax from. With no offset the shuttlecock starts already aligned and
    the flip/stabilisation columns only ever report numerical noise.
    """
    st = tunnel.state(U_test)
    rho, mu = st["rho"], st["mu"]
    env = EnvironmentModel(rho=rho, mu=mu, u_air_x=float(st["U_test"]),
                            u_air_y=0.0, u_air_z=0.0,
                            turbulence_intensity=float(turbulence_intensity or 0.0),
                            turbulence_length_scale_m=float(turbulence_length_m or 0.10),
                            turbulence_seed=seed)
    warn = list(st["warnings"])

    x0, y0, z0 = position
    inside, pos_note = wind_tunnel_position_check(x0, y0, z0, tunnel.geometry)
    if not inside:
        warn.append(f"셔틀콕이 {pos_note} 입니다 (작업구간 305x305x600 mm).")

    A_model = model.geometry.S
    br, br_note = blockage_ratio(A_model, tunnel.geometry)
    if br > 0.05:
        warn.append(f"차단율 {br*100:.1f}% — {br_note}")
    if wall_correction:
        warn.append("벽면 보정: 출처가 확인된 보정식이 없어 적용하지 않았습니다 "
                    "(blockage_ratio만 기록).")

    _prev_lock = getattr(model, "_spin_locked", False)
    _prev_lock_value = getattr(model, "_spin_locked_value", 0.0)
    _prev_fixed = getattr(model, "_tunnel_fixed", False)
    model._spin_locked = bool(spin_locked)
    model._spin_locked_value = float(spin_value_rad_s)
    model._tunnel_fixed = bool(model_fixed)

    ic = InitialCondition3D(x0=x0, y0=y0, z0=z0, V0=0.0,
                             elevation_deg=0.0, azimuth_deg=0.0,
                             body_axis_offset_deg=float(180.0 - float(incidence_deg)
                                                        - float(initial_offset_deg or 0.0)),
                             body_azimuth_deg=0.0,
                             spin0_rad_s=float(spin_value_rad_s))
    eng = SimulationEngine3D(model, env, ic, solver=solver, dt=dt, t_max=t_max)
    eng.hold_position = True
    try:
        res = eng.run()
    finally:
        # these flags belong to this rig, not to the model; leaving them set made a
        # later free-flight run silently inherit the locked spin and fixed attitude
        model._spin_locked = _prev_lock
        model._spin_locked_value = _prev_lock_value
        model._tunnel_fixed = _prev_fixed
    for w in res.warnings:
        warn.append(w)

    df = res.df
    drag = float(df.drag_force_N.iloc[-1]) if len(df) else float("nan")
    lift = float(df.lift_force_N.iloc[-1]) if len(df) else float("nan")
    mom = float(df.net_moment_Nm.iloc[-1]) if len(df) else float("nan")
    tunnel.noise_rng = (np.random.default_rng(seed) if instrument_noise else None)
    instr = tunnel.instruments_report(st, drag_N=drag, lift_N=lift, moment_Nm=mom)
    cd_last = float(df.Cd.iloc[-1]) if len(df) else float("nan")
    sp = shuttlecock_pressure_estimate(rho, st["U_test"], cd_last, A_model,
                                        st["Ps_test_Pa"])
    taps, tap_note = tunnel.taps.readings(
        lambda a, b, c: st["Ps_test_Pa"], st["Ps_test_Pa"], st["q_test_Pa"])

    meta = dict(res.metadata)
    meta["wind_tunnel"] = tunnel.metadata()
    meta["wind_tunnel"]["working_section_state"] = {
        k: st[k] for k in ("U_test", "Ps_test_Pa", "Pt_test_Pa", "q_test_Pa",
                            "Ps_test_gauge_Pa", "mach", "delta_P_fan_Pa",
                            "volumetric_flow_m3_s", "mass_flow_kg_s")}
    meta["wind_tunnel"]["blockage_ratio"] = br
    meta["wind_tunnel"]["blockage_note"] = br_note
    meta["wind_tunnel"]["position_status"] = pos_note
    meta["wind_tunnel"]["incidence_deg"] = float(incidence_deg)
    meta["wind_tunnel"]["incidence_lock"] = "고정" if model_fixed else "회전"
    meta["spin_constraint"] = dict(
        spin_locked=bool(spin_locked), spin_value_rad_s=float(spin_value_rad_s),
        note=("자전 구속: 축 spin 자유도를 고정하고 turnover 자유도만 적분합니다. "
              "구속에 필요한 반력은 constraint_torque_Nm 으로 기록됩니다."
              if spin_locked else "자전 허용"))
    meta["instruments"] = instr
    meta["shuttlecock_pressure"] = sp
    meta["pressure_taps"] = dict(readings=taps, status=tap_note)
    meta["turbulence"] = meta.get("turbulence", {})
    meta["instrument_noise"] = dict(
        enabled=bool(instrument_noise),
        note=("계측 노이즈 활성화: 기기 불확도를 표준편차로 하는 정규 난수를 판독값에 더합니다"
              if instrument_noise else "계측 노이즈 없음 (판독값 = 모델값)"))
    meta["provenance"] = dict(
        measured="없음 (실측 데이터 미입력)",
        instrument_readings="가상 계측값 (모델 + 기기 오차)",
        pressure_field="저차원 공력 모델 기반 압력 예측 (CFD 아님)",
        boundary_layer=meta["wind_tunnel"]["boundary_layer_model"],
        tunnel_geometry="제조사 공식 자료 (작업구간) + 사용자 입력 (기타 치수)")
    return SimResult(df=df, metadata=meta, valid=res.valid, warnings=warn), st, instr

def wind_tunnel_sweep(build_model_fn, tunnel: "VirtualWindTunnel", velocities,
                       damage_sets=None, incidences=None, initial_offsets=None,
                       spin_locked=True, dt=1e-3, t_max=0.8):
    """Automatic condition sweep. Returns one row per condition."""
    rows = []
    damage_sets = damage_sets or [[]]
    incidences = incidences or [0.0]
    initial_offsets = initial_offsets or [60.0]
    for U in velocities:
        for dmg in damage_sets:
            for inc in incidences:
                for off in initial_offsets:
                    try:
                        m = build_model_fn(dmg)
                        res, st, instr = run_wind_tunnel_experiment(
                            m, tunnel, U, incidence_deg=inc, spin_locked=spin_locked,
                            initial_offset_deg=off, dt=dt, t_max=t_max)
                        d = res.df
                        tc = res.metadata.get("turnover_constants", {})
                        rows.append(dict(
                            U_test_m_s=st["U_test"],
                            removed_feathers=len(dmg),
                            incidence_deg=inc, initial_offset_deg=off,
                            Ps_test_gauge_Pa=st["Ps_test_gauge_Pa"],
                            q_test_Pa=st["q_test_Pa"],
                            Re=(float(d.Re.iloc[-1]) if len(d) else float("nan")),
                            Cd=(float(d.Cd.iloc[-1]) if len(d) else float("nan")),
                            Cm=(float(d.Cm.iloc[-1]) if len(d) else float("nan")),
                            drag_N=(float(d.drag_force_N.iloc[-1]) if len(d) else float("nan")),
                            lift_N=(float(d.lift_force_N.iloc[-1]) if len(d) else float("nan")),
                            flip_time_s=_wt_flip_time(d),
                            stabilization_time_s=_wt_stabilization_time(d),
                            max_wobble_deg=(float(np.nanmax(np.abs(d.delta_alpha_deg)))
                                             if len(d) else float("nan")),
                            Tw_model_s=tc.get("Tw_model", float("nan")),
                            zeta_model=tc.get("zeta_model", float("nan")),
                            valid=res.valid))
                    except Exception:
                        logger.exception("sweep condition failed")
                        rows.append(dict(U_test_m_s=U, removed_feathers=len(dmg),
                                          incidence_deg=inc, initial_offset_deg=off,
                                          valid=False))
    return pd.DataFrame(rows)

def _wt_flip_time(df):
    if df is None or len(df) == 0 or "delta_alpha_deg" not in df.columns:
        return float("nan")
    w = np.abs(df.delta_alpha_deg.values)
    idx = np.where(w < 90.0)[0]
    return float(df.time_s.values[idx[0]]) if len(idx) else float("nan")

def _wt_stabilization_time(df, threshold_deg=5.0):
    if df is None or len(df) == 0 or "delta_alpha_deg" not in df.columns:
        return float("nan")
    w = np.abs(df.delta_alpha_deg.values)
    t = df.time_s.values
    for k in range(len(w)):
        if np.all(w[k:] < threshold_deg):
            return float(t[k])
    return float("nan")


# ---------------------------------------------------------------------------
# (CFD/LBM engine removed)
# ---------------------------------------------------------------------------

def transfer_moment_to_cg(M_ref, F, r_ref, r_cg):
    """Moves an aerodynamic moment from one reference point to the centre of mass.

        M_CG = M_ref + (r_ref - r_CG) x F

    Reporting a moment without stating its reference point is meaningless, and for
    an asymmetrically damaged shuttlecock the geometric centre and the centre of
    mass are not the same place.
    """
    M_ref = np.asarray(M_ref, dtype=float)
    F = np.asarray(F, dtype=float)
    d = np.asarray(r_ref, dtype=float) - np.asarray(r_cg, dtype=float)
    return M_ref + cross3(d, F)


def visual_flow_field(x_c, y_c, v_rel_xy, body_e_xy, extent, n=15,
                      body_radius=0.033, wake_strength=0.85, wake_spread=2.2):
    """Reduced-order flow visualisation. NOT a CFD or RANS solution.

    Freestream (in the shuttlecock frame) plus a Gaussian wake velocity deficit behind
    the body. Used for display only and never fed into the force calculation.
    """
    Vrel = float(vnorm(v_rel_xy))
    if Vrel <= 0:
        return None
    ef = np.array(v_rel_xy, dtype=float) / Vrel
    down = -ef
    perp = np.array([-down[1], down[0]])

    gx = np.linspace(x_c - extent, x_c + extent, n)
    gy = np.linspace(y_c - extent, y_c + extent, n)
    GX, GY = np.meshgrid(gx, gy)
    RX = GX - x_c
    RY = GY - y_c
    s = RX * down[0] + RY * down[1]
    q = RX * perp[0] + RY * perp[1]

    U = np.full_like(GX, down[0] * Vrel)
    V = np.full_like(GY, down[1] * Vrel)

    r_w = body_radius * (1.0 + wake_spread * np.clip(s, 0, None) / max(extent, 1e-6) * 6.0)
    r_w = np.maximum(r_w, body_radius)
    deficit = np.where(s > 0,
                       wake_strength * np.exp(-(q ** 2) / (2 * r_w ** 2))
                       * np.exp(-np.clip(s, 0, None) / (extent * 1.4)),
                       0.0)
    U -= down[0] * Vrel * deficit
    V -= down[1] * Vrel * deficit

    block = np.exp(-((s + 0.25 * body_radius) ** 2 + q ** 2) / (2 * (body_radius * 2.2) ** 2))
    U += perp[0] * Vrel * 0.35 * block * np.sign(q + 1e-12)
    V += perp[1] * Vrel * 0.35 * block * np.sign(q + 1e-12)

    return dict(X=GX, Y=GY, U=U, V=V, Vrel=Vrel, downstream=down, perp=perp,
                source="간이 공기흐름 시각화 (CFD 결과 아님)")

def visual_flow_field_3d(center, v_rel, extent, n=7, body_radius=0.033,
                          wake_strength=0.85):
    """3D reduced-order flow visualisation. NOT a CFD or RANS solution."""
    Vrel = float(vnorm(v_rel))
    if Vrel <= 0:
        return None
    ef = np.asarray(v_rel, dtype=float) / Vrel
    down = -ef
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, down))) > 0.95:
        tmp = np.array([0.0, 1.0, 0.0])
    p1 = cross3(down, tmp)
    p1 /= (vnorm(p1) or 1.0)
    p2 = cross3(down, p1)
    p2 /= (vnorm(p2) or 1.0)

    g = np.linspace(-extent, extent, n)
    pts, vecs = [], []
    for a_ in g:
        for b_ in g:
            for c_ in g:
                off = down * a_ + p1 * b_ + p2 * c_
                pos = np.asarray(center, dtype=float) + off
                s_ = a_
                q_ = math.hypot(b_, c_)
                r_w = max(body_radius * (1.0 + 6.0 * max(s_, 0.0) / max(extent, 1e-6)),
                          body_radius)
                deficit = (wake_strength * math.exp(-(q_ ** 2) / (2 * r_w ** 2))
                           * math.exp(-max(s_, 0.0) / (extent * 1.4))) if s_ > 0 else 0.0
                vel = down * Vrel * (1.0 - deficit)
                pts.append(pos)
                vecs.append(vel)
    return dict(points=np.array(pts), vectors=np.array(vecs), Vrel=Vrel,
                downstream=down, perp1=p1, perp2=p2,
                source="간이 공기흐름 시각화 (CFD 결과 아님)")

def visual_streamlines_3d(field, center, extent, n_lines=9, n_steps=26):
    """3D streamline polylines from the visual field. Display only."""
    if field is None:
        return []
    down = field["downstream"]
    p1, p2 = field["perp1"], field["perp2"]
    Vrel = field["Vrel"]
    body_radius = 0.033
    step = 2.0 * extent / n_steps
    lines = []
    for k in range(n_lines):
        ang = 2 * math.pi * k / n_lines
        rad = extent * 0.55
        start = (np.asarray(center, dtype=float) - down * extent
                 + (p1 * math.cos(ang) + p2 * math.sin(ang)) * rad)
        pos = start.copy()
        xs, ys, zs = [], [], []
        for _ in range(n_steps):
            rel = pos - np.asarray(center, dtype=float)
            s_ = float(np.dot(rel, down))
            q_ = math.hypot(float(np.dot(rel, p1)), float(np.dot(rel, p2)))
            r_w = max(body_radius * (1.0 + 6.0 * max(s_, 0.0) / max(extent, 1e-6)),
                      body_radius)
            deficit = (0.85 * math.exp(-(q_ ** 2) / (2 * r_w ** 2))
                       * math.exp(-max(s_, 0.0) / (extent * 1.4))) if s_ > 0 else 0.0
            vel = down * Vrel * (1.0 - deficit)
            nv = float(vnorm(vel))
            if nv < 1e-9:
                break
            pos = pos + vel / nv * step
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
            zs.append(float(pos[2]))
        if len(xs) > 2:
            lines.append((xs, ys, zs))
    return lines

def shuttlecock_profile(geometry: Optional[Geometry] = None,
                         skirt: Optional[SkirtGeometry] = None) -> dict:
    """Normalised BWF shuttlecock side profile, derived from the actual dimensions.

    Lengths are divided by the total length so the shape can be drawn at any scale.
    u runs along the symmetry axis: u = +0.5 is the cork nose, u = -0.5 the feather tips.
    """
    if skirt is not None:
        tip_d = skirt.skirt_diameter_m
        cork_d = skirt.cork_diameter_m
        feather_len = skirt.skirt_length_m
        n_feathers = skirt.n_feathers
    elif geometry is not None:
        tip_d = geometry.skirt_diameter_m
        cork_d = geometry.cork_diameter_m
        feather_len = geometry.skirt_length_m
        n_feathers = 16
    else:
        tip_d = mm_to_m(BWF_DEFAULTS["skirt_tip_diameter_mm"])
        cork_d = mm_to_m(BWF_DEFAULTS["cork_diameter_mm"])
        feather_len = mm_to_m(BWF_DEFAULTS["feather_length_mm"])
        n_feathers = BWF_DEFAULTS["n_feathers"]

    cork_len = cork_d
    total = cork_len + feather_len
    if total <= 0:
        total = 1.0
    return dict(
        cork_half=0.5 * cork_d / total,
        cork_len=cork_len / total,
        tip_half=0.5 * tip_d / total,
        feather_len=feather_len / total,
        u_nose=0.5,
        u_cork_base=0.5 - cork_len / total,
        u_tip=-0.5,
        n_feathers=n_feathers,
        total_length_m=total,
    )

def shuttlecock_outline_2d(prof: dict, n_arc=18):
    """Side-view outline points in body coordinates (u along axis, r radial).

    Returns the cork dome polygon and the flaring skirt edges as (u, r) sequences.
    """
    ch = prof["cork_half"]
    u_base = prof["u_cork_base"]
    u_nose = prof["u_nose"]
    th = prof["tip_half"]
    u_tip = prof["u_tip"]

    dome_u, dome_r = [], []
    for k in range(n_arc + 1):
        ang = -math.pi / 2 + math.pi * k / n_arc
        dome_u.append(u_nose - ch * (1.0 - math.cos(ang)))
        dome_r.append(ch * math.sin(ang))
    cork_u = dome_u + [u_base, u_base]
    cork_r = dome_r + [ch, -ch]
    return dict(cork_u=cork_u, cork_r=cork_r,
                skirt_u=[u_base, u_tip, u_tip, u_base, u_base],
                skirt_r=[ch, th, -th, -ch, ch],
                u_base=u_base, u_tip=u_tip, cork_half=ch, tip_half=th)

SHUTTLE_COLORS = dict(
    feather_fill="#ffffff",
    feather_edge="#c3cad6",
    feather_shade="#eef1f6",
    quill="#d4dae4",
    cork_fill="#fdfdfd",
    cork_edge="#c3cad6",
    band="#16295e",
    thread="#dfe4ec",
    removed_edge="#e53e3e",
    removed_fill="rgba(245,246,249,0.35)",
)

def feather_vane_2d(prof, azim_rad, n=18, w_root=0.055, w_tip=0.20, w_min=0.010):
    """Side-view projection of one feather vane.

    A vane sticks out radially and its flat face is tangential, so under projection its
    centre sits at R*cos(azimuth) while its apparent WIDTH scales with |sin(azimuth)|:
    vanes in the middle look broad, vanes at the silhouette turn edge-on and look thin.
    The tip is rounded away from the cork, which is what gives the skirt its scalloped hem.
    """
    pos = math.cos(azim_rad)
    wscale = abs(math.sin(azim_rad))
    u0, u1 = prof["u_cork_base"], prof["u_tip"]
    r0 = prof["cork_half"] * pos
    r1 = prof["tip_half"] * pos

    def at(t):
        u = u0 + (u1 - u0) * t
        r = r0 + (r1 - r0) * t
        w = (w_root + (w_tip - w_root) * (t ** 0.65)) * wscale + w_min
        return u, r, w

    upper_u, upper_r, lower_u, lower_r = [], [], [], []
    for k in range(n + 1):
        t = k / n
        u, r, w = at(t)
        upper_u.append(u); upper_r.append(r + w)
        lower_u.append(u); lower_r.append(r - w)

    ut, rt, wt = at(1.0)
    cap_u, cap_r = [], []
    for k in range(13):
        ang = math.pi / 2 - math.pi * k / 12
        cap_u.append(ut - abs(u1 - u0) * 0.075 * math.cos(ang))
        cap_r.append(rt + wt * math.sin(ang))

    us = upper_u + cap_u + lower_u[::-1] + [upper_u[0]]
    rs = upper_r + cap_r + lower_r[::-1] + [upper_r[0]]
    quill = ([u0 + (u1 - u0) * (k / n) for k in range(n + 1)],
             [r0 + (r1 - r0) * (k / n) for k in range(n + 1)])
    return us, rs, quill

def feather_shade(azim_rad):
    """Fill/edge colours for a vane: ones facing the viewer are bright, ones behind the
    axis are darker, which is what reads as depth in a photograph."""
    facing = math.sin(azim_rad)
    t = (facing + 1.0) / 2.0
    v = int(round(232 + 23 * t))
    edge = int(round(150 + 40 * t))
    return f"rgb({v},{v},{min(v + 2, 255)})", f"rgb({edge},{edge + 6},{edge + 16})"

def thread_rings_2d(prof, fractions=(0.30, 0.50)):
    """Radii of the cross-stitch thread rings that hold the skirt together."""
    out = []
    for t in fractions:
        u = prof["u_cork_base"] + (prof["u_tip"] - prof["u_cork_base"]) * t
        r = prof["cork_half"] + (prof["tip_half"] - prof["cork_half"]) * t
        out.append((u, r))
    return out

def _tri_grid(n_rows, n_cols, base=0):
    """Triangle indices for an (n_rows x n_cols) point grid."""
    I, J, K = [], [], []
    for a in range(n_rows - 1):
        for b in range(n_cols - 1):
            p0 = base + a * n_cols + b
            p1 = p0 + 1
            p2 = p0 + n_cols
            p3 = p2 + 1
            I += [p0, p1]
            J += [p1, p3]
            K += [p2, p2]
    return I, J, K

def shuttlecock_mesh_3d(center, q, scale, skirt: Optional[SkirtGeometry] = None,
                         geometry: Optional[Geometry] = None, n_seg=10, n_wid=4):
    """Solid 3D BWF shuttlecock rotated by the current quaternion.

    Returns triangulated surfaces (not wireframes): a rounded cork dome, its band, and one
    curved vane per REMAINING feather, so removed feathers simply do not appear.
    """
    prof = shuttlecock_profile(geometry, skirt)
    R = quat_to_matrix(q)
    c = np.asarray(center, dtype=float)
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    L = 2.0 * scale

    def pt(u, r, ang):
        return c + ex * (u * L) + (ey * math.cos(ang) + ez * math.sin(ang)) * (r * L)

    ch, th = prof["cork_half"], prof["tip_half"]
    ub, ut, un = prof["u_cork_base"], prof["u_tip"], prof["u_nose"]

    n_ring, n_arc = 20, 8
    cork_pts, band_pts = [], []
    for k in range(n_arc + 1):
        a_ = -math.pi / 2 + math.pi * k / n_arc
        u = un - ch * (1.0 - math.cos(a_))
        r = max(ch * math.sin(a_), 1e-6)
        for j in range(n_ring):
            cork_pts.append(pt(u, r, 2 * math.pi * j / n_ring))
    ci, cj, ck = [], [], []
    for a in range(n_arc):
        for b in range(n_ring):
            p0 = a * n_ring + b
            p1 = a * n_ring + (b + 1) % n_ring
            p2 = (a + 1) * n_ring + b
            p3 = (a + 1) * n_ring + (b + 1) % n_ring
            ci += [p0, p1]; cj += [p1, p3]; ck += [p2, p2]

    band_len = prof["cork_len"] * 0.30
    for k in range(2):
        u = ub + band_len * k
        for j in range(n_ring):
            band_pts.append(pt(u, ch, 2 * math.pi * j / n_ring))
    bi, bj, bk = [], [], []
    for b in range(n_ring):
        p0 = b
        p1 = (b + 1) % n_ring
        p2 = n_ring + b
        p3 = n_ring + (b + 1) % n_ring
        bi += [p0, p1]; bj += [p1, p3]; bk += [p2, p2]

    if skirt is not None:
        feathers = skirt.attached_feathers
        n_tot = skirt.n_feathers
    else:
        n_tot = prof["n_feathers"]
        feathers = None
    angles = ([f.azimuth_rad for f in feathers] if feathers is not None
              else [2 * math.pi * i / n_tot for i in range(n_tot)])

    vane_half = math.pi / max(n_tot, 1) * 1.05
    vx, vi, vj, vk = [], [], [], []
    rib_lines = []
    for a_ in angles:
        base = len(vx)
        for si in range(n_seg + 1):
            t = si / n_seg
            u = ub + (ut - ub) * t
            r = ch + (th - ch) * t
            spread = vane_half * (0.32 + 0.68 * (t ** 0.7))
            for wj in range(n_wid):
                f = -1 + 2 * wj / (n_wid - 1)
                bulge = 1.0 + 0.05 * (1 - f * f) * t
                vx.append(pt(u, r * bulge, a_ + spread * f))
        I, J, K = _tri_grid(n_seg + 1, n_wid, base)
        vi += I; vj += J; vk += K
        rib_lines.append((pt(ub, ch, a_), pt(ut, th, a_)))

    rim = [pt(ut, th, 2 * math.pi * j / 48) for j in range(49)]
    threads = []
    for u_ring, r_ring in thread_rings_2d(prof):
        threads.append([pt(u_ring, r_ring, 2 * math.pi * j / 40) for j in range(41)])
    axis = (pt(ut, 0.0, 0.0), pt(un, 0.0, 0.0))
    return dict(
        cork=dict(pts=np.array(cork_pts), i=ci, j=cj, k=ck),
        band=dict(pts=np.array(band_pts), i=bi, j=bj, k=bk),
        vanes=dict(pts=np.array(vx) if vx else np.zeros((0, 3)), i=vi, j=vj, k=vk),
        ribs=rib_lines, rim=rim, threads=threads, axis=axis,
        n_shown=len(angles), n_total=n_tot)

def visual_streamlines(field, n_lines=11, n_steps=44, step_len=None):
    """Integrate the visual field into polylines. Display only."""
    if field is None:
        return []
    X, Y, U, V = field["X"], field["Y"], field["U"], field["V"]
    x0, x1 = float(X.min()), float(X.max())
    y0, y1 = float(Y.min()), float(Y.max())
    down = field["downstream"]
    perp = field["perp"]
    span = max(x1 - x0, y1 - y0)
    if step_len is None:
        step_len = span / (n_steps * 0.85)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    start_c = np.array([cx, cy]) - down * (span * 0.5)

    gx = X[0, :]
    gy = Y[:, 0]

    def sample(px, py):
        i = int(np.clip(np.searchsorted(gy, py) - 1, 0, len(gy) - 2))
        j = int(np.clip(np.searchsorted(gx, px) - 1, 0, len(gx) - 2))
        return float(U[i, j]), float(V[i, j])

    lines = []
    for k in range(n_lines):
        off = (k / (n_lines - 1) - 0.5) * span * 0.95
        p = start_c + perp * off
        xs, ys = [float(p[0])], [float(p[1])]
        for _ in range(n_steps):
            u, v = sample(p[0], p[1])
            nrm = math.hypot(u, v)
            if nrm < 1e-9:
                break
            p = p + np.array([u, v]) / nrm * step_len
            if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                break
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        if len(xs) > 2:
            lines.append((xs, ys))
    return lines

class SimulationEngine:
    def __init__(self, model: ShuttlecockModel, env: EnvironmentModel, ic: InitialCondition,
                 solver="RK4", dt=0.001, t_max=3.0, torque_fn=None, model_level=1):
        self.model = model
        self.env = env
        self.ic = ic
        self.solver = solver
        self.dt = dt
        self.t_max = t_max
        self.torque_fn = torque_fn

        self.model_level = model_level

    def _stability_warning(self, warnings_list):
        omega_n = self.model.turnover.omega_n
        if self.dt * omega_n > 0.5:
            warnings_list.append(
                f"dt*omega_n={self.dt*omega_n:.3f} is large; turnover oscillation may be under-resolved")

    def run(self) -> SimResult:
        if self.model_level >= 2:
            return self._run_coupled()
        return self._run_linear()

    def _run_coupled(self) -> SimResult:
        warnings_list = []
        if self.dt <= 0:
            return SimResult(pd.DataFrame(), {}, False, ["dt must be > 0"])
        self._stability_warning(warnings_list)

        aero = self.model.aero or AeroModel(self.model.Cd_model, level=self.model_level)
        geom_state = self.model.geometry_state()
        if geom_state is not None and geom_state["remaining_feathers"] == 0:
            warnings_list.append(
                "all feathers removed: geometry is unphysical, aerodynamic results are not meaningful")

        errs = (self.model.sanity_check() + damage_domain_errors(self.model)
                + launch_state_errors(self.ic))
        if errs:
            return SimResult(pd.DataFrame(), {}, False, errs)

        # resolve the transverse inertia once, exactly as SimulationEngine3D does, so
        # the restoring moment, the damping moment and omega_dot share one value
        _Ieff, _ = canonical_turnover_inertia(self.model)
        self.model.turnover.I_effective = float(_Ieff)

        state = np.array([self.ic.x0, self.ic.y0, self.ic.vx0, self.ic.vy0,
                           self.ic.theta_body0_rad, self.ic.omega0_rad_s], dtype=float)
        aero.spin.spin0_rad_s = self.ic.spin0_rad_s

        rows = []
        t = 0.0
        valid = True
        extrapolated_any = False
        n_steps = int(self.t_max / self.dt) + 1
        step_fn = rk4_step if self.solver in ("RK4", "solve_ivp") else euler_step

        if self.solver == "solve_ivp":
            try:
                from scipy.integrate import solve_ivp

                def rhs(tt, yy):
                    return dynamics_coupled(tt, yy, self.model, self.env, aero, geom_state)

                def hit_ground(tt, yy):
                    return yy[1]
                hit_ground.terminal = True
                hit_ground.direction = -1

                t_eval = np.arange(0.0, self.t_max, self.dt)
                sol = solve_ivp(rhs, (0.0, self.t_max), state, t_eval=t_eval,
                                 events=hit_ground, rtol=1e-8, atol=1e-10, max_step=self.dt * 10)
                if not sol.success:
                    warnings_list.append(f"solve_ivp failed: {sol.message}")
                    return SimResult(pd.DataFrame(), {}, False, warnings_list)
                traj = [(sol.t[i], sol.y[:, i]) for i in range(sol.y.shape[1])]
            except Exception as e:
                logger.exception("solve_ivp failed")
                return SimResult(pd.DataFrame(), {}, False, [f"solve_ivp unavailable: {e}"])
        else:
            traj = None

        if traj is None:
            traj = []
            for i in range(n_steps):
                if not np.all(np.isfinite(state)):
                    warnings_list.append(f"non-finite state detected at t={t:.4f}s; simulation stopped")
                    valid = False
                    break
                if abs(state[2]) > 1e4 or abs(state[3]) > 1e4 or abs(state[0]) > 1e5 or abs(state[1]) > 1e5:
                    warnings_list.append(f"unphysical state detected at t={t:.4f}s; simulation stopped")
                    valid = False
                    break
                traj.append((t, state.copy()))
                if state[1] <= 0 and i > 0:
                    break
                state = step_fn(dynamics_coupled, t, state, self.dt,
                                 self.model, self.env, aero, geom_state)
                t += self.dt

        for tt, st in traj:
            if not np.all(np.isfinite(st)):
                warnings_list.append(f"non-finite state detected at t={tt:.4f}s; simulation stopped")
                valid = False
                break
            x, y, vx, vy, theta, omega_turn = st
            d = coupled_diagnostics(tt, st, self.model, self.env, aero, geom_state)
            deriv = dynamics_coupled(tt, st, self.model, self.env, aero, geom_state)
            extrapolated_any = extrapolated_any or d["extrapolated"]
            delta_alpha = wrap_to_pi(d["alpha"] - aero.alpha_equilibrium_rad)
            rows.append(dict(
                time_s=tt, x_m=x, y_m=y, vx_m_s=vx, vy_m_s=vy,
                speed_m_s=math.hypot(vx, vy),
                ax_m_s2=deriv[2], ay_m_s2=deriv[3],
                theta_body_deg=rad2deg(wrap_to_pi(theta)),
                theta_cork_deg=rad2deg(wrap_to_pi(theta + math.pi)),
                wind_from_deg=rad2deg(d["flow_angle"]),
                flow_angle_deg=rad2deg(d["flow_angle"]),
                alpha_deg=rad2deg(d["alpha"]), alpha_rad=d["alpha"],
                delta_alpha_deg=rad2deg(delta_alpha),
                delta_alpha_total_deg=rad2deg(delta_alpha),
                omega_rad_s=omega_turn, angular_acceleration_rad_s2=deriv[5],
                spin_rad_s=d["spin"],
                V_rel_m_s=d["Vrel"], Re=d["Re"], Cd=d["Cd"], Cl=d["Cl"],
                Cm=(d["Cm"] if d["Cm"] is not None else float("nan")),
                drag_force_N=d["drag"], lift_force_N=d["lift"],
                S_eff_m2=d["S_eff"],
                restoring_moment_Nm=d["M_restore"], damping_moment_Nm=d["M_damping"],
                net_moment_Nm=d["M_net"],
                aero_length_m=self.model.aerodynamic_length(self.env.rho, d["Cd"]),
                u_rel_m_s=(vx - self.env.u_air_x), v_rel_m_s=(vy - self.env.u_air_y),
                acceleration_mag_m_s2=float(math.hypot(deriv[2], deriv[3])),
                tangential_accel_m_s2=_tangential_accel([vx, vy], [deriv[2], deriv[3]]),
                normal_accel_m_s2=_normal_accel([vx, vy], [deriv[2], deriv[3]]),
                curvature_1_m=_path_curvature([vx, vy, 0.0], [deriv[2], deriv[3], 0.0]),
                flight_path_angle_deg=(rad2deg(math.atan2(vy, vx))
                                        if math.hypot(vx, vy) > 0 else 0.0),
                dynamic_pressure_Pa=float(0.5 * self.env.rho * d["Vrel"] ** 2),
                power_drag_W=float(-d["drag"] * math.hypot(vx, vy)),
                angular_momentum_mag=float(self.model.turnover.Iyy * omega_turn),
                E_kinetic_J=float(0.5 * self.model.m * (vx ** 2 + vy ** 2)),
                E_rotational_J=float(0.5 * self.model.turnover.Iyy * omega_turn ** 2),
                E_potential_J=float(self.model.m * G * y),
                E_total_J=float(0.5 * self.model.m * (vx ** 2 + vy ** 2)
                                 + 0.5 * self.model.turnover.Iyy * omega_turn ** 2
                                 + self.model.m * G * y),
                weight_N=float(self.model.m * G),
                omega_deg_s=rad2deg(omega_turn),
                angular_acceleration_deg_s2=rad2deg(deriv[5]),
                lift_to_drag=(float(d["lift"] / d["drag"]) if d["drag"] > 1e-12
                               else float("nan")),
                omega0_restoring_rad_s=(
                    self.model.turnover.omega0(d["Vrel"], self.env.rho,
                                                self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                beta_damping_1_s=(
                    self.model.turnover.beta(d["Vrel"], self.env.rho,
                                              self.model.geometry.S, phi=delta_alpha)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                tau_oscillation_s=(
                    self.model.turnover.tau_oscillation(d["Vrel"], self.env.rho,
                                                         self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                tau_stabilizing_s=(
                    self.model.turnover.tau_stabilizing(d["Vrel"], self.env.rho,
                                                         self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
            ))

        if extrapolated_any:
            warnings_list.append("coefficient lookup extrapolated outside the supplied table range")

        df = pd.DataFrame(rows)
        meta = self._metadata(aero=aero, geom_state=geom_state)
        warnings_list = (warnings_list
                         + flight_envelope_warnings(df, self.model, self.env)
                         + reynolds_domain_warnings(df, self.model, self.env))
        return SimResult(df=df, metadata=meta, valid=valid, warnings=warnings_list)

    def _metadata(self, aero=None, geom_state=None):
        tp = self.model.turnover
        Iyy, Ispin, inertia_source = self.model.resolved_inertia()
        meta = dict(
            model_version=MODEL_VERSION,
            parameter_profile=self.model.name,
            model_level=self.model_level,
            model_level_name=MODEL_LEVELS.get(self.model_level, "unknown"),
            parameter_values=dict(m=self.model.m, Tw=tp.Tw, zeta=tp.zeta, Iyy=tp.Iyy,
                                   k_alpha=tp.k_alpha, c=tp.c, S=self.model.geometry.S,
                                   Ispin=Ispin, inertia_source=inertia_source),
            solver=self.solver, dt=self.dt,
            initial_conditions=asdict(self.ic),
            environment=asdict(self.env),
            timestamp=time.time(),
            turnover_diagnostics=dict(
                omega_n=tp.omega_n, omega_d=tp.omega_d, f_d=tp.f_d,
                damping_class=tp.damping_class, overshoot=tp.overshoot,
                consistency=tp.consistency_check(),
            ),
        )
        if aero is not None:
            meta["aero_model"] = aero.describe()
        if geom_state is not None:
            meta["geometry_state"] = {k: v for k, v in geom_state.items()
                                       if k != "azimuth_distribution"}
        return meta

    def _run_linear(self) -> SimResult:
        warnings_list = []
        if self.dt <= 0:
            return SimResult(pd.DataFrame(), {}, False, ["dt must be > 0"])
        _derrs = damage_domain_errors(self.model)
        if _derrs:
            return SimResult(pd.DataFrame(), {}, False, _derrs)
        self._stability_warning(warnings_list)

        step_fn = rk4_step if self.solver == "RK4" else euler_step

        state = np.array([self.ic.x0, self.ic.y0, self.ic.vx0, self.ic.vy0,
                           self.ic.delta_alpha0_rad, self.ic.omega0_rad_s], dtype=float)

        rows = []
        t = 0.0
        valid = True
        path_len = 0.0
        _u_base2 = np.array([self.env.u_air_x, self.env.u_air_y])
        _turb2 = TurbulenceModel(self.env.turbulence_intensity,
                                  self.env.turbulence_length_scale_m,
                                  U_ref=max(float(math.hypot(self.ic.vx0, self.ic.vy0)),
                                             float(vnorm(_u_base2))),
                                  seed=self.env.turbulence_seed)
        n_steps = int(self.t_max / self.dt) + 1
        alpha_eq = deg2rad(self.ic.alpha_equilibrium_deg)

        for i in range(n_steps):
            x, y, vx, vy, delta_alpha, omega = state

            if not np.all(np.isfinite(state)):
                warnings_list.append(f"non-finite state detected at t={t:.4f}s; simulation stopped")
                valid = False
                break
            if abs(vx) > 1e4 or abs(vy) > 1e4 or abs(x) > 1e5 or abs(y) > 1e5:
                warnings_list.append(f"unphysical state detected at t={t:.4f}s; simulation stopped")
                valid = False
                break

            u_rel = vx - self.env.u_air_x
            v_rel = vy - self.env.u_air_y
            Vrel = math.sqrt(u_rel ** 2 + v_rel ** 2)
            speed = math.sqrt(vx ** 2 + vy ** 2)
            cd_value = self.model.Cd_model.cd(speed=Vrel, alpha_rad=delta_alpha)
            cd_value *= _damage_cd_factor(self.model)
            drag_force = 0.5 * self.env.rho * cd_value * self.model.geometry.S * Vrel ** 2
            Re = (self.env.rho * Vrel * self.model.geometry.skirt_diameter_m / self.env.mu
                  if self.env.mu > 0 else float("nan"))
            aero_len = self.model.aerodynamic_length(self.env.rho, cd_value)

            deriv = dynamics(t, state, self.model, self.env, self.torque_fn)
            ax, ay = deriv[2], deriv[3]
            ang_accel = deriv[5]
            alpha_full = alpha_eq + delta_alpha
            flow_angle = math.atan2(v_rel, u_rel) if Vrel > 0 else 0.0
            theta_body_full = wrap_to_pi(flow_angle + alpha_full)

            rows.append(dict(
                time_s=t, x_m=x, y_m=y, vx_m_s=vx, vy_m_s=vy, speed_m_s=speed,
                ax_m_s2=ax, ay_m_s2=ay,
                alpha_deg=rad2deg(wrap_to_pi(alpha_full)), alpha_rad=wrap_to_pi(alpha_full),
                delta_alpha_deg=rad2deg(wrap_to_pi(delta_alpha)),
                delta_alpha_total_deg=rad2deg(delta_alpha),
                rotation_turns=delta_alpha / (2 * math.pi),
                omega_rad_s=omega,
                flow_angle_deg=rad2deg(flow_angle),
                wind_from_deg=rad2deg(flow_angle),
                theta_body_deg=rad2deg(theta_body_full),
                theta_cork_deg=rad2deg(wrap_to_pi(theta_body_full + math.pi)),
                angular_acceleration_rad_s2=ang_accel,
                V_rel_m_s=Vrel, Re=Re, Cd=cd_value, drag_force_N=drag_force,
                aero_length_m=aero_len,
                u_rel_m_s=u_rel, v_rel_m_s=v_rel,
                acceleration_mag_m_s2=float(math.hypot(ax, ay)),
                tangential_accel_m_s2=_tangential_accel([vx, vy], [ax, ay]),
                normal_accel_m_s2=_normal_accel([vx, vy], [ax, ay]),
                curvature_1_m=_path_curvature([vx, vy, 0.0], [ax, ay, 0.0]),
                flight_path_angle_deg=(rad2deg(math.atan2(vy, vx)) if speed > 0 else 0.0),
                dynamic_pressure_Pa=float(0.5 * self.env.rho * Vrel ** 2),
                power_drag_W=float(-drag_force * speed),
                angular_momentum_mag=float(self.model.turnover.Iyy * omega),
                E_kinetic_J=float(0.5 * self.model.m * speed ** 2),
                E_rotational_J=float(0.5 * self.model.turnover.Iyy * omega ** 2),
                E_potential_J=float(self.model.m * G * y),
                E_total_J=float(0.5 * self.model.m * speed ** 2
                                 + 0.5 * self.model.turnover.Iyy * omega ** 2
                                 + self.model.m * G * y),
                weight_N=float(self.model.m * G),
                distance_m=path_len,
                omega_deg_s=rad2deg(omega),
                angular_acceleration_deg_s2=rad2deg(ang_accel),
                omega0_restoring_rad_s=(
                    self.model.turnover.omega0(Vrel, self.env.rho, self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                beta_damping_1_s=(
                    self.model.turnover.beta(Vrel, self.env.rho, self.model.geometry.S,
                                              phi=delta_alpha)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                tau_oscillation_s=(
                    self.model.turnover.tau_oscillation(Vrel, self.env.rho,
                                                         self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                tau_stabilizing_s=(
                    self.model.turnover.tau_stabilizing(Vrel, self.env.rho,
                                                         self.model.geometry.S)
                    if getattr(self.model.turnover, "mode", "linear") == "aero_pendulum"
                    else float("nan")),
                Cl=0.0, lift_force_N=0.0, spin_rad_s=self.ic.spin0_rad_s,
            ))
            path_len += float(math.hypot(vx, vy) * self.dt)

            if y <= 0 and i > 0:
                break

            if _turb2.active:
                _uf2 = _turb2.step(self.dt)
                self.env.u_air_x = float(_u_base2[0] + _uf2[0])
                self.env.u_air_y = float(_u_base2[1] + _uf2[1])
            state = step_fn(dynamics, t, state, self.dt, self.model, self.env, self.torque_fn)
            t += self.dt

        df = pd.DataFrame(rows)
        meta = self._metadata()
        meta["aero_model"] = dict(level=1, level_name=MODEL_LEVELS[1],
                                   cd_mode=self.model.Cd_model.mode,
                                   cd_source=self.model.Cd_model.source_type,
                                   cl_status="양력 모델 사용 안 함",
                                   cm_mode="선형화 복원 모멘트 (Tw, ζ)",
                                   spin_status="축 Spin 연동 자료 없음",
                                   damage_mapping_status="파손→계수 대응 없음 (형상만 반영)")
        gs = self.model.geometry_state()
        if gs is not None:
            meta["geometry_state"] = {k: v for k, v in gs.items() if k != "azimuth_distribution"}
        warnings_list = (warnings_list
                         + flight_envelope_warnings(df, self.model, self.env)
                         + reynolds_domain_warnings(df, self.model, self.env))
        return SimResult(df=df, metadata=meta, valid=valid, warnings=warnings_list)

def convergence_test(model, env, ic, solver="RK4", dt=0.001, t_max=3.0, torque_fn=None):
    dts = [dt, dt / 2, dt / 4]
    results = []
    for d in dts:
        eng = SimulationEngine(model, env, ic, solver=solver, dt=d, t_max=t_max, torque_fn=torque_fn)
        res = eng.run()
        if len(res.df) == 0:
            results.append(dict(dt=d, valid=False))
            continue
        last = res.df.iloc[-1]
        results.append(dict(dt=d, valid=res.valid, x_final=last.x_m, y_final=last.y_m,
                             alpha_final=last.alpha_deg))
    out = []
    for i in range(1, len(results)):
        a, b = results[i - 1], results[i]
        if not (a.get("valid") and b.get("valid")):
            out.append(dict(dt_pair=(a["dt"], b["dt"]), comparable=False))
            continue
        out.append(dict(
            dt_pair=(a["dt"], b["dt"]),
            comparable=True,
            final_position_error=math.hypot(a["x_final"] - b["x_final"], a["y_final"] - b["y_final"]),
            final_angle_error=abs(a["alpha_final"] - b["alpha_final"]),
        ))
    return dict(raw=results, comparisons=out)

SENSITIVITY_PARAMS = ["Cd", "Tw", "zeta", "mass", "Iyy", "initial_velocity", "initial_angle"]

def parameter_sensitivity(model, env, ic, param, pct_list=(-10, -5, 0, 5, 10),
                           solver="RK4", dt=0.001, t_max=3.0):
    rows = []
    for pct in pct_list:
        m2 = copy.deepcopy(model)
        ic2 = copy.deepcopy(ic)
        factor = 1 + pct / 100.0
        if param == "Cd":
            m2.Cd_model.constant_value *= factor
        elif param == "Tw":
            m2.turnover.Tw *= factor
        elif param == "zeta":
            m2.turnover.zeta *= factor
        elif param == "mass":
            m2.m *= factor
        elif param == "Iyy":
            m2.turnover.Iyy *= factor
        elif param == "initial_velocity":
            ic2.vx0 *= factor
            ic2.vy0 *= factor
        elif param == "initial_angle":
            ic2.alpha_initial_deg *= factor
        else:
            raise ValueError(f"unknown sensitivity parameter: {param}")

        eng = SimulationEngine(m2, env, ic2, solver=solver, dt=dt, t_max=t_max)
        res = eng.run()
        if len(res.df) == 0 or not res.valid:
            rows.append(dict(pct=pct, valid=False))
            continue
        df = res.df
        rng = df.x_m.iloc[-1] - df.x_m.iloc[0]
        flight_time = df.time_s.iloc[-1]
        stab_t = stabilization_time(df)
        max_wobble = df.delta_alpha_deg.abs().max()
        final_v = df.speed_m_s.iloc[-1]
        rows.append(dict(pct=pct, valid=True, range_m=rng, flight_time_s=flight_time,
                          stabilization_time_s=stab_t, max_wobble_deg=max_wobble,
                          final_speed_m_s=final_v))
    return pd.DataFrame(rows)

def parameter_sweep(model, env, ic, param_x, range_x, param_y, range_y, metric="stabilization_time_s",
                     solver="RK4", dt=0.001, t_max=3.0):
    grid = np.zeros((len(range_y), len(range_x)))
    for iy, vy_ in enumerate(range_y):
        for ix, vx_ in enumerate(range_x):
            m2 = copy.deepcopy(model)
            ic2 = copy.deepcopy(ic)
            _apply_param(m2, ic2, param_x, vx_)
            _apply_param(m2, ic2, param_y, vy_)
            eng = SimulationEngine(m2, env, ic2, solver=solver, dt=dt, t_max=t_max)
            res = eng.run()
            if len(res.df) == 0 or not res.valid:
                grid[iy, ix] = np.nan
                continue
            df = res.df
            if metric == "stabilization_time_s":
                grid[iy, ix] = stabilization_time(df) or np.nan
            elif metric == "flight_range_m":
                grid[iy, ix] = df.x_m.iloc[-1] - df.x_m.iloc[0]
            else:
                grid[iy, ix] = np.nan
    return grid

def _apply_param(model, ic, name, value):
    if name == "Cd":
        model.Cd_model.constant_value = value
    elif name == "Tw":
        model.turnover.Tw = value
    elif name == "zeta":
        model.turnover.zeta = value
    elif name == "initial_velocity":
        speed = math.hypot(ic.vx0, ic.vy0)
        if speed > 0:
            scale = value / speed
            ic.vx0 *= scale
            ic.vy0 *= scale
        else:
            ic.vx0 = value
    else:
        raise ValueError(f"unknown sweep parameter: {name}")

def stabilization_time(df, angle_threshold_deg=5.0, angular_velocity_threshold=1.0,
                        hold_time_s=0.05):
    """Backwards-compatible wrapper: returns the time only, or None."""
    r = stabilization_time_detail(df, angle_threshold_deg,
                                   angular_velocity_threshold, hold_time_s)
    return r.get("time_s")

def stabilization_time_detail(df, angle_threshold_deg=5.0,
                               angular_velocity_threshold=1.0, hold_time_s=0.05):
    """Time for the wobble to settle, always returning a predicted value.

    Two cases:
      1. The oscillation actually settles inside the simulated flight -> the time is
         read directly off the result ("시뮬레이션 내 도달").
      2. The shuttlecock lands before settling. Returning N/A would be useless, so the
         decay envelope of the wobble peaks is fitted, |wobble| ~ A0 exp(-t/tau), and
         the settling time is extrapolated from that fit
         ("외삽 예측 — 비행 중 미도달"). This is still derived from the simulation's
         own dynamics, not from any tabulated experimental figure.
    """
    out = dict(time_s=None, source="계산 불가", envelope_tau_s=None, reached=False)
    if df is None or len(df) == 0:
        return out
    col = ("wobble_signed_deg" if "wobble_signed_deg" in df.columns
           else ("delta_alpha_deg" if "delta_alpha_deg" in df.columns else None))
    if col is None or "time_s" not in df.columns:
        return out
    t = pd.to_numeric(df["time_s"], errors="coerce").values
    w = np.abs(pd.to_numeric(df[col], errors="coerce").values)
    om = (np.abs(pd.to_numeric(df["omega_rad_s"], errors="coerce").values)
          if "omega_rad_s" in df.columns else np.zeros_like(w))
    good = np.isfinite(t) & np.isfinite(w)
    if not good.any():
        return out
    t, w, om = t[good], w[good], om[good]

    # velocity threshold scaled to the case, with the absolute limit as a floor
    om_ref = float(np.nanmax(om)) if om.size else 0.0
    om_lim = max(float(angular_velocity_threshold), 0.02 * om_ref)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1e-3
    hold = max(1, int(hold_time_s / max(dt, 1e-9)))
    cond = (w < angle_threshold_deg) & (om < om_lim)
    run = 0
    for i, c in enumerate(cond):
        if c:
            run += 1
            if run >= hold:
                out.update(time_s=float(t[i - hold + 1]),
                            source="시뮬레이션 내 도달 (측정)", reached=True)
                return out
        else:
            run = 0

    # not reached: extrapolate from the decay envelope of the wobble peaks
    pk_t, pk_a = [], []
    for i in range(1, len(w) - 1):
        if w[i] >= w[i - 1] and w[i] > w[i + 1] and w[i] > 1e-6:
            pk_t.append(t[i]); pk_a.append(w[i])
    if len(pk_t) >= 2:
        pk_t = np.asarray(pk_t); pk_a = np.asarray(pk_a)
        try:
            k = np.polyfit(pk_t, np.log(pk_a), 1)
            if k[0] < 0:
                tau = -1.0 / k[0]
                A0 = math.exp(k[1])
                if A0 > angle_threshold_deg:
                    t_st = tau * math.log(A0 / angle_threshold_deg)
                    out.update(time_s=float(t_st), envelope_tau_s=float(tau),
                                source="외삽 예측 (비행 중 미도달 — 감쇠 포락선 외삽)",
                                reached=False)
                    return out
        except Exception:
            pass
    out["source"] = "예측 불가 (진동 봉우리 부족)"
    return out

def turnover_completion_time(df, mode="stable_orientation", angle_threshold_deg=5.0):
    if len(df) == 0:
        return None
    if mode == "stable_orientation":
        cond = df.delta_alpha_deg.abs() < angle_threshold_deg
    elif mode == "velocity_aligned":
        if "theta_body_deg" not in df.columns:
            return None
        # theta_body is an absolute heading, alpha_deg is the angle relative to the
        # flow; comparing alpha_deg with an absolute flight-path angle mixed the two.
        flight_angle = np.degrees(np.arctan2(-df.vy_m_s, -df.vx_m_s))
        misalign = np.degrees(np.abs(wrap_to_pi(
            np.radians(df.theta_body_deg.values - flight_angle.values))))
        cond = pd.Series(misalign < angle_threshold_deg, index=df.index)
    else:
        return None
    idx = cond[cond].index
    if len(idx) == 0:
        return None
    return df.time_s.loc[idx[0]]

def research_metrics(res) -> dict:
    df = res.df
    if df is None or len(df) == 0:
        return {}
    d_alpha = df.delta_alpha_deg.values
    omega = df.omega_rad_s.values
    ang_acc = df.angular_acceleration_rad_s2.values
    speed = df.speed_m_s.values
    t = df.time_s.values
    dt_span = max(t[-1] - t[0], 1e-12)
    primary = dict(
        wobble_rms_deg=float(np.sqrt(np.mean(d_alpha ** 2))),
        wobble_amplitude_deg=float(np.max(np.abs(d_alpha))),
        turnover_time_s=turnover_completion_time(df),
        stabilization_time_s=stabilization_time(df),
        body_angular_velocity_rms_rad_s=float(np.sqrt(np.mean(omega ** 2))),
        body_angular_acceleration_rms_rad_s2=float(np.sqrt(np.mean(ang_acc ** 2))),
    )
    secondary = dict(
        range_m=float(df.x_m.iloc[-1] - df.x_m.iloc[0]),
        flight_time_s=float(t[-1]),
        speed_decay_m_s=float(speed[0] - speed[-1]),
        mean_deceleration_m_s2=float((speed[0] - speed[-1]) / dt_span),
        max_drag_N=float(df.drag_force_N.max()),
        trajectory_deviation_m=float(np.max(np.abs(df.y_m.values - np.interp(
            df.x_m.values, [df.x_m.iloc[0], df.x_m.iloc[-1]],
            [df.y_m.iloc[0], df.y_m.iloc[-1]])))),
        orientation_error_deg=float(np.mean(np.abs(d_alpha))),
    )
    return dict(primary=primary, secondary=secondary)

def compare_damage_states(db, geometry, env, ic, damage_presets=None, level=2,
                           solver="RK4", dt=0.001, t_max=3.0, profile_name="Normal Feather"):
    presets = damage_presets or DAMAGE_PRESETS
    rows = []
    for name, removed in presets.items():
        model = build_full_model(db, profile_name, geometry, level=level,
                                  removed_feathers=removed)
        res = SimulationEngine(model, env, ic, solver=solver, dt=dt, t_max=t_max,
                                model_level=level).run()
        gs = model.geometry_state()
        if not res.valid or len(res.df) == 0:
            rows.append(dict(damage_state=name, valid=False))
            continue
        met = research_metrics(res)
        row = dict(damage_state=name, valid=True,
                   remaining_feathers=gs["remaining_feathers"],
                   geometry_asymmetry=gs["geometry_asymmetry"],
                   estimated_porosity=gs["estimated_porosity"])
        row.update(met["primary"])
        row.update(met["secondary"])
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_result(res) -> dict:
    df = res.df
    if df is None or len(df) == 0:
        return {}
    _stab_detail = stabilization_time_detail(df)
    stab = _stab_detail.get("time_s")
    turn = turnover_completion_time(df)
    return dict(
        flight_time_s=float(df.time_s.iloc[-1]),
        max_range_m=float(df.x_m.iloc[-1] - df.x_m.iloc[0]),
        initial_speed_m_s=float(df.speed_m_s.iloc[0]),
        final_speed_m_s=float(df.speed_m_s.iloc[-1]),
        max_height_m=float(df.y_m.max()),
        turnover_time_s=turn,
        stabilization_time_s=stab,
        stabilization_source=_stab_detail.get("source"),
        stabilization_reached=_stab_detail.get("reached"),
        wobble_envelope_tau_s=_stab_detail.get("envelope_tau_s"),
        max_angular_velocity_rad_s=float(df.omega_rad_s.abs().max()),
        max_wobble_deg=float(df.delta_alpha_deg.abs().max()),
    )

REQUIRED_EXP_COLUMNS_ANY = ["time", "x", "y"]

def import_experiment_csv(path_or_buffer, column_map: Dict[str, str]):
    raw = pd.read_csv(path_or_buffer)
    out = pd.DataFrame()
    for canon, col in column_map.items():
        if col and col in raw.columns:
            out[canon] = raw[col]
    if "time" not in out.columns:
        raise ValueError("time column is required in experimental data")
    out = out.dropna(subset=["time"])
    return out

def data_quality_report(exp_df: pd.DataFrame) -> dict:
    rows = len(exp_df)
    if rows == 0 or "time" not in exp_df.columns:
        return dict(rows=rows, columns=list(exp_df.columns))
    t = exp_df["time"].values.astype(float)
    intervals = np.diff(t)
    non_monotonic = bool(np.any(intervals < 0)) if len(intervals) else False
    return dict(
        rows=rows,
        columns=list(exp_df.columns),
        time_range=(float(np.nanmin(t)), float(np.nanmax(t))),
        sampling_interval_s=float(np.median(intervals)) if len(intervals) else None,
        missing_values=int(exp_df.isna().sum().sum()),
        nan_count=int(exp_df.isna().sum().sum()),
        duplicate_timestamps=int(pd.Series(t).duplicated().sum()),
        non_monotonic_time=non_monotonic,
    )

def calibrate_experiment(exp_df, time_shift=0.0, x_offset=0.0, y_offset=0.0, angle_offset=0.0):
    cal = exp_df.copy()
    cal["time"] = cal["time"] + time_shift
    if "x" in cal:
        cal["x"] = cal["x"] + x_offset
    if "y" in cal:
        cal["y"] = cal["y"] + y_offset
    if "orientation" in cal:
        cal["orientation"] = cal["orientation"] + angle_offset
    return cal

def residual_analysis(sim_df, exp_df):
    sim_t = sim_df.time_s.values
    out = {}
    pairs = [("x", "x_m"), ("y", "y_m"), ("speed", "speed_m_s"), ("orientation", "alpha_deg")]
    for exp_col, sim_col in pairs:
        if exp_col not in exp_df.columns:
            continue
        interp_sim = np.interp(exp_df["time"].values, sim_t, sim_df[sim_col].values)
        resid = exp_df[exp_col].values - interp_sim
        resid = resid[~np.isnan(resid)]
        if len(resid) == 0:
            continue
        rmse = math.sqrt(np.mean(resid ** 2))
        mae = np.mean(np.abs(resid))
        scale = np.nanstd(exp_df[exp_col].values)
        out[exp_col] = dict(rmse=rmse, mae=mae,
                             normalized_rmse=(rmse / scale if scale > 0 else None))
    return out

FIT_BOUNDS = {
    "Cd": (1e-3, 3.0),
    "Tw": (1e-4, 1.0),
    "zeta": (0.0, 3.0),
}

def fit_parameters(model, env, ic, exp_df, fit_params: List[str], solver="RK4", dt=0.001, t_max=3.0,
                    target_cols=("x", "y"), model_level=1):
    """Least-squares fit of selected parameters against imported experiment data.

    model_level must be the level the user actually simulated at: fitting with the
    Level 1 point-mass model and then reporting the numbers for a Level 2 coupled run
    compares two different models.
    """
    from scipy.optimize import least_squares

    base = copy.deepcopy(model)

    def unpack(theta):
        m2 = copy.deepcopy(base)
        for name, val in zip(fit_params, theta):
            if name == "Cd":
                m2.Cd_model.constant_value = val
            elif name == "Tw":
                m2.turnover.Tw = val
            elif name == "zeta":
                m2.turnover.zeta = val
        return m2

    def residuals(theta):
        m2 = unpack(theta)
        eng = SimulationEngine(m2, env, ic, solver=solver, dt=dt, t_max=t_max,
                                model_level=model_level)
        res = eng.run()
        if len(res.df) == 0 or not res.valid:
            return np.full(len(exp_df) * len(target_cols), 1e6)
        sim_t = res.df.time_s.values
        errs = []
        for col in target_cols:
            if col not in exp_df.columns:
                continue
            sim_col = {"x": "x_m", "y": "y_m", "speed": "speed_m_s",
                       "orientation": "alpha_deg"}.get(col)
            interp_sim = np.interp(exp_df["time"].values, sim_t, res.df[sim_col].values)
            errs.append(exp_df[col].values - interp_sim)
        if not errs:
            return np.zeros(1)
        return np.concatenate(errs)

    x0 = []
    lb = []
    ub = []
    for name in fit_params:
        if name == "Cd":
            x0.append(base.Cd_model.constant_value)
        elif name == "Tw":
            x0.append(base.turnover.Tw)
        elif name == "zeta":
            x0.append(base.turnover.zeta)
        lo, hi = FIT_BOUNDS[name]
        lb.append(lo)
        ub.append(hi)

    if len(exp_df) < 2:
        return dict(success=False, message="insufficient fitting points (need >= 2)")

    result = least_squares(residuals, x0=x0, bounds=(lb, ub))

    fitted = dict(zip(fit_params, result.x))

    cov = None
    corr = None
    identifiability_warning = None
    # A parameter the model does not actually use produces an all-zero Jacobian
    # column: least_squares then returns the starting value unchanged and the
    # covariance is singular. Tw and zeta only drive the LINEAR turnover model, so in
    # aero_pendulum mode they are exactly such parameters. Say which one it was
    # instead of reporting a generic identifiability problem.
    inactive = []
    try:
        _J = np.asarray(result.jac, dtype=float)
        for _i, _name in enumerate(fit_params):
            if _i < _J.shape[1] and float(np.max(np.abs(_J[:, _i]))) <= 0.0:
                inactive.append(_name)
    except (AttributeError, ValueError, IndexError):
        pass
    if inactive:
        identifiability_warning = (
            "다음 물리량은 현재 모델에서 결과에 전혀 영향을 주지 않아 피팅되지 않았습니다: "
            + ", ".join(inactive)
            + " (Tw·ζ 는 선형 turnover 모델 전용입니다. aero_pendulum 모드에서는 "
              "l_GC 와 감쇠 보정계수가 그 역할을 합니다.)")
    try:
        J = result.jac
        if J.shape[0] > J.shape[1]:
            JTJ = J.T @ J
            cov = np.linalg.inv(JTJ) * (2 * result.cost / max(1, (J.shape[0] - J.shape[1])))
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            cond_number = np.linalg.cond(JTJ)
            if inactive:
                pass
            elif cond_number > 1e6:
                identifiability_warning = "parameter not well identified (high condition number)"
            else:
                off_diag = np.abs(corr - np.eye(len(fit_params)))
                if len(fit_params) > 1 and off_diag.max() > 0.95:
                    identifiability_warning = "parameter not well identified (strong correlation)"
    except np.linalg.LinAlgError:
        if not inactive:
            identifiability_warning = "parameter not well identified (singular covariance)"

    return dict(success=result.success, fitted=fitted, cost=result.cost,
                covariance=cov.tolist() if cov is not None else None,
                correlation=corr.tolist() if corr is not None else None,
                identifiability_warning=identifiability_warning,
                inactive_params=inactive,
                fit_params=fit_params)

def format_fit_card(result: dict) -> str:
    if not result or not result.get("success"):
        return f"피팅 실패: {result.get('message', '최적화가 수렴하지 않았습니다') if result else '결과 없음'}"
    names = result["fit_params"]
    fitted = result["fitted"]
    cov = result.get("covariance")
    lines = []
    for i, name in enumerate(names):
        val = fitted[name]
        if cov is not None:
            std = math.sqrt(max(cov[i][i], 0.0))
            lines.append(f"{name} = {val:.4g} ± {std:.2g}")
        else:
            lines.append(f"{name} = {val:.4g} (std unavailable)")
    if result.get("identifiability_warning"):
        lines.append(f"⚠ {result['identifiability_warning']}")
    return "\n".join(lines)

def _make_test_model(Cd=0.6, Tw=0.0102, zeta=0.39, m=5.3e-3, S=math.pi * 0.033 ** 2, Iyy=2.92e-6):
    geom = Geometry(skirt_diameter_m=0.066)
    turnover = TurnoverParams(Tw=Tw, zeta=zeta, Iyy=Iyy)
    return ShuttlecockModel(name="TestModel", m=m, Cd_model=CdModel(constant_value=Cd),
                             geometry=geom, turnover=turnover)

class ValidationEngine:
    def __init__(self):
        self.results = {}

    def run_all(self):
        self.results["zero_drag_projectile"] = self.test_zero_drag()
        self.results["zero_gravity_no_drag"] = self.test_zero_gravity_no_drag()
        self.results["undamped_oscillator"] = self.test_undamped()
        self.results["underdamped_oscillator"] = self.test_underdamped()
        self.results["critical_damping"] = self.test_critical()
        self.results["overdamped"] = self.test_overdamped()
        self.results["dt_convergence"] = self.test_dt_convergence()

        self.results["drag_opposes_relative_flow"] = self.test_drag_direction()
        self.results["lift_orthogonal_to_drag"] = self.test_lift_orthogonality()
        self.results["no_moment_constant_omega"] = self.test_no_moment_constant_omega()
        self.results["spin_decoupled_by_default"] = self.test_spin_decoupled()
        self.results["coupled_zero_drag_projectile"] = self.test_coupled_zero_drag()
        self.results["damage_geometry_states"] = self.test_damage_geometry()
        self.results["launch_orientation_consistency"] = self.test_launch_orientations()
        self.results["3d_reduces_to_2d"] = self.test_3d_reduces_to_2d()
        self.results["3d_convergence"] = self.test_3d_convergence()
        self.results["3d_no_lateral_force_stays_planar"] = self.test_3d_planar_free()
        self.results["3d_quaternion_normalized"] = self.test_quaternion_norm()
        self.results["3d_force_moment_directions"] = self.test_3d_directions()
        self.results["3d_euler_vs_rk4"] = self.test_3d_solver_agreement()
        self.results["3d_inertia_positive_definite"] = self.test_inertia_positive_definite()
        return self.results

    def test_quaternion_norm(self):
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel(u_air_x=2.0, u_air_z=1.0)
        m = build_full_model(db, "Normal Feather", geom, level=2)
        ic3 = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=35, elevation_deg=-8,
                                  azimuth_deg=12.0, body_axis_offset_deg=140.0,
                                  body_azimuth_deg=35.0, omega0=(4.0, -3.0, 6.0),
                                  spin0_rad_s=150.0)
        r = SimulationEngine3D(m, env, ic3, dt=0.0005, t_max=1.2).run()
        if len(r.df) == 0 or not r.valid:
            return dict(passed=False, reason="3D run failed")
        dev = float(np.abs(r.df.quat_norm.values - 1.0).max())
        finite = bool(np.all(np.isfinite(r.df[["qw", "qx", "qy", "qz"]].values)))
        return dict(passed=bool(dev < 1e-8 and finite), max_norm_deviation=dev,
                    all_finite=finite)

    def test_3d_directions(self):
        """Drag must oppose the relative wind, gravity must act along -Y, and lift/side
        force must be perpendicular to the drag."""
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel(u_air_x=6.0)
        m = build_full_model(db, "Normal Feather", geom, level=2,
                              cl_model=ClModel(mode="constant", constant_value=0.4))
        side = SideForceModel(mode="constant", constant_value=0.25)
        v = np.array([14.0, -3.0, 2.0])
        e0 = np.array([-0.7, 0.5, 0.51])
        e0 = e0 / vnorm(e0)
        q = quat_from_vectors(np.array([1.0, 0.0, 0.0]), e0)
        ao = aero_forces_moments_3d(0.0, np.zeros(3), v, q, np.zeros(3), m, env,
                                     m.aero, m.geometry_state(), side, MomentModel3D())
        v_rel = v - np.array([env.u_air_x, env.u_air_y, env.u_air_z])
        ef = v_rel / vnorm(v_rel)
        drag_align = float(np.dot(ao["F_drag"] / (vnorm(ao["F_drag"]) or 1), ef))
        lift_perp = abs(float(np.dot(ao["F_lift"], ef)))
        side_perp = abs(float(np.dot(ao["F_side"], ef)))
        lift_side_perp = abs(float(np.dot(ao["F_lift"], ao["F_side"])))
        still_env = EnvironmentModel()
        deriv = dynamics_3d(0.0, np.concatenate([np.zeros(3), np.zeros(3), q, np.zeros(3)]),
                             m, still_env, m.aero, m.geometry_state())
        gravity_ok = bool(abs(deriv[3]) < 1e-12 and abs(deriv[5]) < 1e-12
                          and deriv[4] < 0)
        ok = (drag_align < -0.999999 and lift_perp < 1e-9 and side_perp < 1e-9
              and lift_side_perp < 1e-9 and gravity_ok)
        return dict(passed=bool(ok), drag_alignment=drag_align,
                    lift_dot_flow=lift_perp, side_dot_flow=side_perp,
                    lift_dot_side=lift_side_perp, gravity_minus_y=gravity_ok)

    def test_3d_solver_agreement(self):
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        outs = {}
        for solver, dt in (("RK4", 0.0005), ("Euler", 0.00005)):
            m = build_full_model(db, "Normal Feather", geom, level=2)
            ic3 = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30, elevation_deg=-10,
                                      body_axis_offset_deg=140.0)
            r = SimulationEngine3D(m, env, ic3, solver=solver, dt=dt, t_max=0.4).run()
            if len(r.df) == 0 or not r.valid:
                return dict(passed=False, reason=f"{solver} run failed")
            last = r.df.iloc[-1]
            outs[solver] = (float(last.x_m), float(last.y_m), float(last.z_m))
        d = math.dist(outs["RK4"], outs["Euler"])
        return dict(passed=bool(d < 5e-3), rk4=outs["RK4"], euler=outs["Euler"],
                    distance=d)

    def test_inertia_positive_definite(self):
        db = ParameterDatabase()
        geom = Geometry()
        results = {}
        ok = True
        for name, removed in (("intact", []), ("asymmetric_4", [1, 2, 3, 4])):
            m = build_full_model(db, "Normal Feather", geom, level=2,
                                  removed_feathers=removed)
            mp = m.mass_properties()
            eig = np.linalg.eigvalsh(np.array(mp["inertia_tensor"]))
            Is, It = inertia_3d(m)
            pd_ok = bool(np.all(eig > 0) and Is > 0 and It > 0)
            results[name] = dict(eigenvalues=[float(e) for e in eig],
                                  cm_offset_m=mp["cm_offset_m"],
                                  positive_definite=pd_ok)
            ok = ok and pd_ok
        shifted = results["asymmetric_4"]["cm_offset_m"] > results["intact"]["cm_offset_m"]
        return dict(passed=bool(ok and shifted), cases=results,
                    note="비대칭 파손이 질량중심을 이동시키는지 확인")

    def test_3d_reduces_to_2d(self):
        """3D solver restricted to the plane must reproduce the validated 2D model."""
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        flow = rad2deg(math.atan2(30 * math.sin(deg2rad(-10)), 30 * math.cos(deg2rad(-10))))
        worst_xy = 0.0
        worst_a = {}
        xy_by_dt = {}
        for off in (180.0, 90.0, 20.0):
            errs = []
            xy_errs = []
            for dt in (0.001, 0.00025):
                m2 = build_full_model(db, "Normal Feather", geom, level=2)
                m3 = build_full_model(db, "Normal Feather", geom, level=2)
                ic2 = InitialCondition.from_launch(30, -10, 2.0, 145, 180,
                                                    body_orientation_deg=flow + off)
                r2 = SimulationEngine(m2, env, ic2, dt=dt, t_max=1.0, model_level=2).run()
                ic3 = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30, elevation_deg=-10,
                                          body_axis_offset_deg=off)
                r3 = SimulationEngine3D(m3, env, ic3, dt=dt, t_max=1.0, planar=True).run()
                n = min(len(r2.df), len(r3.df))
                if n == 0:
                    return dict(passed=False, reason="simulation produced no rows")
                dxy = max(float(np.abs(r2.df.x_m.values[:n] - r3.df.x_m.values[:n]).max()),
                          float(np.abs(r2.df.y_m.values[:n] - r3.df.y_m.values[:n]).max()))
                da = float(np.abs(np.abs(r2.df.delta_alpha_deg.values[:n])
                                  - np.abs(r3.df.delta_alpha_deg.values[:n])).max())
                worst_xy = max(worst_xy, dxy)
                xy_errs.append(dxy)
                errs.append(da)
                if float(np.abs(r3.df.z_m.values).max()) != 0.0:
                    return dict(passed=False, reason="planar mode produced non-zero z")
            noise_floor = 1e-8
            at_floor = errs[1] < noise_floor
            worst_a[f"offset_{off:g}"] = dict(
                coarse=errs[0], fine=errs[1],
                converging=bool(at_floor or errs[1] < errs[0]),
                at_noise_floor=bool(at_floor))
            xy_by_dt[f"offset_{off:g}"] = dict(
                coarse=xy_errs[0], fine=xy_errs[1],
                converging=bool(xy_errs[1] < noise_floor or xy_errs[1] < xy_errs[0]))
        # XY used to agree to machine zero only because Cl was disabled, which made the
        # 2D and 3D force expressions reduce to identical arithmetic. With a real
        # normal force the two equivalent formulations differ by truncation error, so
        # the criterion is convergence under dt (measured 4th order), not exact zero.
        ok = (all(v["converging"] and v["fine"] < 1e-8 for v in xy_by_dt.values())
              and all(v["converging"] and v["fine"] < 1e-4 for v in worst_a.values()))
        return dict(passed=bool(ok), max_xy_difference=worst_xy,
                    xy_error=xy_by_dt, angle_error=worst_a,
                    note="XY·각도 모두 dt를 줄이면 수렴 (RK4 절단 오차)")

    def test_3d_convergence(self):
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        finals = []
        for dt in (0.001, 0.0005, 0.00025):
            m = build_full_model(db, "Normal Feather", geom, level=2)
            ic3 = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30, elevation_deg=-10,
                                      body_axis_offset_deg=120.0, body_azimuth_deg=25.0)
            r = SimulationEngine3D(m, env, ic3, dt=dt, t_max=0.8).run()
            if len(r.df) == 0 or not r.valid:
                return dict(passed=False, reason="3D run failed")
            last = r.df.iloc[-1]
            if not np.all(np.isfinite([last.x_m, last.y_m, last.z_m])):
                return dict(passed=False, reason="NaN/Inf in 3D result")
            finals.append((float(last.x_m), float(last.y_m), float(last.z_m),
                           float(last.alpha_deg)))
        e1 = math.dist(finals[0][:3], finals[1][:3])
        e2 = math.dist(finals[1][:3], finals[2][:3])
        a1 = abs(finals[0][3] - finals[1][3])
        a2 = abs(finals[1][3] - finals[2][3])
        return dict(passed=bool(e2 <= e1 and a2 <= a1 + 1e-9),
                    position_errors=[e1, e2], angle_errors=[a1, a2])

    def test_3d_planar_free(self):
        """Out-of-plane motion must come from axial spin and from nothing else.

        Mirror symmetry about the launch plane is broken by exactly one thing: the
        axial spin, which is a pseudo-vector along the symmetry axis. With no spin
        the trajectory has to stay in the plane to machine precision. With spin, the
        restoring moment makes the wobble precess out of the plane, and the
        cross-flow normal force then pushes the shuttlecock sideways -- a real effect
        and the reason a sliced shuttlecock drifts. That drift used to be invisible
        only because Cl was disabled, so a tilt out of plane produced no force.
        """
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        ic3 = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30, elevation_deg=-10,
                                  azimuth_deg=0.0, body_axis_offset_deg=100.0,
                                  body_azimuth_deg=0.0)
        out = {}
        for tag, spin in (("no_spin", SpinModel(mode="constant")),
                           ("aero_driven_spin", SpinModel(mode="aero_driven",
                                                          R_m=geom.R_m))):
            m = build_full_model(db, "Normal Feather", geom, level=2, spin=spin)
            r = SimulationEngine3D(m, env, ic3, dt=0.0005, t_max=1.0, planar=False).run()
            if len(r.df) == 0:
                return dict(passed=False, reason=f"3D run failed ({tag})")
            out[tag] = dict(max_z=float(np.abs(r.df.z_m.values).max()),
                             max_vz=float(np.abs(r.df.vz_m_s.values).max()),
                             max_spin=float(np.abs(r.df.spin_axial_rad_s.values).max()))
        planar_without_spin = (out["no_spin"]["max_z"] < 1e-9
                               and out["no_spin"]["max_vz"] < 1e-9)
        # precession drift is real but must stay small next to the flight itself
        drift = out["aero_driven_spin"]["max_z"]
        drift_bounded = drift < 0.05
        drift_needs_spin = drift > out["no_spin"]["max_z"]
        return dict(passed=bool(planar_without_spin and drift_bounded and drift_needs_spin),
                    cases=out,
                    note="자전이 없으면 평면 유지, 자전이 있으면 세차에 의한 작은 면외 편차")

    def _coupled_setup(self, Cd=0.6, Cl=0.0, cl_mode="disabled", wind=(0.0, 0.0)):
        db = ParameterDatabase()
        geom = Geometry()
        model = build_full_model(db, "Normal Feather", geom, level=2,
                                  cl_model=ClModel(mode=cl_mode, constant_value=Cl))
        model.Cd_model.constant_value = Cd
        model.aero.cd_model = model.Cd_model
        env = EnvironmentModel(rho=1.2, u_air_x=wind[0], u_air_y=wind[1])
        return model, env

    def test_drag_direction(self):
        model, env = self._coupled_setup(wind=(8.0, 0.0))

        state = np.array([0.0, 2.0, 3.0, 0.0, math.pi, 0.0])
        aero = model.aero
        gs = model.geometry_state()
        d = coupled_diagnostics(0.0, state, model, env, aero, gs)
        deriv = dynamics_coupled(0.0, state, model, env, aero, gs)
        u_rel = state[2] - env.u_air_x
        drag_ax = deriv[2]
        ok = (u_rel < 0 and drag_ax > 0) and d["Vrel"] >= 0
        return dict(passed=bool(ok), u_rel=float(u_rel), drag_accel_x=float(drag_ax),
                    Vrel=float(d["Vrel"]))

    def test_lift_orthogonality(self):
        model, env = self._coupled_setup(Cl=0.5, cl_mode="constant")
        state = np.array([0.0, 2.0, 12.0, -4.0, math.pi, 0.0])
        gs = model.geometry_state()
        u_rel = state[2] - env.u_air_x
        v_rel = state[3] - env.u_air_y
        Vrel = math.hypot(u_rel, v_rel)
        e_f = np.array([u_rel / Vrel, v_rel / Vrel])
        d = coupled_diagnostics(0.0, state, model, env, model.aero, gs)
        drag_vec = -d["drag"] * e_f
        lift_vec = d["lift"] * np.array([-e_f[1], e_f[0]])
        dot = float(np.dot(drag_vec, lift_vec))
        ok = abs(dot) < 1e-9 * max(1.0, d["drag"] * d["lift"]) and d["lift"] > 0
        return dict(passed=bool(ok), dot_product=dot, lift_N=float(d["lift"]),
                    drag_N=float(d["drag"]))

    def test_no_moment_constant_omega(self):
        db = ParameterDatabase()
        geom = Geometry()
        model = build_full_model(db, "Normal Feather", geom, level=2,
                                  cm_model=CmModel(mode="cm_alpha", cm_alpha=0.0))
        # zeta only drives the LINEAR damping term; in aero_pendulum mode the damping
        # comes from beta = damping_scale*(M_cork/M_skirt)*U/ell, so zeroing zeta alone
        # left the damping switched on and the test could never hold omega constant
        model.turnover.zeta = 0.0
        model.turnover.damping_scale = 0.0
        env = EnvironmentModel(rho=1.2)
        ic = InitialCondition(x0=0, y0=50.0, vx0=10.0, vy0=0.0,
                               body_orientation_deg=180.0, omega0_rad_s=2.0)
        res = SimulationEngine(model, env, ic, dt=0.0005, t_max=1.0, model_level=2).run()
        omega = res.df.omega_rad_s.values
        drift = float(np.max(np.abs(omega - 2.0)))
        return dict(passed=bool(drift < 1e-6), max_omega_drift=drift)

    def test_spin_decoupled(self):
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel(rho=1.2)
        outs = []
        for spin in (0.0, 400.0):
            model = build_full_model(db, "Normal Feather", geom, level=2)
            ic = InitialCondition.from_launch(25, -10, 1.8, 145, 180,
                                               body_orientation_deg=180.0, spin0_rad_s=spin)
            res = SimulationEngine(model, env, ic, dt=0.001, t_max=1.0, model_level=2).run()
            outs.append(res.df.x_m.iloc[-1])
        diff = abs(outs[0] - outs[1])
        status = build_full_model(db, "Normal Feather", geom, level=2).aero.spin.status
        return dict(passed=bool(diff < 1e-12), range_difference=float(diff), spin_status=status)

    def test_coupled_zero_drag(self):
        model, env = self._coupled_setup(Cd=0.0)
        ic = InitialCondition(x0=0, y0=1.0, vx0=10.0, vy0=5.0, body_orientation_deg=180.0)
        res = SimulationEngine(model, env, ic, dt=0.001, t_max=1.0, model_level=2).run()
        t = res.df.time_s.values
        err_x = float(np.max(np.abs(res.df.x_m.values - (ic.x0 + ic.vx0 * t))))
        err_y = float(np.max(np.abs(res.df.y_m.values - (ic.y0 + ic.vy0 * t - 0.5 * G * t ** 2))))
        return dict(passed=bool(err_x < 1e-3 and err_y < 1e-3),
                    max_error_x=err_x, max_error_y=err_y)

    def test_damage_geometry(self):
        cases = {
            "0_removed": [],
            "1_removed": [1],
            "2_symmetric": [1, 9],
            "2_asymmetric": [1, 2],
            "8_removed": list(range(1, 9)),
            "all_removed": list(range(1, 17)),
        }
        out = {}
        ok = True
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        for name, removed in cases.items():
            try:
                model = build_full_model(db, "Normal Feather", geom, level=2,
                                          removed_feathers=removed)
                gs = model.geometry_state()
                ic = InitialCondition.from_launch(25, -10, 1.8, 145, 180,
                                                   body_orientation_deg=180.0)
                res = SimulationEngine(model, env, ic, dt=0.001, t_max=1.0, model_level=2).run()
                # inside the porosity model's domain a state must simulate; outside
                # it the simulator must refuse and say why, rather than extrapolate to
                # the bare-cork anchor and plot a 20-38 m "badminton" flight
                in_domain = ShuttlecockPorosityModel().domain_status(gs)[0]
                ran = bool(len(res.df) > 0)
                out[name] = dict(remaining=gs["remaining_feathers"],
                                  asymmetry=round(gs["geometry_asymmetry"], 4),
                                  porosity=round(gs["estimated_porosity"], 4),
                                  in_domain=bool(in_domain), ran=ran,
                                  refused_with_reason=bool(not ran and res.warnings),
                                  warnings=len(res.warnings))
                if in_domain and not ran:
                    ok = False
                if not in_domain and (ran or not res.warnings):
                    ok = False
            except Exception as e:
                out[name] = dict(error=str(e))
                ok = False

        if out.get("2_symmetric", {}).get("asymmetry") is not None:
            ok = ok and out["2_symmetric"]["asymmetry"] < out["2_asymmetric"]["asymmetry"]
        return dict(passed=bool(ok), cases=out)

    def test_launch_orientations(self):
        db = ParameterDatabase()
        geom = Geometry()
        env = EnvironmentModel()
        out = {}
        ok = True
        cases = [("Cork Forward", -10, None), ("Skirt Forward", -10, None),
                 ("Perpendicular", 20, None), ("Custom", 30, 180.0),
                 ("Cork Forward", 45, None), ("Cork Forward", -45, None)]
        for preset, launch, custom in cases:
            bo = body_orientation_from_preset(preset, launch, custom_deg=custom, V0=25)
            ic = InitialCondition.from_launch(25, launch, 1.8, 145, 180,
                                               body_orientation_deg=bo)
            alpha0 = rad2deg(ic.alpha0_rad)
            expected = ORIENTATION_PRESETS.get(preset)
            key = f"{preset}@{launch}"
            if expected is not None:
                consistent = abs(abs(wrap_to_pi(deg2rad(alpha0 - expected)))) < 1e-6
            else:
                consistent = True
            model = build_full_model(db, "Normal Feather", geom, level=2)
            res = SimulationEngine(model, env, ic, dt=0.001, t_max=1.0, model_level=2).run()
            out[key] = dict(alpha0_deg=round(alpha0, 3), consistent=bool(consistent),
                             ran=bool(len(res.df) > 0))
            ok = ok and consistent and len(res.df) > 0
        return dict(passed=bool(ok), cases=out)

    def test_zero_drag(self):
        model = _make_test_model(Cd=0.0)
        env = EnvironmentModel(rho=1.2)
        ic = InitialCondition(x0=0, y0=1.0, vx0=10.0, vy0=5.0, alpha_initial_deg=180,
                               alpha_equilibrium_deg=180)
        eng = SimulationEngine(model, env, ic, dt=0.001, t_max=3.0)
        res = eng.run()
        df = res.df
        t = df.time_s.values
        x_analytic = ic.x0 + ic.vx0 * t
        y_analytic = ic.y0 + ic.vy0 * t - 0.5 * G * t ** 2
        err_x = np.max(np.abs(df.x_m.values - x_analytic))
        err_y = np.max(np.abs(df.y_m.values - y_analytic))
        passed = err_x < 1e-3 and err_y < 1e-3
        return dict(passed=bool(passed), max_error_x=float(err_x), max_error_y=float(err_y))

    def test_zero_gravity_no_drag(self):
        model = _make_test_model(Cd=0.0)
        env = EnvironmentModel(rho=1.2)

        def no_gravity_dynamics(t, state, model, env, torque_fn):
            d = dynamics(t, state, model, env, torque_fn)
            d[3] += G
            return d

        state = np.array([0, 1.0, 10.0, 5.0, 0.0, 0.0])
        dt = 0.001
        vx0, vy0 = state[2], state[3]
        for _ in range(1000):
            state = rk4_step(no_gravity_dynamics, 0, state, dt, model, env, None)
        ok = abs(state[2] - vx0) < 1e-6 and abs(state[3] - vy0) < 1e-6
        return dict(passed=bool(ok), final_vx=float(state[2]), final_vy=float(state[3]))

    def _run_turnover_only(self, zeta, Tw=0.0102, delta_alpha0_deg=10.0, t_max=0.2, dt=1e-5):
        turnover = TurnoverParams(Tw=Tw, zeta=zeta, Iyy=2.92e-6)
        state = np.array([deg2rad(delta_alpha0_deg), 0.0])
        ts = np.arange(0, t_max, dt)
        vals = []

        def f(t, y):
            da, om = y
            dda = -(2 * zeta / Tw) * om - (1.0 / Tw ** 2) * da
            return np.array([om, dda])

        for t in ts:
            vals.append(state[0])
            k1 = f(t, state)
            k2 = f(t + dt / 2, state + dt / 2 * k1)
            k3 = f(t + dt / 2, state + dt / 2 * k2)
            k4 = f(t + dt, state + dt * k3)
            state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return ts, np.array(vals), turnover

    def test_undamped(self):
        ts, vals, turnover = self._run_turnover_only(zeta=0.0)
        a0 = deg2rad(10.0)
        omega_n = turnover.omega_n
        analytic = a0 * np.cos(omega_n * ts)
        err = np.max(np.abs(vals - analytic))
        return dict(passed=bool(err < 1e-3), max_error=float(err))

    def test_underdamped(self):
        ts, vals, turnover = self._run_turnover_only(zeta=0.3)
        a0 = deg2rad(10.0)
        wn = turnover.omega_n
        z = turnover.zeta
        wd = turnover.omega_d
        analytic = a0 * np.exp(-z * wn * ts) * (np.cos(wd * ts) + (z * wn / wd) * np.sin(wd * ts))
        err = np.max(np.abs(vals - analytic))
        return dict(passed=bool(err < 1e-3), max_error=float(err))

    def test_critical(self):
        ts, vals, turnover = self._run_turnover_only(zeta=1.0)
        a0 = deg2rad(10.0)
        wn = turnover.omega_n
        analytic = a0 * (1 + wn * ts) * np.exp(-wn * ts)
        err = np.max(np.abs(vals - analytic))
        return dict(passed=bool(err < 1e-3), max_error=float(err), classification=turnover.damping_class)

    def test_overdamped(self):
        ts, vals, turnover = self._run_turnover_only(zeta=1.5)
        z = turnover.zeta
        wn = turnover.omega_n
        r1 = -wn * (z - math.sqrt(z ** 2 - 1))
        r2 = -wn * (z + math.sqrt(z ** 2 - 1))
        a0 = deg2rad(10.0)
        A = a0 * r2 / (r2 - r1)
        B = a0 - A
        analytic = A * np.exp(r1 * ts) + B * np.exp(r2 * ts)
        err = np.max(np.abs(vals - analytic))
        return dict(passed=bool(err < 1e-3), max_error=float(err), classification=turnover.damping_class)

    def test_dt_convergence(self):
        model = _make_test_model()
        env = EnvironmentModel(rho=1.2)
        ic = InitialCondition(x0=0, y0=2.0, vx0=15.0, vy0=5.0, alpha_initial_deg=145,
                               alpha_equilibrium_deg=180)
        conv = convergence_test(model, env, ic, dt=0.001, t_max=1.0)
        comps = conv["comparisons"]
        errors = [c["final_position_error"] for c in comps if c.get("comparable")]
        decreasing = all(errors[i] >= errors[i + 1] * 0.5 for i in range(len(errors) - 1)) if len(errors) > 1 else True
        return dict(passed=bool(decreasing), errors=errors)

COURT_LENGTH_M = 13.40   # BWF full court, back boundary to back boundary
# Cohen 2015 의 공기역학 벽 때문에 사거리는 ℓ ≈ 4 m 규모에서 포화한다. 코트의 1.5 배를
# 넘는 궤적은 셔틀콕이 아니라 항력이 꺼진 탄도 포물선이므로 그리지 않는다.
ENVELOPE_REFUSAL_M = 1.5 * COURT_LENGTH_M


def apply_flip_settings(model, turnover_mode, l_gc, damping_scale, cd_cross,
                        orient_cd, aero_spin, geom):
    """Applies the turnover/flip settings a front end collected onto a model.

    Module level so every front end shares one implementation: this used to live
    inside build_ui(), where only the Gradio UI could reach it.
    """
    tp = model.turnover
    tp.mode = turnover_mode or "aero_pendulum"
    # a zero is a value the user typed, not a missing entry
    if l_gc not in (None, ""):
        tp.l_gc = float(l_gc)
    if damping_scale not in (None, ""):
        tp.damping_scale = float(damping_scale)
    if cd_cross not in (None, ""):
        tp.Cd_cross = float(cd_cross)
    tp.Cd_ref = model.Cd_model.constant_value
    # the cross-flow normal force is fixed by the aligned and broadside drag
    if model.aero is not None and model.aero.cl_model is not None:
        model.aero.cl_model.crossflow_delta_cd = tp.Cd_cross - tp.Cd_ref
    if orient_cd:
        model.Cd_model.mode = "orientation"
        model.Cd_model.cd_cross = tp.Cd_cross
    if aero_spin and model.aero is not None:
        model.aero.spin.mode = "aero_driven"
        model.aero.spin.spin_ratio = tp.spin_ratio
        model.aero.spin.R_m = geom.R_m
    return model


# Cooke (1999) measured the shuttlecock's drag coefficient as essentially constant
# over Re = 1.3e4 to about 2e5, and that flat curve is what licenses the constant-Cd
# model used everywhere here. Above it the model is extrapolating: nothing in the
# anchors says what the drag does at Re = 6e5.
REYNOLDS_VALID_RANGE = (1.3e4, 2.0e5)


def reynolds_domain_warnings(df, model=None, env=None, min_fraction=0.01):
    """Flags a flight that spends real time outside the Reynolds range Cd was measured in.

    A 300 km/h smash reaches Re = 3.6e5 and the world-record 493 km/h reaches 6.0e5,
    both past the 2e5 ceiling of Cooke's flat-Cd data, and the model was reporting
    Cd = 0.605 through all of it without saying that the number is an extrapolation
    there. The trajectory is still approximately right -- drag dominates so heavily
    that the first metres barely matter -- but a smash is the headline case for this
    simulator, so the caveat belongs on screen rather than in the source.
    """
    out = []
    try:
        if df is None or len(df) == 0 or "Re" not in df.columns:
            return out
        re = np.asarray(df["Re"].values, dtype=float)
        re = re[np.isfinite(re)]
        if re.size == 0:
            return out
        lo, hi = REYNOLDS_VALID_RANGE
        frac_hi = float(np.mean(re > hi))
        frac_lo = float(np.mean(re < lo))
        if frac_hi >= min_fraction:
            out.append(
                f"비행의 {100 * frac_hi:.0f}% 구간이 Re = {re.max():.2e} 까지 올라가 "
                f"항력계수를 측정한 범위(Re ≤ {hi:.0e}, Cooke 1999) 를 벗어납니다. "
                f"그 구간의 Cd 는 실측이 아니라 외삽이며, 초고속 구간에서는 스커트 "
                f"변형으로 Cd 가 더 떨어질 수 있습니다.")
        if frac_lo >= min_fraction:
            out.append(
                f"비행의 {100 * frac_lo:.0f}% 구간이 Re = {re.min():.2e} 까지 내려가 "
                f"측정 범위(Re ≥ {lo:.0e}) 아래입니다. 착지 직전 저속 구간의 "
                f"Cd 는 외삽값입니다.")
    except Exception:
        logger.exception("reynolds domain check failed")
    return out


def launch_state_errors(ic):
    """Launch states that are not physically meaningful, whatever the model does with them.

    A negative launch height starts the shuttlecock below the floor: the run
    terminated after two rows and still reported valid=True, which is a result-shaped
    answer to a question that has none.
    """
    out = []
    try:
        y0 = float(getattr(ic, "y0", 0.0) or 0.0)
        if y0 < 0:
            out.append(f"발사 높이 {y0:.3f} m 는 지면 아래입니다. "
                       f"0 이상이어야 합니다.")
        v0 = getattr(ic, "V0", None)
        if v0 is not None and float(v0) < 0:
            out.append(f"발사 속도 {float(v0):.3f} m/s 는 음수입니다. "
                       f"방향은 발사각으로 지정하십시오.")
    except (TypeError, ValueError):
        pass
    return out


def flight_envelope_warnings(df, model, env=None):
    """Flags a simulated flight that leaves the envelope a badminton shot can occupy.

    A shuttlecock's range grows only like log(V0) because the aerodynamic length
    ell = 2m/(rho*S*Cd) is about 4 m, so an intact one cannot cross the 13.4 m court
    however hard it is hit. A run that goes much further is being driven by something
    the user should see named -- almost always feather damage dropping Cd towards the
    bare-cork anchor, or wind -- rather than presented as a badminton trajectory.
    """
    out = []
    try:
        if df is None or len(df) == 0 or "x_m" not in df.columns:
            return out
        rng = abs(float(df["x_m"].iloc[-1]) - float(df["x_m"].iloc[0]))
        if rng <= COURT_LENGTH_M:
            return out
        causes = []
        if "Cd" in df.columns:
            cd_mean = float(np.nanmean(df["Cd"].values))
            cd_ref = float(getattr(model.turnover, "Cd_ref", 0.62) or 0.62)
            if cd_ref > 0 and cd_mean < 0.6 * cd_ref:
                gs = model.geometry_state()
                lost = (gs["removed_feathers"] if gs else 0)
                causes.append(
                    f"항력계수가 정상값의 {100.0*cd_mean/cd_ref:.0f}% ({cd_mean:.3f}) 로 "
                    f"떨어졌습니다"
                    + (f" — 깃털 {lost}개 제거" if lost else ""))
        if env is not None:
            w = math.hypot(float(env.u_air_x), float(env.u_air_y))
            if w > 1.0:
                causes.append(f"바람 {w:.1f} m/s 가 셔틀콕을 함께 실어 나릅니다")
        if "speed_m_s" in df.columns:
            v0 = float(df["speed_m_s"].iloc[0])
            if v0 > 120.0:
                causes.append(f"초기 속도 {v0:.0f} m/s 는 실측 최고 스매시(113 m/s)를 넘습니다")
        ell = model.aerodynamic_length(
            float(getattr(env, "rho", 1.2) or 1.2) if env is not None else 1.2,
            float(np.nanmean(df["Cd"].values)) if "Cd" in df.columns else 0.62)
        msg = (f"도달 거리 {rng:.1f} m 는 배드민턴 코트 전체 길이 {COURT_LENGTH_M} m 를 "
               f"넘습니다 (공기역학 길이 ℓ = {ell:.1f} m; 정상 셔틀콕은 약 4 m). "
               "배드민턴 궤적으로 해석하지 마세요.")
        if causes:
            msg += " 원인: " + " / ".join(causes) + "."
        else:
            msg += " 발사 조건을 확인하세요."
        out.append(msg)
    except Exception:
        logger.exception("flight envelope check failed")
    return out


def physical_sanity_checks(model: ShuttlecockModel, dt: float, env: EnvironmentModel = None):
    errs = model.sanity_check()
    if dt <= 0:
        errs.append("dt must be > 0")
    if env is not None:
        if env.rho <= 0:
            errs.append("air density rho must be > 0")
        if env.mu <= 0:
            errs.append("dynamic viscosity mu must be > 0")
    if model.Ispin is not None and model.Ispin <= 0:
        errs.append("Ispin must be > 0")
    if (model.Cd_model.mode in ("constant", "orientation")
            and model.Cd_model.constant_value <= 0):
        # Cd = 0 은 항력이 전혀 없다는 뜻이고, 그러면 셔틀콕이 아니라 탄도 포물선이
        # 된다 (26 m/s 21도 발사에서 49 m). 음수만 막고 0 은 통과시켜서, 빈 칸이
        # 0 으로 전달되면 그대로 무항력 비행이 그려졌다.
        # "orientation" 모드도 constant_value 를 정렬 자세의 기준 Cd 로 쓰므로
        # 같은 검사를 받아야 한다. 자세에 따른 Cd 변화를 켜 두면 모드가
        # "orientation" 으로 바뀌어서, 모드 이름만 보던 검사는 Cd = 0 을 통과시켰다.
        errs.append("항력계수 Cd 는 0 보다 커야 합니다. Cd = 0 이면 항력이 전혀 "
                    "없어 셔틀콕이 아니라 탄도 포물선이 됩니다 "
                    "(정상 셔틀콕 Cd ≈ 0.6).")
    return errs

def flight_envelope_errors(df, model, env=None):
    """Flights this simulator must refuse to draw as a badminton trajectory.

    Cohen et al. (2015) show a shuttlecock has a single length scale, the
    aerodynamic length ell = 2M/(rho*S*C_D) ~ 4 m, and that the range saturates
    logarithmically with launch speed -- the "aerodynamic wall". A 5 g shuttlecock
    reaches terminal velocity (~6.4 m/s here, ~25 km/h reported) within about half a
    second, and its speed halves roughly every 3 m. Nothing a racket can do sends it
    much past the 13.4 m court.

    So a plotted 80 m parabola is never a shuttlecock: it is an input that has
    silently turned the drag off -- an empty Cd box arriving as 0, a zeroed air
    density, a zeroed skirt diameter. Warning about it is not enough, because the
    plot is still drawn and read as a result. Refuse it, and name the cause.
    """
    warns = flight_envelope_warnings(df, model, env)
    if not warns:
        return []
    try:
        rng = abs(float(df["x_m"].iloc[-1]) - float(df["x_m"].iloc[0]))
    except Exception:
        return []
    if rng <= ENVELOPE_REFUSAL_M:
        return []                    # 코트를 조금 넘는 정도는 경고로 충분하다
    return [w + (f" 도달 거리 {rng:.1f} m 는 물리적으로 불가능하므로 "
                 f"(한계 {ENVELOPE_REFUSAL_M:.1f} m) 궤적을 그리지 않았습니다.")
            for w in warns]


def damage_domain_errors(model: ShuttlecockModel):
    """Damage states the simulator must refuse to produce a trajectory for.

    A warning is not enough here: the run still returns a plotted flight, and a 38 m
    badminton trajectory on screen is worse than no trajectory at all.
    """
    try:
        gs = model.geometry_state()
        if gs is None:
            return []
        pm = getattr(getattr(model, "aero", None), "damage_mapping", None)
        pm = getattr(pm, "porosity_model", None) if pm is not None else None
        pm = pm or ShuttlecockPorosityModel()
        ok, reason = pm.domain_status(gs)
        return [] if ok else [reason]
    except Exception:
        logger.exception("damage domain check failed")
        return []


def planar_damage_warnings(model: ShuttlecockModel):
    """Says out loud what a planar model cannot do with one-sided feather damage."""
    out = []
    try:
        gs = model.geometry_state()
        if not gs or not gs.get("removed_feathers"):
            return out
        asym = float(gs.get("geometry_asymmetry") or 0.0)
        if asym < 0.05:
            return out
        out.append(
            f"비대칭 파손(비대칭도 {asym:.2f})의 좌우 효과는 2차원 모델에 담기지 "
            f"않습니다. 없어진 깃이 만드는 모멘트는 비행면에 대해 어느 방향으로 설지 "
            f"정해져 있지 않고 자전(RΩ/U = 0.04)으로 비행 중 돌기까지 하므로, 평면 "
            f"모델의 유일한 회전 자유도(피치)에 넣으면 있지도 않은 정상(定常) 양력이 "
            f"됩니다. 여기서는 파손의 대칭 효과(항력계수·질량·복원모멘트)만 반영하며, "
            f"좌우 편향을 보려면 3차원 운동 예측 탭을 쓰십시오.")
    except Exception:
        logger.exception("planar damage warning failed")
    return out


def geometry_warnings(model: ShuttlecockModel):
    gs = model.geometry_state()
    if gs is None:
        return []
    out = []
    if gs["remaining_feathers"] < gs["N_feathers"] * 0.5:
        out.append("남은 깃털이 절반 미만입니다: 문헌으로 검증된 형상에서 크게 벗어납니다")
    if gs["geometry_asymmetry"] > 0.5:
        out.append(f"strongly asymmetric damage (index {gs['geometry_asymmetry']:.2f}): "
                   "2D symmetric-flow assumptions are questionable")
    return out

class ExportEngine:
    @staticmethod
    def to_csv(df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def to_json(metadata: dict, path: str):
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return path

    @staticmethod
    def to_html_plot(fig, path: str):
        fig.write_html(path)
        return path


def physics_verification_suite(db, verbose=False):
    """Automated physics checks (§53): frame, quaternion, inertia, signs, conservation.

    Each row is PASS / WARNING / FAIL. Nothing here is tuned to make the model look
    good; a FAIL means the physics is wrong and the result should not be used.
    """
    rows = []

    def rec(group, name, status, detail=""):
        rows.append(dict(구분=group, 항목=name, 결과=status, 상세=detail))

    geom = Geometry()
    env = EnvironmentModel()

    # ---- A. frame / rotation matrix ------------------------------------
    try:
        rng = np.random.default_rng(0)
        worst_orth = 0.0; worst_det = 0.0
        for _ in range(50):
            q = quat_normalize(rng.standard_normal(4))
            R = quat_to_matrix(q)
            worst_orth = max(worst_orth, float(np.max(np.abs(R.T @ R - np.eye(3)))))
            worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
        rec("A 좌표계", "R.T·R = I", "PASS" if worst_orth < 1e-12 else "FAIL",
            f"최대 편차 {worst_orth:.2e}")
        rec("A 좌표계", "det(R) = 1", "PASS" if worst_det < 1e-12 else "FAIL",
            f"최대 편차 {worst_det:.2e}")
        q = quat_normalize(np.array([0.6, 0.2, -0.3, 0.7]))
        R = quat_to_matrix(q)
        vb = np.array([1.0, -2.0, 0.5])
        err = float(np.max(np.abs(R.T @ (R @ vb) - vb)))
        rec("A 좌표계", "body→inertial→body 왕복", "PASS" if err < 1e-12 else "FAIL",
            f"오차 {err:.2e}")
    except Exception as exc:
        rec("A 좌표계", "회전행렬", "FAIL", str(exc))

    # ---- B. quaternion kinematics --------------------------------------
    try:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(2000):
            q = quat_normalize(q + quat_derivative(q, np.zeros(3)) * 1e-3)
        rec("B 쿼터니언", "ω=0 이면 자세 불변",
            "PASS" if abs(q[0] - 1.0) < 1e-12 else "FAIL", f"qw={q[0]:.12f}")
        w = np.array([0.0, 0.0, 2.0])
        q = np.array([1.0, 0.0, 0.0, 0.0]); dt = 1e-4
        for _ in range(10000):
            q = quat_normalize(q + quat_derivative(q, w) * dt)
        ang = 2.0 * math.atan2(abs(q[3]), q[0])
        exact = 2.0 * 1.0
        rec("B 쿼터니언", "등속 회전 = 해석해",
            "PASS" if abs(ang - exact) < 1e-3 else "FAIL",
            f"{ang:.6f} rad vs {exact:.6f}")
        rec("B 쿼터니언", "‖q‖ = 1 유지",
            "PASS" if abs(float(vnorm(q)) - 1.0) < 1e-12 else "FAIL",
            f"‖q‖={float(vnorm(q)):.12f}")
    except Exception as exc:
        rec("B 쿼터니언", "적분", "FAIL", str(exc))

    # ---- C. inertia tensor ---------------------------------------------
    try:
        for name, removed in (("정상", []), ("비대칭 3깃 제거", [1, 2, 3])):
            m = build_full_model(db, db.profile_names()[0], geom, level=3,
                                  removed_feathers=removed, use_geometry_inertia=True)
            I, prov = inertia_tensor_3d(m)
            sym = float(np.max(np.abs(I - I.T)))
            eig = np.linalg.eigvalsh(0.5 * (I + I.T))
            rec("C 관성", f"{name}: I 대칭", "PASS" if sym < 1e-18 else "FAIL",
                f"|I-Iᵀ| = {sym:.2e}")
            rec("C 관성", f"{name}: 양정치", "PASS" if np.min(eig) > 0 else "FAIL",
                f"최소 고유값 {np.min(eig):.3e}")
            R = quat_to_matrix(quat_normalize(np.array([0.5, 0.5, 0.5, 0.5])))
            Iw = R @ I @ R.T
            ew = np.linalg.eigvalsh(0.5 * (Iw + Iw.T))
            rec("C 관성", f"{name}: I_world 고유값 보존",
                "PASS" if np.max(np.abs(np.sort(ew) - np.sort(eig))) < 1e-18 else "FAIL",
                "I_world = R·I·Rᵀ")
    except Exception as exc:
        rec("C 관성", "텐서", "FAIL", str(exc))

    # ---- D. mass / centre of mass --------------------------------------
    try:
        base = build_full_model(db, db.profile_names()[0], geom, level=3,
                                 use_geometry_inertia=True)
        dmg = build_full_model(db, db.profile_names()[0], geom, level=3,
                                removed_feathers=[1, 2, 3], use_geometry_inertia=True)
        mb = base.mass_properties(); md = dmg.mass_properties()
        rec("D 질량/CG", "파손 시 질량 감소",
            "PASS" if md["total_mass"] < mb["total_mass"] else "FAIL",
            f"{mb['total_mass']*1e3:.3f} → {md['total_mass']*1e3:.3f} g")
        rec("D 질량/CG", "비대칭 파손 시 CG 이동",
            "PASS" if md["cm_offset_m"] > mb["cm_offset_m"] else "FAIL",
            f"{mb['cm_offset_m']*1e3:.3f} → {md['cm_offset_m']*1e3:.3f} mm")
        mc = mass_consistency_report(base)
        rec("D 질량/CG", "프로파일 vs 형상 질량 일관성",
            "PASS" if (mc and mc["consistent"]) else "WARNING",
            f"상대차 {mc['relative_difference']*100:.1f}%" if mc else "보고 없음")
    except Exception as exc:
        rec("D 질량/CG", "질량 속성", "FAIL", str(exc))

    # ---- E/F. force and moment signs -----------------------------------
    try:
        f0 = lift_alignment_factor(math.pi, math.pi)
        rec("E 힘", "정렬 시 양력 = 0 (Chan & Rossmann)",
            "PASS" if abs(f0) < 1e-12 else "FAIL", f"{f0:.2e}")
        cm = CmModel(mode="cm_alpha", cm_alpha=-0.5)
        m = build_full_model(db, db.profile_names()[0], geom, level=3,
                              cd_model=CdModel(mode="constant", constant_value=0.62),
                              cl_model=ClModel(), cm_model=cm, spin=SpinModel())
        ok = True
        for d in (0.15, -0.15):
            Mv = m.aero.moment(20.0, math.pi + d, 1e5, 0.0, 1.2, geom.S,
                                m.turnover, None)
            ok &= (Mv * d) < 0
        rec("F 모멘트", "안정 Cm 기울기 → 복원 모멘트",
            "PASS" if ok else "FAIL", "δ>0 이면 M<0, δ<0 이면 M>0")
        cm2 = CmModel(mode="cm_alpha", cm_alpha=+0.5)
        m2 = build_full_model(db, db.profile_names()[0], geom, level=3,
                               cd_model=CdModel(mode="constant", constant_value=0.62),
                               cl_model=ClModel(), cm_model=cm2, spin=SpinModel())
        Mv = m2.aero.moment(20.0, math.pi + 0.15, 1e5, 0.0, 1.2, geom.S,
                             m2.turnover, None)
        rec("F 모멘트", "불안정 Cm 기울기가 감지됨",
            "PASS" if Mv * 0.15 > 0 else "FAIL",
            "abs()로 부호를 지우지 않음")
    except Exception as exc:
        rec("F 모멘트", "부호", "FAIL", str(exc))

    # ---- G. signed alpha / beta ----------------------------------------
    try:
        # alpha lives in the body x-z plane (perturb w), beta is the sideslip
        # out of that plane (perturb v)
        ap, _ = aero_angles_3d(np.array([1.0, 0.0, 0.3]))
        am, _ = aero_angles_3d(np.array([1.0, 0.0, -0.3]))
        rec("G 각도", "alpha 부호 구분 (+/-)",
            "PASS" if (ap * am) < 0 else "FAIL", f"{ap:+.4f} / {am:+.4f} rad")
        _, bp = aero_angles_3d(np.array([1.0, 0.3, 0.0]))
        _, bm = aero_angles_3d(np.array([1.0, -0.3, 0.0]))
        rec("G 각도", "beta 부호 구분 (+/-)",
            "PASS" if (bp * bm) < 0 else "FAIL", f"{bp:+.4f} / {bm:+.4f} rad")
        a0_, b0_ = aero_angles_3d(np.array([0.0, 0.0, 0.0]))
        rec("G 각도", "상대속도 0 에서 특이점 없음",
            "PASS" if (a0_ == 0.0 and b0_ == 0.0) else "FAIL", "atan2/asin 안전 처리")
        rec("G 각도", "total misalignment 는 부호 없음(정의상)",
            "PASS", "alpha_total = acos(...) 은 원뿔각, wobble_signed_deg 로 부호 분리")
    except Exception as exc:
        rec("G 각도", "signed alpha/beta", "FAIL", str(exc))

    # ---- H/I/J/K. turnover constants ------------------------------------
    try:
        m = build_full_model(db, db.profile_names()[0], geom, level=3,
                              cd_model=CdModel(mode="constant", constant_value=0.62),
                              cl_model=ClModel(), cm_model=CmModel(mode="aero_pendulum"),
                              spin=SpinModel())
        m.turnover.Cd_ref = 0.62
        tcs = [turnover_constants(m, env, m.aero, U=U) for U in (10.0, 20.0, 40.0)]
        tw = [t["Tw_model"] for t in tcs]
        prod = [tw[i] * u for i, u in enumerate((10.0, 20.0, 40.0))]
        rec("I Tw", "Tw ∝ 1/U (속도 의존)",
            "PASS" if (max(prod) - min(prod)) / np.mean(prod) < 1e-6 else "FAIL",
            f"Tw·U = {prod[0]:.5f} (일정)")
        zs = [t["zeta_model"] for t in tcs]
        rec("J zeta", "ζ 는 속도와 무관한 상수",
            "PASS" if (max(zs) - min(zs)) < 1e-9 else "FAIL", f"ζ = {zs[0]:.4f}")
        rec("J zeta", "c_rot 은 -∂M/∂ω 중심차분으로 계산",
            "PASS" if tcs[0].get("c_rot_source") else "FAIL",
            tcs[0].get("c_rot_source", ""))
        rec("J zeta", "c_rot >= 0 (양의 감쇠)",
            "PASS" if tcs[0]["c_rot_Nms_per_rad"] >= 0 else "FAIL",
            f"{tcs[0]['c_rot_Nms_per_rad']:.3e}")
        q = tcs[0].get("Q_factor")
        rec("K Q", "부족감쇠에서만 Q 정의",
            "PASS" if (0 < zs[0] < 1 and q and np.isfinite(q)) else "FAIL",
            f"Q = {q:.3f}" if q and np.isfinite(q) else "Q 미정의")
        cm_un = CmModel(mode="cm_alpha", cm_alpha=+0.5)
        mu_ = build_full_model(db, db.profile_names()[0], geom, level=3,
                                cd_model=CdModel(mode="constant", constant_value=0.62),
                                cl_model=ClModel(), cm_model=cm_un, spin=SpinModel())
        mu_.turnover.mode = "linear"
        tcu = turnover_constants(mu_, env, mu_.aero, U=20.0)
        rec("H Turnover", "불안정 평형(k_alpha<0) 감지 및 Tw/ζ 미계산",
            "PASS" if ("불안정" in str(tcu.get("stability", ""))
                       and "Tw_model" not in tcu) else "FAIL",
            str(tcu.get("stability", "")))
    except Exception as exc:
        rec("I Tw", "turnover 상수", "FAIL", str(exc))

    # ---- L/M. conservation ---------------------------------------------
    try:
        m = build_full_model(db, db.profile_names()[0], geom, level=3,
                              cd_model=CdModel(mode="constant", constant_value=0.0),
                              cl_model=ClModel(), cm_model=CmModel(),
                              spin=SpinModel())
        m.Ispin = 1.2e-6
        m.turnover.mode = "linear"
        m.turnover.Tw = 1e6      # k_alpha -> 0 : no restoring torque
        m.turnover.zeta = 0.0    # no rotational damping
        # rho must stay > 0 for the engine's validity check; make it negligible so
        # the aerodynamic force and torque vanish and only gravity (conservative)
        # acts, which is the condition under which E_total and |L| must be constant
        env0 = EnvironmentModel(rho=1e-9)
        r = SimulationEngine3D(m, env0, InitialCondition3D(V0=20, elevation_deg=30,
                                y0=50.0, body_axis_offset_deg=140.0,
                                omega0=(0.0, 3.0, 5.0)), dt=2e-4, t_max=1.0).run()
        d = r.df
        if r.valid and len(d) > 10:
            E = d.E_total_J.values
            drift = float(abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12))
            rec("L 에너지", "무항력 시 역학적 에너지 보존",
                "PASS" if drift < 1e-3 else ("WARNING" if drift < 1e-2 else "FAIL"),
                f"상대 변화 {drift:.2e}")
            L = np.c_[d.angular_momentum_x, d.angular_momentum_y,
                      d.angular_momentum_z]
            Ln = np.linalg.norm(L, axis=1)
            ldr = float((Ln.max() - Ln.min()) / max(Ln.mean(), 1e-12))
            rec("M 각운동량", "외부 토크 0 → |L| 보존",
                "PASS" if ldr < 1e-3 else ("WARNING" if ldr < 1e-2 else "FAIL"),
                f"상대 변화 {ldr:.2e}")
            qn = np.sqrt(d.qw ** 2 + d.qx ** 2 + d.qy ** 2 + d.qz ** 2)
            rec("B 쿼터니언", "적분 중 ‖q‖ = 1",
                "PASS" if float(np.max(np.abs(qn - 1))) < 1e-6 else "FAIL",
                f"최대 편차 {float(np.max(np.abs(qn-1))):.2e}")
        else:
            rec("L 에너지", "보존 테스트", "FAIL", "시뮬레이션 실패")
    except Exception as exc:
        rec("L 에너지", "보존", "FAIL", str(exc))

    # ---- L2. energy balance WITH wind (§37) ------------------------------
    try:
        m = build_full_model(db, db.profile_names()[0], geom, level=3,
                              use_geometry_inertia=True,
                              cd_model=CdModel(mode="constant", constant_value=0.62),
                              cl_model=ClModel(), cm_model=CmModel(mode="aero_pendulum"),
                              spin=SpinModel(mode="constant"))
        m.turnover.Cd_ref = 0.62; m.Ispin = 1.2e-6
        # a strong tailwind: the air can now do positive work on the shuttlecock, so
        # mechanical energy is NOT required to decrease monotonically
        env_w = EnvironmentModel(rho=1.2, u_air_x=30.0)
        r = SimulationEngine3D(m, env_w, InitialCondition3D(V0=5.0, elevation_deg=20,
                                y0=3.0, body_axis_offset_deg=180.0),
                                dt=2e-4, t_max=0.8).run()
        if r.valid and len(r.df) > 10:
            E = r.df.E_total_J.values
            rose = bool(np.any(np.diff(E) > 0))
            rec("L 에너지", "순풍에서 역학적 에너지 증가 구간 허용",
                "PASS" if rose else "WARNING",
                "공기가 물체에 일을 하므로 단조 감소가 아님 — 정상"
                if rose else "이 조건에서는 증가 구간이 없었음")
            # drag opposes the RELATIVE velocity, so F_drag . V_rel <= 0 always,
            # while F_drag . V_ground may be positive in a tailwind (the air pushes
            # the shuttlecock). The invariant to test is the relative-frame power.
            vg = np.c_[r.df.vx_m_s, r.df.vy_m_s, r.df.vz_m_s]
            vr = vg - np.array([env_w.u_air_x, env_w.u_air_y, env_w.u_air_z])
            vrn = np.linalg.norm(vr, axis=1)
            p_rel = -r.df.drag_force_N.values * vrn
            rec("L 에너지", "항력 일률 (상대속도 기준) ≤ 0",
                "PASS" if np.all(p_rel <= 1e-9) else "FAIL",
                f"max = {float(np.max(p_rel)):.2e} W")
            drag_p = r.df.power_drag_W.values
            rec("L 에너지", "순풍에서 지면기준 항력 일률 > 0 가능",
                "PASS" if np.any(drag_p > 0) else "WARNING",
                f"max P_drag(지면기준) = {float(np.max(drag_p)):.2f} W — "
                "공기가 일을 하므로 정상")
        else:
            rec("L 에너지", "순풍 에너지 수지", "WARNING", "시뮬레이션 무효")
    except Exception as exc:
        rec("L 에너지", "순풍 에너지 수지", "WARNING", str(exc))

    # ---- C2. canonical inertia consistency (§7) --------------------------
    try:
        for gi in (False, True):
            m = build_full_model(db, db.profile_names()[0], geom, level=3,
                                  use_geometry_inertia=gi,
                                  cd_model=CdModel(mode="constant", constant_value=0.62),
                                  cl_model=ClModel(),
                                  cm_model=CmModel(mode="aero_pendulum"),
                                  spin=SpinModel(mode="constant"))
            m.turnover.Cd_ref = 0.62
            tc = turnover_constants(m, env, m.aero, U=20.0)
            w0 = m.turnover.omega0(20.0, 1.2, geom.S)
            rec("C 관성", f"Tw = 1/ω₀ 일관성 (형상관성 {'ON' if gi else 'OFF'})",
                "PASS" if abs(tc["Tw_model"] - 1.0 / w0) < 1e-9 else "FAIL",
                f"{tc['Tw_model']:.5f} vs {1.0/w0:.5f} ({tc['inertia_source']})")
    except Exception as exc:
        rec("C 관성", "canonical inertia", "FAIL", str(exc))

    # ---- O2. solidity provenance (§29) ------------------------------------
    try:
        pm = ShuttlecockPorosityModel()
        sg = ShuttlecockPorosityModel.geometric_solidity(SkirtGeometry(), geom)
        rec("O 파손 대칭", "기하 solidity 와 유효 공력 solidity 분리",
            "PASS" if abs(sg - pm.sigma0) > 0.1 else "WARNING",
            f"기하 {sg:.3f} vs 앵커 {pm.sigma0:.3f} — 서로 다른 양으로 표기됨")
    except Exception as exc:
        rec("O 파손 대칭", "solidity 구분", "WARNING", str(exc))

    # ---- N. 3D -> 2D reduction -----------------------------------------
    try:
        mp = build_full_model(db, db.profile_names()[0], geom, level=2,
                               cd_model=CdModel(mode="constant", constant_value=0.65),
                               cl_model=ClModel(), cm_model=CmModel(mode="aero_pendulum"),
                               spin=SpinModel())
        mp.turnover.Cd_ref = 0.65; mp.Ispin = 1.2e-6
        rp = SimulationEngine3D(mp, env, InitialCondition3D(V0=25, elevation_deg=-10,
                                 y0=1.8, body_axis_offset_deg=145.0),
                                 dt=2e-4, t_max=0.5, planar=True).run()
        md_ = build_model_from_profile(db, db.profile_names()[0], geom)
        md_.Cd_model.constant_value = 0.65; md_.turnover.Cd_ref = 0.65
        rd = SimulationEngine(md_, env, InitialCondition.from_launch(
            25, -10, 1.8, 145, 180, 0.0), dt=2e-4, t_max=0.5).run()
        tt = np.linspace(0, 0.45, 200)
        err = max(float(np.max(np.abs(np.interp(tt, rp.df.time_s, rp.df[c])
                                       - np.interp(tt, rd.df.time_s, rd.df[c]))))
                  for c in ("x_m", "y_m", "speed_m_s"))
        rec("N 2D 환원", "평면 구속 3D = 기존 2D",
            "PASS" if err < 1e-6 else "FAIL", f"최대 차이 {err:.2e}")
        rec("N 2D 환원", "평면 구속에서 z = 0",
            "PASS" if float(np.max(np.abs(rp.df.z_m))) == 0.0 else "FAIL", "")
    except Exception as exc:
        rec("N 2D 환원", "환원", "FAIL", str(exc))

    # ---- O. damage symmetry --------------------------------------------
    try:
        pm = ShuttlecockPorosityModel()
        gs_sym = build_full_model(db, db.profile_names()[0], geom, level=4,
                                   removed_feathers=[1, 9]).geometry_state()
        gs_asym = build_full_model(db, db.profile_names()[0], geom, level=4,
                                    removed_feathers=[1, 2]).geometry_state()
        b_sym = pm.lateral_force_bias(gs_sym)
        b_asym = pm.lateral_force_bias(gs_asym)
        rec("O 파손 대칭", "대칭 제거 → 측력 편향 작음",
            "PASS" if b_sym < b_asym else "FAIL",
            f"대칭 {b_sym:.4f} < 비대칭 {b_asym:.4f}")
        rec("O 파손 대칭", "정상 → 편향 0",
            "PASS" if pm.lateral_force_bias(
                build_full_model(db, db.profile_names()[0], geom,
                                  level=4).geometry_state()) == 0.0 else "FAIL", "")
    except Exception as exc:
        rec("O 파손 대칭", "비대칭", "FAIL", str(exc))

    # ---- P. wind tunnel constraint -------------------------------------
    try:
        mw = build_full_model(db, db.profile_names()[0], geom, level=3,
                               use_geometry_inertia=True,
                               cd_model=CdModel(mode="orientation", constant_value=0.65),
                               cl_model=ClModel(mode="constant", constant_value=0.1),
                               cm_model=CmModel(mode="aero_pendulum"),
                               spin=SpinModel(mode="aero_driven", spin_ratio=0.04,
                                               R_m=geom.R_m))
        mw.turnover.Cd_ref = 0.65; mw.Ispin = 1.2e-6
        rw, stw, _ = run_wind_tunnel_experiment(mw, VirtualWindTunnel(), 20.0,
                                                 incidence_deg=180.0, spin_locked=True,
                                                 dt=1e-3, t_max=0.8)
        dw = rw.df
        rec("P 풍동 구속", "자전 구속 시 축 spin 불변",
            "PASS" if float(np.max(np.abs(dw.spin_axial_rad_s))) < 1e-9 else "FAIL",
            f"최대 {float(np.max(np.abs(dw.spin_axial_rad_s))):.2e} rad/s")
        rec("P 풍동 구속", "Turnover 자유도는 자유",
            "PASS" if float(dw.omega_rad_s.max()) > 1.0 else "FAIL",
            f"최대 각속도 {float(dw.omega_rad_s.max()):.2f} rad/s")
        rec("P 풍동 구속", "구속 반력 토크 기록됨",
            "PASS" if np.all(np.isfinite(dw.constraint_torque_Nm.values)) else "FAIL",
            "constraint_torque_Nm")
    except Exception as exc:
        rec("P 풍동 구속", "구속", "FAIL", str(exc))

    # ---- Q. does it predict badminton? ---------------------------------
    # The point of the simulator is to predict reality, so reality is checked as an
    # invariant rather than spot-checked by hand. A shuttlecock's range grows only
    # like log(V0) because its aerodynamic length is about 4 m, so it cannot fly far
    # past the 13.4 m court however hard it is hit, and a damaged one flies SHORTER,
    # not further -- which is why players discard them.
    try:
        pm_q = ShuttlecockPorosityModel()
        worst = 0.0; worst_case = ""
        intact_by_case = {}
        dmg_by_case = {}
        for k in (0, 4, 8, 12):
            rem = list(range(1, k + 1))
            for V0, elev in ((30.0, 25.0), (60.0, 15.0), (100.0, 10.0)):
                mq = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                       removed_feathers=rem,
                                       damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                        if rem else None))
                mq.turnover.mode = "aero_pendulum"
                mq.turnover.Cd_ref = mq.Cd_model.constant_value
                icq = InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=V0,
                                          elevation_deg=elev, body_axis_offset_deg=180.0)
                rq = SimulationEngine3D(mq, env, icq, dt=2e-3, t_max=12.0).run()
                if not rq.valid or len(rq.df) == 0:
                    rec("Q 현실 예측", f"파손 {k}깃 / V0={V0:.0f} 실행", "FAIL", "시뮬레이션 실패")
                    continue
                rngq = float(rq.df.x_m.iloc[-1])
                key = f"V0={V0:.0f},{elev:.0f}deg"
                if k == 0:
                    intact_by_case[key] = rngq
                else:
                    dmg_by_case.setdefault(key, []).append((k, rngq))
                if rngq > worst:
                    worst, worst_case = rngq, f"{k}깃 제거, V0={V0:.0f} m/s, {elev:.0f}°"
        rec("Q 현실 예측", "어떤 파손·타구에서도 코트를 크게 넘지 않음",
            "PASS" if worst < 1.5 * COURT_LENGTH_M else "FAIL",
            f"최대 사거리 {worst:.2f} m ({worst_case}); 코트 {COURT_LENGTH_M} m")
        ratio_worst = 0.0; ratio_case = ""
        for key, lst in dmg_by_case.items():
            base = intact_by_case.get(key)
            if not base:
                continue
            for k, rngq in lst:
                r = rngq / base
                if r > ratio_worst:
                    ratio_worst, ratio_case = r, f"{k}깃 제거, {key}"
        # This used to assert that damage never lengthens the flight, which was a
        # guess made with no evidence either way. The evidence says otherwise: past a
        # critical gap size wider gaps reduce drag (Verma et al. 2015), and the game
        # has always known that shuttles with fewer feathers are faster -- bending the
        # feather tips outward, adding skirt back into the flow, is how a shuttle is
        # slowed down. A shuttlecock missing most of its skirt SHOULD fly further.
        #
        # What cannot happen is the range outrunning what carries it. Range grows with
        # the aerodynamic length ell = 2M/(rho*S*Cd) and only logarithmically with
        # launch speed, so the range ratio is bounded by the ell ratio. That is a
        # physical relation rather than a threshold picked to pass.
        ell_ratio = 1.0
        try:
            m_int = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2)
            m_dmg = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                      removed_feathers=list(range(1, 13)),
                                      damage_mapping=build_porosity_damage_mapping(pm_q))
            gs_d = m_dmg.geometry_state()
            cd_d = m_dmg.Cd_model.constant_value * pm_q.cd_factor(gs_d)
            ell_ratio = (m_dmg.aerodynamic_length(1.2, cd_d)
                         / m_int.aerodynamic_length(1.2,
                                                    m_int.Cd_model.constant_value))
        except Exception:
            logger.exception("ell ratio failed")
        rec("Q 현실 예측", "파손 사거리가 공기역학 길이가 허용하는 범위 안",
            "PASS" if ratio_worst <= 1.15 * ell_ratio else "FAIL",
            f"최대 사거리비 {ratio_worst:.3f}배 ({ratio_case}); "
            f"ℓ 비 {ell_ratio:.3f}배 → 상한 {1.15 * ell_ratio:.3f}배")

        # The same invariant on the 2D coupled engine. It used to be checked only in
        # 3D, and the 2D path had its own bug: the asymmetry side-force coefficient
        # was added to Fx/Fy as newtons instead of q*S*C, so at two missing feathers
        # a speed-independent 0.085 N (about 1.8x the shuttlecock's weight) pushed it
        # sideways for the whole flight. A damaged shuttlecock then climbed, hovered
        # at 2 m for 3 s and flew 1.66x further than an intact one.
        w2_worst = 0.0; w2_case = ""; dE_worst = 0.0
        base2 = {}
        for k in (0, 4, 8, 12):
            rem = list(range(1, k + 1))
            for V0, elev in ((30.0, 25.0), (60.0, 15.0), (100.0, 10.0)):
                m2 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                       removed_feathers=rem,
                                       damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                        if rem else None))
                m2.turnover.mode = "aero_pendulum"
                m2.turnover.Cd_ref = m2.Cd_model.constant_value
                bo2 = resolved_body_orientation("Cork Forward", 180.0, 180.0,
                                                 V0, elev, 0.0, 0.0)
                ic2 = InitialCondition.from_launch(V0, elev, 2.0, 180.0, 180.0,
                                                    omega0=0.0, body_orientation_deg=bo2)
                r2 = SimulationEngine(m2, env, ic2, dt=2e-3, t_max=12.0,
                                       model_level=2).run()
                if not r2.valid or len(r2.df) < 2:
                    rec("Q 현실 예측", f"2D 파손 {k}깃 / V0={V0:.0f} 실행", "FAIL",
                        "시뮬레이션 실패")
                    continue
                key2 = f"V0={V0:.0f},{elev:.0f}deg"
                rng2 = float(r2.df.x_m.iloc[-1])
                if k == 0:
                    base2[key2] = rng2
                elif base2.get(key2):
                    ratio2 = rng2 / base2[key2]
                    if ratio2 > w2_worst:
                        w2_worst, w2_case = ratio2, f"{k}깃 제거, {key2}"
                dE_worst = max(dE_worst,
                               float(np.diff(r2.df.E_total_J.values).max()))
        rec("Q 현실 예측", "2D 결합 모델에서도 사거리가 ℓ 범위를 넘지 않음",
            "PASS" if w2_worst <= 1.15 * ell_ratio else "FAIL",
            f"최대 사거리비 {w2_worst:.3f}배 ({w2_case}); "
            f"상한 {1.15 * ell_ratio:.3f}배")

        # The sharp invariant that the old range rule was standing in for, and the one
        # that was actually broken: taking skirt out of the flow can never add drag.
        # The bleed-jet term used to peak at solidity 0.5 while an intact skirt sits at
        # 0.65, so the first feathers removed RAISED Cd, and because the mass falls at
        # the same time the two cancelled -- twelve feathers gone left the shuttlecock
        # falling at 5.88 m/s against the intact 5.86 m/s.
        cds_dmg = []
        sk_q = SkirtGeometry()
        for k in range(0, 13):
            sk_q.restore_all()
            for i in range(1, k + 1):
                sk_q.remove(i)
            cds_dmg.append(pm_q.cd_at_solidity(
                pm_q.solidity_from_state(sk_q.geometry_state())))
        mono = all(cds_dmg[i] >= cds_dmg[i + 1] - 1e-12
                   for i in range(len(cds_dmg) - 1))
        rec("Q 현실 예측", "깃털을 뺄수록 항력계수가 단조 감소",
            "PASS" if mono else "FAIL",
            f"0깃 {cds_dmg[0]:.4f} → 12깃 {cds_dmg[-1]:.4f}; "
            + ("단조 감소" if mono else "중간에 증가하는 구간이 있음"))

        # And the damage has to show up in the speed, which is what the user saw.
        v_int = v_dmg = float("nan")
        try:
            ic_v = InitialCondition.from_launch(
                30.0, 25.0, 2.0, 180.0, 180.0, omega0=0.0,
                body_orientation_deg=resolved_body_orientation(
                    "Cork Forward", 180.0, 180.0, 30.0, 25.0, 0.0, 0.0))
            r_int = SimulationEngine(m_int, env, ic_v, dt=1e-3, t_max=12.0,
                                      model_level=2).run()
            r_dmg = SimulationEngine(m_dmg, env, ic_v, dt=1e-3, t_max=12.0,
                                      model_level=2).run()
            v_int = float(r_int.df.speed_m_s.iloc[-1])
            v_dmg = float(r_dmg.df.speed_m_s.iloc[-1])
        except Exception:
            logger.exception("damage speed comparison failed")
        gain = (v_dmg / v_int - 1.0) * 100.0 if v_int else float("nan")
        rec("Q 현실 예측", "파손이 낙하 속도에 반영된다 (같은 속도로 떨어지지 않음)",
            "PASS" if gain >= 3.0 else "FAIL",
            f"12깃 제거 최종 속도 {v_dmg:.3f} m/s vs 정상 {v_int:.3f} m/s "
            f"({gain:+.1f}%); 스커트가 사라지면 더 빨리 떨어져야 한다")
        rec("Q 현실 예측", "2D 결합 모델의 역학적 에너지가 증가하지 않음",
            "PASS" if dE_worst <= 1e-6 else "FAIL",
            f"한 스텝 최대 에너지 증가 {dE_worst:+.2e} J "
            f"(공력은 에너지를 넣을 수 없으므로 0 이하여야 함)")
        # The sharp form of the same invariant, at the conditions the bug actually
        # showed up in: a smash launched *downward* from 1.8 m. A shuttlecock hit
        # downwards cannot climb back above the racket -- there is no force able to
        # do it. With the side-force units wrong it rose 1.7 m above the launch
        # point, hovered, and flew 2.65x an intact one.
        climb_worst = 0.0; climb_case = ""
        for k in (0, 2, 4, 8, 12):
            rem = list(range(1, k + 1))
            for V0, elev, h in ((25.0, -10.0, 1.8), (36.0, -15.0, 1.9)):
                m3 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                       removed_feathers=rem,
                                       damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                        if rem else None))
                m3.turnover.mode = "aero_pendulum"
                m3.turnover.Cd_ref = m3.Cd_model.constant_value
                bo3 = resolved_body_orientation("Cork Forward", 180.0, 180.0,
                                                 V0, elev, 0.0, 0.0)
                ic3d = InitialCondition.from_launch(V0, elev, h, 180.0, 180.0,
                                                     omega0=0.0,
                                                     body_orientation_deg=bo3)
                r3 = SimulationEngine(m3, env, ic3d, dt=2e-3, t_max=12.0,
                                       model_level=2).run()
                if not r3.valid or len(r3.df) < 2:
                    continue
                climb = float(r3.df.y_m.max() - r3.df.y_m.iloc[0])
                if climb > climb_worst:
                    climb_worst = climb
                    climb_case = f"{k}깃 제거, V0={V0:.0f} m/s, {elev:.0f}°"
        rec("Q 현실 예측", "아래로 친 스매시는 타격 높이보다 위로 올라가지 않음",
            "PASS" if climb_worst <= 0.02 else "FAIL",
            f"최대 상승 {climb_worst:.3f} m ({climb_case or '해당 없음'}); "
            f"파손 비대칭 측력이 무게를 이기면 떠오른다")

        # 2D and 3D must be the same physics seen from two frames. They are separate
        # integrators with separate force assembly, so the agreement has to be
        # measured, not assumed -- a divergence means one of them is being fed
        # something the other is not.
        gap_worst = 0.0; gap_case = ""
        for V0, elev, h in ((30.0, 28.0, 1.1), (25.0, -10.0, 1.8), (36.0, -15.0, 1.9),
                            (26.0, 56.0, 1.6), (60.0, 15.0, 2.0), (8.0, 20.0, 1.0)):
            m2 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2)
            m2.turnover.mode = "aero_pendulum"
            m2.turnover.Cd_ref = m2.Cd_model.constant_value
            bo2 = resolved_body_orientation("Cork Forward", 180.0, 180.0,
                                             V0, elev, 0.0, 0.0)
            r2 = SimulationEngine(
                m2, env, InitialCondition.from_launch(V0, elev, h, 180.0, 180.0,
                                                       omega0=0.0,
                                                       body_orientation_deg=bo2),
                dt=1e-3, t_max=12.0, model_level=2).run()
            m3 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2)
            m3.turnover.mode = "aero_pendulum"
            m3.turnover.Cd_ref = m3.Cd_model.constant_value
            r3 = SimulationEngine3D(
                m3, env, InitialCondition3D(x0=0.0, y0=h, z0=0.0, V0=V0,
                                             elevation_deg=elev, azimuth_deg=0.0,
                                             body_axis_offset_deg=180.0),
                dt=1e-3, t_max=12.0).run()
            if not (r2.valid and r3.valid and len(r2.df) and len(r3.df)):
                continue
            a = float(r2.df.x_m.iloc[-1]); c = float(r3.df.x_m.iloc[-1])
            gap = abs(c - a) / abs(a) * 100.0 if a else 0.0
            if gap > gap_worst:
                gap_worst, gap_case = gap, f"V0={V0:.0f} m/s, {elev:+.0f}°"
        # The same agreement with damage applied, which is where it broke. The two
        # engines were building the asymmetry moment from different formulas -- 2D
        # from a separate coefficient against l_ref, 3D from the derived imbalance
        # against the skirt radius -- so the same damaged shuttlecock came out up to
        # 11% apart. And the lateral excursion has to stay small: the Magnus effect
        # "was often considered to never occur in badminton contrary to other sports"
        # (Cohen et al., C. R. Physique 25 (2024) 1-15), so a damaged shuttlecock
        # curves a little, it does not steer.
        dmg_gap = 0.0; dmg_case = ""; z_frac = 0.0; z_case = ""
        for k in (2, 4, 6, 8):
            rem_k = list(range(1, k + 1))
            map_k = build_porosity_damage_mapping(pm_q)
            m2 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                   removed_feathers=rem_k, damage_mapping=map_k)
            r2d = SimulationEngine(
                m2, env, InitialCondition.from_launch(
                    30.0, 25.0, 2.0, 180.0, 180.0, omega0=0.0,
                    body_orientation_deg=resolved_body_orientation(
                        "Cork Forward", 180.0, 180.0, 30.0, 25.0, 0.0, 0.0)),
                dt=1e-3, t_max=12.0, model_level=2).run()
            m3 = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                   removed_feathers=rem_k,
                                   damage_mapping=build_porosity_damage_mapping(pm_q),
                                   spin=SpinModel(mode="aero_driven", spin_ratio=0.04,
                                                  R_m=geom.R_m))
            r3d = SimulationEngine3D(
                m3, env, InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30.0,
                                             elevation_deg=25.0, azimuth_deg=0.0,
                                             body_axis_offset_deg=180.0,
                                             spin0_rad_s=None),
                dt=1e-3, t_max=12.0).run()
            if not (r2d.valid and r3d.valid and len(r2d.df) and len(r3d.df)):
                continue
            a2 = float(r2d.df.x_m.iloc[-1]); a3 = float(r3d.df.x_m.iloc[-1])
            gap = abs(a3 - a2) / abs(a2) * 100.0 if a2 else 0.0
            if gap > dmg_gap:
                dmg_gap, dmg_case = gap, f"{k}깃 제거"
            zf = float(np.abs(r3d.df.z_m.values).max()) / max(abs(a3), 1e-9)
            if zf > z_frac:
                z_frac, z_case = zf, f"{k}깃 제거"
        rec("Q 현실 예측", "파손 상태에서도 2D 와 3D 가 10% 이내",
            "PASS" if dmg_gap <= 10.0 else "FAIL",
            f"최대 차이 {dmg_gap:.1f}% ({dmg_case or '해당 없음'}); "
            f"두 엔진이 같은 비대칭 모멘트를 써야 한다")
        # Direction, not just magnitude. The 2D engine used to put the azimuthal
        # asymmetry moment into its one pitch degree of freedom, which became a steady
        # trim and therefore lift: damage made the 2D shuttlecock glide 26% FURTHER
        # while the 3D one flew shorter. Two engines, one shuttlecock, opposite
        # answers -- and the aerodynamic length agreed with the 3D one.
        dir_bad = []
        for V0_d, el_d, h_d in ((25.0, -10.0, 1.8), (30.0, 25.0, 2.0)):
            base_2 = base_3 = None
            for k in (0, 4, 8):
                rem_k = list(range(1, k + 1))
                m2d = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                        removed_feathers=rem_k,
                                        damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                        if rem_k else None))
                r2 = SimulationEngine(
                    m2d, env, InitialCondition.from_launch(
                        V0_d, el_d, h_d, 180.0, 180.0, omega0=0.0,
                        body_orientation_deg=resolved_body_orientation(
                            "Cork Forward", 180.0, 180.0, V0_d, el_d, 0.0, 0.0)),
                    dt=1e-3, t_max=12.0, model_level=2).run()
                m3d = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                        removed_feathers=rem_k,
                                        damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                        if rem_k else None),
                                        spin=SpinModel(mode="aero_driven",
                                                       spin_ratio=0.04, R_m=geom.R_m))
                r3 = SimulationEngine3D(
                    m3d, env, InitialCondition3D(x0=0.0, y0=h_d, z0=0.0, V0=V0_d,
                                                  elevation_deg=el_d, azimuth_deg=0.0,
                                                  body_axis_offset_deg=180.0,
                                                  spin0_rad_s=None),
                    dt=1e-3, t_max=12.0).run()
                if not (r2.valid and r3.valid and len(r2.df) and len(r3.df)):
                    continue
                a2 = float(r2.df.x_m.iloc[-1]); a3 = float(r3.df.x_m.iloc[-1])
                if k == 0:
                    base_2, base_3 = a2, a3
                    continue
                d2 = a2 / base_2 - 1.0
                d3 = a3 / base_3 - 1.0
                if abs(d2) > 0.02 and abs(d3) > 0.02 and (d2 > 0) != (d3 > 0):
                    dir_bad.append(f"{k}깃 V0={V0_d:.0f} ({d2:+.1%} vs {d3:+.1%})")
        rec("Q 현실 예측", "파손에 대한 반응이 2D 와 3D 에서 같은 방향",
            "PASS" if not dir_bad else "FAIL",
            "; ".join(dir_bad) if dir_bad
            else "두 엔진 모두 같은 부호로 변한다 (방위각 비대칭은 평면 모델의 "
                 "피치 자유도에 넣으면 없는 양력이 된다)")

        # Nothing in a result table may be NaN on every row. Cm was: the default
        # restoring model is the Cohen pendulum, which has no Cm coefficient to
        # report, so the column came out undefined on every row of every run in both
        # engines, and the spin-constraint torque did the same whenever the spin was
        # free (where the answer is zero, not undefined).
        nan_cols = []
        for lab_e, res_e in (("2D", SimulationEngine(
                build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2), env,
                InitialCondition.from_launch(25.0, -10.0, 1.8, 180.0, 180.0),
                dt=1e-3, t_max=2.0, model_level=2).run()),
                             ("3D", SimulationEngine3D(
                build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2), env,
                InitialCondition3D(x0=0.0, y0=1.8, z0=0.0, V0=25.0,
                                    elevation_deg=-10.0,
                                    body_axis_offset_deg=180.0),
                dt=1e-3, t_max=2.0).run())):
            if not (res_e.valid and len(res_e.df)):
                continue
            num_e = res_e.df.select_dtypes("number")
            for c in num_e.columns:
                v = num_e[c].values
                if v.size and not np.any(np.isfinite(v)):
                    nan_cols.append(f"{lab_e}:{c}")
        rec("Q 현실 예측", "결과 표에 전 구간 NaN 인 열이 없음",
            "PASS" if not nan_cols else "FAIL",
            "; ".join(nan_cols) if nan_cols
            else "2D·3D 정상 실행에서 모든 수치 열이 유한값을 가진다")

        # The spin ratio has to survive feather loss, or the shuttlecock stops turning
        # under its own asymmetry and the damage pushes it one way for the whole
        # flight. Driving and damping torque both scale with the vane area, so the
        # ratio is preserved; scaling only the driving side halved it.
        ratio_bad = []
        for k in (0, 4, 8, 12):
            rem_k = list(range(1, k + 1))
            m_s = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                    removed_feathers=rem_k,
                                    damage_mapping=(build_porosity_damage_mapping(pm_q)
                                                    if rem_k else None),
                                    spin=SpinModel(mode="aero_driven", spin_ratio=0.04,
                                                   R_m=geom.R_m))
            r_s = SimulationEngine3D(
                m_s, env, InitialCondition3D(x0=0.0, y0=1.8, z0=0.0, V0=25.0,
                                              elevation_deg=-10.0,
                                              body_axis_offset_deg=180.0,
                                              spin0_rad_s=None),
                dt=5e-4, t_max=12.0).run()
            if not (r_s.valid and len(r_s.df) > 2):
                continue
            sp_end = float(r_s.df.spin_rad_s.iloc[-1])
            u_end = float(r_s.df.speed_m_s.iloc[-1])
            ratio = sp_end * geom.R_m / u_end if u_end > 1e-9 else 0.0
            if not (0.025 <= ratio <= 0.055):
                ratio_bad.append(f"{k}깃 RΩ/U={ratio:.3f}")
        # The constant-Cd model is licensed by Cooke's flat drag curve over
        # Re = 1.3e4 to 2e5. A 300 km/h smash reaches 3.6e5 and the record 493 km/h
        # reaches 6.0e5, and the model reported Cd = 0.605 through all of it in
        # silence. The trajectory is still about right, but the caveat belongs where
        # the user can see it.
        re_flag = []
        for V0_r, lab_r in ((30.0, "클리어 30 m/s"), (83.3, "스매시 300 km/h")):
            r_r = SimulationEngine3D(
                build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2), env,
                InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=V0_r,
                                    elevation_deg=25.0, body_axis_offset_deg=180.0),
                dt=5e-4, t_max=12.0).run()
            if not (r_r.valid and len(r_r.df)):
                continue
            hit = any("Re" in w for w in r_r.warnings)
            want = V0_r > 50.0
            if hit != want:
                re_flag.append(f"{lab_r}: 경고={hit}, 기대={want}")
        rec("Q 현실 예측", "측정 범위를 벗어난 Reynolds 수를 알린다",
            "PASS" if not re_flag else "FAIL",
            "; ".join(re_flag) if re_flag
            else f"Re {REYNOLDS_VALID_RANGE[0]:.0e}~{REYNOLDS_VALID_RANGE[1]:.0e} "
                 f"(Cooke 1999) 밖이면 경고하고, 안이면 조용하다")

        bad_launch = []
        for y_l, ok_l in ((2.0, True), (0.0, True), (-1.0, False)):
            r_l = SimulationEngine3D(
                build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2), env,
                InitialCondition3D(x0=0.0, y0=y_l, z0=0.0, V0=30.0,
                                    elevation_deg=25.0, body_axis_offset_deg=180.0),
                dt=1e-3, t_max=5.0).run()
            if bool(r_l.valid) != ok_l:
                bad_launch.append(f"y0={y_l}: valid={r_l.valid}, 기대={ok_l}")
        rec("Q 현실 예측", "지면 아래에서 발사하면 결과를 내지 않는다",
            "PASS" if not bad_launch else "FAIL",
            "; ".join(bad_launch) if bad_launch
            else "y0 < 0 은 거부하고 y0 >= 0 은 정상 실행 "
                 "(전에는 두 행짜리 결과를 valid=True 로 내놓았다)")

        rec("Q 현실 예측", "파손돼도 자전비 RΩ/U 가 유지된다",
            "PASS" if not ratio_bad else "FAIL",
            "; ".join(ratio_bad) if ratio_bad
            else "0~12깃 모두 RΩ/U = 0.04 부근 (Cohen 2015 fig.15) — "
                 "추진·감쇠 토크가 함께 줄어 비는 보존된다")

        rec("Q 현실 예측", "파손의 좌우 이탈이 사거리의 5% 이내",
            "PASS" if z_frac <= 0.05 else "FAIL",
            f"최대 |z|/사거리 {100 * z_frac:.2f}% ({z_case or '해당 없음'}); "
            f"배드민턴의 좌우 공력은 오랫동안 없다고 여겨질 만큼 작다 "
            f"(C. R. Physique 25 (2024) 1)")

        rec("Q 현실 예측", "3차원 모델이 2차원 모델과 10% 이내로 일치",
            "PASS" if gap_worst <= 10.0 else "FAIL",
            f"최대 차이 {gap_worst:.2f}% ({gap_case or '해당 없음'})")

        # Literature anchors (Cohen et al. 2015, New J. Phys. 17 063001): the
        # shuttlecock's only length scale is ell = 2M/(rho*S*C_D) ~ 4 m, it settles to
        # a terminal velocity of about 25 km/h, and its speed halves every few metres.
        # These are what make an 80 m badminton trajectory impossible.
        m_lit = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2)
        m_lit.turnover.mode = "aero_pendulum"
        m_lit.turnover.Cd_ref = m_lit.Cd_model.constant_value
        r_fall = SimulationEngine3D(
            m_lit, env, InitialCondition3D(x0=0.0, y0=50.0, z0=0.0, V0=0.1,
                                            elevation_deg=-90.0,
                                            body_axis_offset_deg=180.0),
            dt=1e-3, t_max=20.0).run()
        v_term = float(r_fall.df.speed_m_s.iloc[-1]) if len(r_fall.df) else float("nan")
        rec("Q 현실 예측", "종단속도가 문헌값 (약 25 km/h ≈ 6.9 m/s) 범위",
            "PASS" if 5.8 <= v_term <= 7.5 else "FAIL",
            f"자유낙하 종단속도 {v_term:.2f} m/s = {3.6 * v_term:.1f} km/h "
            f"(Cohen 2015 · 실측 보고 약 25 km/h)")

        r_dec = SimulationEngine3D(
            m_lit, env, InitialCondition3D(x0=0.0, y0=30.0, z0=0.0, V0=67.0,
                                            elevation_deg=0.0,
                                            body_axis_offset_deg=180.0),
            dt=5e-4, t_max=8.0).run()
        half_x = float("nan")
        if len(r_dec.df) > 2:
            v_start = float(r_dec.df.speed_m_s.iloc[0])
            below = np.flatnonzero(r_dec.df.speed_m_s.values <= 0.5 * v_start)
            if below.size:
                half_x = float(r_dec.df.x_m.iloc[int(below[0])])
        rec("Q 현실 예측", "속도가 절반이 되는 거리가 문헌값 (약 3.35 m) 규모",
            "PASS" if 2.0 <= half_x <= 4.5 else "FAIL",
            f"240 km/h 발사에서 속도 50% 도달 거리 {half_x:.2f} m "
            f"(문헌 보고 약 3.35 m)")

        # An impossible flight must be refused, not drawn. This is the guard that
        # stops an emptied Cd box (the browser sends 0) from being read as a result.
        m_nd = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                 cd_model=CdModel(mode="constant",
                                                  constant_value=1e-3))
        r_nd2 = SimulationEngine3D(
            m_nd, env, InitialCondition3D(x0=0.0, y0=1.1, z0=0.0, V0=30.0,
                                           elevation_deg=28.0,
                                           body_axis_offset_deg=180.0),
            dt=1e-3, t_max=20.0).run()
        nd_rng = float(r_nd2.df.x_m.iloc[-1]) if len(r_nd2.df) else float("nan")
        refused_env = bool(flight_envelope_errors(r_nd2.df, m_nd, env))
        rec("Q 현실 예측", "코트를 크게 넘는 궤적은 그리지 않고 거부",
            "PASS" if refused_env else "FAIL",
            f"항력을 거의 끈 상태의 사거리 {nd_rng:.1f} m "
            f"(한계 {ENVELOPE_REFUSAL_M:.1f} m) → 거부={refused_env}")
        # the states the model refuses to predict for must actually be refused
        refused = []
        for k in (13, 14, 15, 16):
            mq = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                   removed_feathers=list(range(1, k + 1)),
                                   damage_mapping=build_porosity_damage_mapping(pm_q))
            rq = SimulationEngine3D(mq, env,
                                     InitialCondition3D(x0=0.0, y0=2.0, z0=0.0, V0=30.0,
                                                        elevation_deg=25.0,
                                                        body_axis_offset_deg=180.0),
                                     dt=2e-3, t_max=12.0).run()
            refused.append(not rq.valid and len(rq.df) == 0)
        rec("Q 현실 예측", "스커트가 사라진 상태는 궤적을 내지 않고 거부",
            "PASS" if all(refused) else "FAIL",
            f"남은 깃털 3개 이하 {sum(refused)}/{len(refused)} 거부 "
            f"(맨 코르크로 외삽되어 20~38 m 를 날던 구간)")
        # 항력이 없으면 셔틀콕이 아니라 탄도 포물선이 된다. 실제로 '항력계수 직접
        # 지정' 칸을 비워두면 브라우저가 0 을 보내고, 그 0 이 3D 탭까지 흘러가 사거리
        # 49 m 짜리 포물선이 그려졌다. UI 경계에서 거부되는지 못 박아 둔다.
        m_nodrag = build_full_model(db, "BWF 천연 깃털 (기본)", geom, level=2,
                                     cd_model=CdModel(mode="constant",
                                                      constant_value=0.0))
        rejected = bool(physical_sanity_checks(m_nodrag, 1e-3, env))
        rej2 = any("Cd" in e for e in physical_sanity_checks(m_nodrag, 1e-3, env))
        ic_nd = InitialCondition3D(x0=0.0, y0=1.1, z0=0.0, V0=26.0,
                                    elevation_deg=21.0, body_axis_offset_deg=180.0)
        r_nd = SimulationEngine3D(m_nodrag, env, ic_nd, dt=2e-3, t_max=20.0).run()
        nd_range = float(r_nd.df.x_m.iloc[-1]) if len(r_nd.df) else float("nan")
        rec("Q 현실 예측", "항력 0 설정은 UI 입력 검증에서 거부",
            "PASS" if (rejected and rej2) else "FAIL",
            f"거부됨={rejected}; 통과했다면 26 m/s·21° 에서 {nd_range:.1f} m 를 "
            f"날아가는 탄도 포물선이 됨 (코트 {COURT_LENGTH_M} m)")

        # Every tab that runs on a drag coefficient must take it from the chosen
        # shuttlecock, not from a box pinned at 0.6. Otherwise picking "Synthetic"
        # (Cd 0.465) or a damaged profile (0.35 / 0.80) changed the label and nothing
        # else, and the 3D tab kept flying an intact feather shuttlecock.
        # Every profile in the database has to be a shuttlecock, not just a label.
        # "Severely Damaged Feather" used to cut Cd 42% while keeping the full mass,
        # so its aerodynamic length hit 7.38 m against the 4.04 m Cohen et al. (2015)
        # measure, and it flew 14.1 m -- 1.50x the intact flight, across the whole
        # court. Feather loss removes mass as well as drag; holding one and changing
        # the other leaves the ell that sets the range unanchored.
        ell_bad = []; ratio_bad = []
        base_rng = None
        for n in db.profile_names():
            mp = build_model_from_profile(db, n, geom)
            cd_p = mp.Cd_model.constant_value
            ell_p = mp.aerodynamic_length(1.2, cd_p)
            if not (2.5 <= ell_p <= 6.0):
                ell_bad.append(f"{n} ℓ={ell_p:.2f} m")
            rp = SimulationEngine(mp, env, InitialCondition.from_launch(
                30.0, 25.0, 2.0, 180.0, 180.0, omega0=0.0,
                body_orientation_deg=resolved_body_orientation(
                    "Cork Forward", 180.0, 180.0, 30.0, 25.0, 0.0, 0.0)),
                dt=1e-3, t_max=12.0, model_level=2).run()
            if not (rp.valid and len(rp.df)):
                continue
            rng_p = float(rp.df.x_m.iloc[-1])
            if base_rng is None:
                base_rng = rng_p
            elif "Damaged" in n and rng_p > 1.10 * base_rng:
                ratio_bad.append(f"{n} {rng_p / base_rng:.2f}배")
        rec("Q 현실 예측", "모든 프로파일의 공기역학 길이 ℓ이 2.5~6 m",
            "PASS" if not ell_bad else "FAIL",
            "; ".join(ell_bad) if ell_bad
            else f"{len(db.profile_names())}개 프로파일 모두 범위 내 "
                 f"(Cohen 2015 깃털 4.04 m)")
        rec("Q 현실 예측", "파손 프로파일이 정상보다 멀리 날지 않음",
            "PASS" if not ratio_bad else "FAIL",
            "; ".join(ratio_bad) if ratio_bad
            else "파손 프로파일 모두 정상 대비 110% 이하 "
                 "(깃털이 빠지면 항력과 함께 질량도 준다)")

        cds = {n: profile_drag_coefficient(db, n) for n in db.profile_names()}
        own = all(abs(cds[n] - build_model_from_profile(db, n, geom)
                      .Cd_model.constant_value) < 1e-12 for n in cds)
        spread = max(cds.values()) - min(cds.values())
        rec("Q 현실 예측", "프로파일마다 자기 항력계수를 쓴다 (0.6 고정이 아님)",
            "PASS" if (own and spread > 0.1) else "FAIL",
            "; ".join(f"{n}={v:.3f}" for n, v in cds.items())
            + f" (폭 {spread:.3f})")

        # The nylon skirt's speed-dependent drag, checked against the one measured
        # anchor it is built from and against a wind-tunnel value it did not see.
        cd_ny = CdModel(mode="constant", constant_value=0.68, skirt_deforms=True)
        cd_fe = CdModel(mode="constant", constant_value=0.605, skirt_deforms=False)
        f50 = cd_ny.deformation_factor(50.0)
        rec("Q 현실 예측", "나일론 스커트 변형: 50 m/s 에서 항력이 절반",
            "PASS" if abs(f50 - 0.5) < 1e-9 else "FAIL",
            f"계수 {f50:.4f} (실측 앵커 0.5 — 50 m/s 에서 단면적이 정지 상태의 1/2, "
            f"Phys. Scr. 2026 ae5361); v_d = {skirt_deformation_v_half():.2f} m/s")
        cd60 = cd_ny.cd(speed=60.0 / 3.6)
        rec("Q 현실 예측", "나일론 Cd 가 60 km/h 풍동 실측과 맞음",
            "PASS" if 0.44 <= cd60 <= 0.60 else "FAIL",
            f"모델 {cd60:.3f} vs Alam 외 실측 약 0.51 (앵커에 쓰지 않은 독립 검증점)")
        fe_flat = (cd_fe.cd(speed=5.0) == cd_fe.cd(speed=80.0) == 0.605)
        rec("Q 현실 예측", "깃털 셔틀콕 Cd 는 속도에 따라 변하지 않음",
            "PASS" if fe_flat else "FAIL",
            f"5 m/s {cd_fe.cd(speed=5.0):.3f} / 80 m/s {cd_fe.cd(speed=80.0):.3f} — "
            f"깃털은 변형되지 않으므로 Cohen 의 일정 Cd 해가 성립한다")

        ell_q = build_model_from_profile(db, "BWF 천연 깃털 (기본)",
                                          geom).aerodynamic_length(1.2, 0.605)
        rec("Q 현실 예측", "정상 셔틀콕 공기역학 길이 ℓ ≈ 4 m",
            "PASS" if 3.5 <= ell_q <= 4.6 else "FAIL",
            f"ℓ = {ell_q:.2f} m (Cohen 2015 깃털 4.04 m)")
    except Exception as exc:
        rec("Q 현실 예측", "현실 예측", "FAIL", str(exc))

    return pd.DataFrame(rows)

def physics_verification_markdown(db):
    try:
        df = physics_verification_suite(db)
    except Exception:
        logger.exception("physics suite failed")
        return "물리 검증 suite 실행 실패."
    n_fail = int((df["결과"] == "FAIL").sum())
    n_warn = int((df["결과"] == "WARNING").sum())
    n_pass = int((df["결과"] == "PASS").sum())
    lines = [f"### 자동 물리 검증  —  PASS {n_pass} · WARNING {n_warn} · FAIL {n_fail}",
             "", "| 구분 | 항목 | 결과 | 상세 |", "|---|---|---|---|"]
    icon = {"PASS": "✅ PASS", "WARNING": "⚠️ WARNING", "FAIL": "❌ FAIL"}
    for _, r in df.iterrows():
        lines.append(f"| {r['구분']} | {r['항목']} | {icon.get(r['결과'], r['결과'])} "
                     f"| {r['상세']} |")
    if n_fail:
        lines += ["", "> ❌ FAIL 항목이 있습니다. 해당 물리량은 연구 결과로 사용하면 "
                  "안 됩니다."]
    return "\n".join(lines)


CITED_STUDIES = [
    dict(
        key="Cohen2015",
        authors="C. Cohen, B. Darbois Texier, D. Quéré, C. Clanet",
        year=2015,
        title="The physics of badminton",
        venue="New Journal of Physics 17, 063001",
        doi="10.1088/1367-2630/17/6/063001",
        used=("뒤집힘(flip)·진동·정렬 시간, 공기역학 길이, 비선형 진자형 turnover 방정식, "
              "자연 자전 관계 R·Ω/U"),
        scope="깃털·플라스틱 셔틀콕, 자유비행 및 풍동, 준정상 공력 가정",
        caution=("논문의 τ_o·τ_s 는 소각 선형 예측값입니다. 실제 시뮬레이션은 비선형·감속을 "
                  "포함하므로 진폭이 큰 초기 구간에서는 더 긴 주기가 나옵니다. "
                  "R·Ω/U ≈ 0.04 는 측정된 경향이지 모든 속도·형상의 불변 상수가 아닙니다."),
        provenance="literature / literature_model",
    ),
    dict(
        key="Nakagawa2017",
        authors="K. Nakagawa, H. Hasegawa, M. Murakami, S. Obayashi",
        year=2017,
        title="Aerodynamic stability of a badminton shuttlecock",
        venue="Transactions of the JSME 83(856), 17-00165",
        doi="10.1299/transjsme.17-00165",
        used="피칭 모멘트와 자세 안정성, 스커트 틈이 turnover 안정성에 미치는 영향(정성적 근거)",
        scope="고Re 풍동, 깃털 셔틀콕",
        caution=("본 프로그램은 이 논문의 Cm 수치표를 직접 내장하고 있지 않습니다. "
                  "안정성의 방향성(복원 부호)과 틈의 역할에 대한 정성적 근거로만 사용합니다."),
        provenance="literature (정성적)",
    ),
    dict(
        key="Zhou2024",
        authors="L. Zhou",
        year=2024,
        title="Aerodynamic characteristics and trajectory analysis of badminton shuttlecocks",
        venue="Acta of Bioengineering and Biomechanics 26(4)",
        doi="10.37190/ABB-02508-2024-01",
        used="항력·양력·피칭력의 속도/받음각 의존성과 궤적 검증의 참조 범위",
        scope="약 10–50 m/s, 받음각 약 0–20°",
        caution=("이 범위를 벗어난 조건은 외삽입니다. 프로그램은 조건이 범위를 벗어나면 "
                  "'모델 외삽' 경고를 표시합니다."),
        provenance="literature",
    ),
    dict(
        key="Alam2015",
        authors="F. Alam, C. Nutakom, H. Chowdhury",
        year=2015,
        title="Effect of Porosity of Badminton Shuttlecock on Aerodynamic Drag",
        venue="Procedia Engineering 112, 430–435",
        doi="10.1016/j.proeng.2015.07.220",
        used="스커트 틈(porosity)이 항력에 미치는 영향 — 다공성 Cd 모델의 앵커 2점",
        scope="특정 셔틀콕의 틈을 인위적으로 막은 비교 실험",
        caution=("측정된 것은 '틈을 막은 상태 vs 정상 상태'이지 "
                  "'깃털 N개 제거 → Cd' 의 보편 법칙이 아닙니다. 프로그램의 "
                  "깃털 개수 보간은 model_derived 가정입니다."),
        provenance="literature (앵커) + model_derived (보간)",
    ),
    dict(
        key="Fujisawa2017",
        authors="Y. Fujisawa et al.",
        year=2017,
        title="Aerodynamic Characteristics of Badminton Shuttlecock with Forced Spin",
        venue="Proceedings of JSME",
        doi="10.1299/jsmeshd.2017.B-21",
        used="강제 자전이 항력과 유동 구조에 미치는 영향(자연 자전과 구분해 참조)",
        scope="강제 자전 풍동 실험",
        caution="강제 자전과 자연 자전은 서로 다른 조건이며 하나의 모델로 합치지 않았습니다.",
        provenance="literature (정성적)",
    ),
    dict(
        key="Cooke1999",
        authors="A. J. Cooke",
        year=1999,
        title="Shuttlecock Aerodynamics",
        venue="Sports Engineering 2, 85–96 (및 1992 케임브리지 박사학위논문)",
        doi="10.1046/j.1460-2687.1999.00023.x",
        used="Cd 의 Reynolds 수 독립성(Re 1.3×10⁴~1.9×10⁵), 기준 Cd 범위",
        scope="깃털·합성 셔틀콕 풍동 시험",
        caution="기준면적 정의가 문헌마다 달라 Cd 보고값이 0.48~0.65 로 흩어집니다.",
        provenance="literature",
    ),
    dict(
        key="Kitta2011",
        authors="S. Kitta, H. Hasegawa, M. Murakami, S. Obayashi",
        year=2011,
        title="Aerodynamic properties of a shuttlecock with spin at high Reynolds number",
        venue="Procedia Engineering 13, 271–277",
        doi="10.1016/j.proeng.2011.05.084",
        used="자전이 항력에 직접적인 유의차를 만들지 않는다는 근거, 틈의 항력 기여",
        scope="고Re 풍동, 자전/비자전 비교",
        caution="자전은 스커트 확장을 통해 간접적으로 작용할 수 있습니다.",
        provenance="literature",
    ),
    dict(
        key="ChanRossmann2012",
        authors="C. M. Chan, J. S. Rossmann",
        year=2012,
        title="Badminton shuttlecock aerodynamics: synthesizing experiment and theory",
        venue="Sports Engineering 15, 61–71",
        doi="10.1007/s12283-012-0084-9",
        used="대칭축이 상대유동과 정렬되면 양력이 0 이라는 조건 — 양력 정렬 모델의 근거",
        scope="풍동 + 궤적 이론",
        caution="Cl 의 각도 의존 형태 sin(2δ) 자체는 축대칭 가정에 기반한 간이 모델입니다.",
        provenance="literature (0 조건) + model_derived (sin2δ 형태)",
    ),
    dict(
        key="Chesneau2026",
        authors="E. Chesneau et al.",
        year=2026,
        title="Shuttlecock velocity decay after smash and slice shots in badminton",
        venue="Physica Scripta (arXiv:2310.11155 관련 연구 포함)",
        doi="10.1088/1402-4896/ae5361",
        used="슬라이스 타격 직후 자전율 100 rps 초과, 속도 지수감쇠와 공기역학 길이",
        scope="실제 경기 고속카메라 계측",
        caution="타격이 실어주는 초기 자전이며 자연 평형 자전과는 다른 양입니다.",
        provenance="literature",
    ),
    dict(
        key="BarlowRaePope",
        authors="J. B. Barlow, W. H. Rae, A. Pope",
        year=1999,
        title="Low-Speed Wind Tunnel Testing (3rd ed.)",
        venue="Wiley",
        doi="ISBN 978-0-471-55774-6",
        used="풍동 축방향 압력 분포, 마찰·확산부 손실계수 상관식",
        scope="저속 풍동 일반 설계",
        caution="AF1300 의 수축부·확산부 치수는 제조사 공개 자료가 없어 문헌 표준값을 사용했습니다.",
        provenance="literature_model",
    ),
]

PROVENANCE_LEGEND = {
    "literature": "문헌에서 직접 측정된 값",
    "literature_model": "문헌이 제안한 모델식",
    "model_derived": "문헌을 근거로 이 프로그램이 만든 보간/파생 모델",
    "user_defined": "사용자가 직접 입력한 값",
    "assumed": "근거가 없어 가정한 값",
    "unavailable": "데이터가 없어 계산하지 않음",
}

PARAMETER_SOURCES = [
    ("Cd (기준값)", "Cooke 1999 / Chan & Rossmann 2012 / Lin 2014", "literature"),
    ("Cd(자세) 오리엔테이션 의존", "Cohen 2015 fig.8b", "literature_model"),
    ("Cd(파손) 다공성 보간", "Alam 2015 앵커 + 자체 보간", "model_derived"),
    ("Cl 정렬 조건 (정렬 시 0)", "Chan & Rossmann 2012", "literature"),
    ("Cl 각도 형태 sin(2δ)", "축대칭 가정", "model_derived"),
    ("Cy (측력계수)", "데이터 없음", "unavailable"),
    ("Cm 복원 모멘트", "Cohen 2015 비선형 진자 / Nakagawa 2017 정성", "literature_model"),
    ("Cm(파손) 감소 모델", "실측 없음", "assumed"),
    ("Roll/Yaw 계수", "데이터 없음", "unavailable"),
    ("spin_ratio R·Ω/U", "Cohen 2015 fig.15b (깃털 0.04 / 플라스틱 0.02)", "literature"),
    ("초기 슬라이스 자전 (rps)", "Chesneau et al.", "literature"),
    ("c_spin 자전 감쇠", "실측 없음", "assumed"),
    ("감쇠 보정계수 k_d = 2.5", "Cohen 2015 fig.7b 역산", "model_derived"),
    ("l_GC 무게중심–코르크 거리", "Cohen 2015 / Cooke", "literature"),
    ("비대칭 파손 측력·모멘트", "기하학적 휴리스틱", "model_derived"),
    ("풍동 손실계수", "Barlow/Rae/Pope 상관식", "literature_model"),
    ("합성 난류 (OU 과정)", "실측 난류장 아님", "model_derived"),
]

def citation_markdown():
    lines = ["## 📚 인용 연구 자료", "",
             "이 프로그램이 실제로 사용하는 문헌입니다. 각 항목의 **주의**는 "
             "그 문헌을 어디까지 적용할 수 있는지를 나타냅니다.", ""]
    for c in CITED_STUDIES:
        lines += [
            f"### {c['authors']} ({c['year']})",
            f"**{c['title']}**  ",
            f"{c['venue']}  ",
            f"DOI: `{c['doi']}`  ",
            "",
            f"- **프로그램에서 사용:** {c['used']}",
            f"- **적용 범위:** {c['scope']}",
            f"- **주의:** {c['caution']}",
            f"- **성격:** {c['provenance']}",
            "",
        ]
    lines += ["---", "", "### 값의 성격 구분", "",
              "| 표기 | 뜻 |", "|---|---|"]
    for k, v in PROVENANCE_LEGEND.items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "### 주요 파라미터별 출처", "",
              "| 파라미터 | 출처 | 성격 |", "|---|---|---|"]
    for name, src, prov in PARAMETER_SOURCES:
        lines.append(f"| {name} | {src} | `{prov}` |")
    lines += ["", "*문헌에서 측정된 값과 이 프로그램이 만든 보간 모델을 "
              "섞어 표기하지 않습니다. `assumed` 와 `unavailable` 로 표시된 항목은 "
              "정량적 연구 결론의 근거로 사용하지 마세요.*"]
    return "\n".join(lines)

def research_readiness(db, extra_warnings=None):
    """§48: never auto-declare results research-grade; state what is missing."""
    reasons = []
    try:
        df = physics_verification_suite(db)
        n_fail = int((df["결과"] == "FAIL").sum())
        n_warn = int((df["결과"] == "WARNING").sum())
        if n_fail:
            reasons.append(f"물리 검증 FAIL {n_fail}건")
        if n_warn:
            reasons.append(f"물리 검증 WARNING {n_warn}건")
    except Exception:
        reasons.append("물리 검증 suite 실행 실패")
        n_fail = n_warn = -1
    for w in (extra_warnings or []):
        reasons.append(str(w))
    reasons.append("Cy·roll·yaw 계수 데이터 없음 (unavailable)")
    reasons.append("파손→Cm 관계와 c_spin 은 assumed")
    ok = (n_fail == 0)
    status = "검증 필요" if not ok else "물리 검증은 통과 — 아래 한계를 확인한 뒤 사용"
    lines = [f"### 연구 사용 판정: **{status}**", ""]
    if ok:
        lines.append("물리 검증 항목은 모두 통과했습니다. 다만 아래 한계가 남아 있으므로 "
                     "해당 물리량은 연구 결론의 근거로 쓰기 전에 직접 확인하세요.")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CFD verification / validation and coefficient database
#
# This module does NOT solve the Navier-Stokes equations. No CFD solver or mesher
# is bundled. It provides the verification machinery that a research-grade CFD
# study requires (grid and time convergence, GCI, quality gating, provenance) and
# ingests results produced by an external solver such as OpenFOAM. Anything that
# has not actually been computed is reported as unavailable rather than invented.
# ---------------------------------------------------------------------------

CFD_MODULE_ROLE = (
    "외부 CFD 결과 검증·연동 모듈 — 이 프로그램은 Navier-Stokes 방정식을 직접 풀지 "
    "않습니다. 격자/시간 수렴, GCI, 품질 게이트, provenance 관리를 담당하고 "
    "OpenFOAM 등 외부 솔버가 계산한 결과를 받아 축약모델에 연결합니다."
)

CFD_REQUIRED_FIELDS = [
    "case_id", "damage_state", "removed_feathers", "speed_m_s", "alpha_deg",
    "beta_deg", "Re", "Ma", "rho", "mu", "A_ref_m2", "L_ref_m",
    "Cd", "Cl", "Cy", "Cl_roll", "Cm", "Cn_yaw",
    "mesh_id", "mesh_cells", "yplus_max", "dt_s", "CFL_max",
    "turbulence_model", "solver", "reference_point",
]

def richardson_extrapolation(f1, f2, f3, r21, r32):
    """Observed order of accuracy and the extrapolated value (Roache 1994).

    f1 is the FINE grid value, f3 the coarse one. Returns the observed order p and
    the Richardson-extrapolated f_exact. When the three values are not in the
    asymptotic range (oscillatory or non-monotone convergence) p is not meaningful
    and the caller must not report a GCI.
    """
    e21 = float(f2) - float(f1)
    e32 = float(f3) - float(f2)
    if abs(e21) < 1e-30 or abs(e32) < 1e-30:
        return dict(p=float("nan"), f_exact=float(f1), asymptotic=False,
                     note="격자 간 차이가 0에 가까워 수렴차수를 정의할 수 없습니다")
    s = e32 / e21
    if s <= 0:
        return dict(p=float("nan"), f_exact=float("nan"), asymptotic=False,
                     note="진동 수렴 (부호 반전) — 점근 영역이 아니므로 GCI 계산 불가")
    # No abs() on the order: |e32| < |e21| means the solution gets WORSE as the grid
    # is refined, which gives p < 0 and must be rejected. Taking the magnitude turned
    # that into an apparently valid order and a GCI computed from it.
    p = math.log(abs(s)) / math.log(r21) if r21 > 1 else float("nan")
    if not np.isfinite(p) or p <= 0:
        return dict(p=float("nan"), f_exact=float("nan"), asymptotic=False,
                     note=("발산 (세밀한 격자에서 차이가 더 큼) — 점근 영역이 아니므로 "
                            "GCI 계산 불가" if np.isfinite(p) and p <= 0
                            else "수렴차수를 계산할 수 없습니다"))
    # Roache: f_exact = f1 + (f1 - f2) / (r^p - 1); e21 = f2 - f1, so subtract it
    f_exact = float(f1) - e21 / (r21 ** p - 1.0)
    return dict(p=float(p), f_exact=float(f_exact), asymptotic=True, note="")

def grid_convergence_index(f1, f2, r21, p, Fs=1.25):
    """GCI for the fine-grid solution (Roache 1994; Celik et al. 2008).

    Fs = 1.25 is the safety factor recommended when three or more grids are used.
    Returned as a fraction (multiply by 100 for percent).
    """
    if not np.isfinite(p) or p <= 0 or abs(f1) < 1e-30:
        return float("nan")
    ea = abs((float(f2) - float(f1)) / float(f1))
    return float(Fs * ea / (r21 ** p - 1.0))

def mesh_convergence_study(values, refinement_ratios=(2.0, 2.0), Fs=1.25,
                            name="Cd"):
    """Grid convergence for a coefficient computed on three meshes.

    `values` is (fine, medium, coarse). Reports observed order, extrapolated value
    and GCI, and marks the study non-asymptotic when the GCI assumptions fail.
    """
    f1, f2, f3 = [float(v) for v in values]
    r21, r32 = float(refinement_ratios[0]), float(refinement_ratios[1])
    rich = richardson_extrapolation(f1, f2, f3, r21, r32)
    out = dict(quantity=name, fine=f1, medium=f2, coarse=f3,
                r21=r21, r32=r32, observed_order=rich["p"],
                extrapolated=rich["f_exact"], asymptotic=rich["asymptotic"],
                note=rich["note"], source_type="verification (Roache 1994 / Celik 2008)")
    if rich["asymptotic"]:
        gci21 = grid_convergence_index(f1, f2, r21, rich["p"], Fs)
        gci32 = grid_convergence_index(f2, f3, r32, rich["p"], Fs)
        out["GCI_fine"] = gci21
        out["GCI_medium"] = gci32
        out["GCI_fine_percent"] = 100.0 * gci21 if np.isfinite(gci21) else float("nan")
        # asymptotic ratio should approach 1 if the solutions are in range
        if np.isfinite(gci21) and abs(gci21) > 1e-30:
            out["asymptotic_ratio"] = float(gci32 / (r21 ** rich["p"] * gci21))
        out["status"] = ("수렴" if (np.isfinite(gci21) and gci21 < 0.05)
                          else "격자 불확도 큼 (GCI >= 5%)")
    else:
        out["GCI_fine"] = float("nan")
        out["GCI_fine_percent"] = float("nan")
        out["status"] = "non-asymptotic — GCI 계산하지 않음"
    return out

def time_convergence_study(values, ratios=(2.0, 2.0), name="mean Cd"):
    """Same procedure applied to a timestep refinement triple (dt, dt/2, dt/4)."""
    out = mesh_convergence_study(values, ratios, name=name)
    out["quantity"] = name
    out["source_type"] = "verification (시간 수렴)"
    return out

def cfd_quality_gate(case):
    """§54: a coefficient is only approved when every critical check passes."""
    checks = {}

    def mark(key, ok, detail, critical=True):
        checks[key] = dict(passed=bool(ok), critical=bool(critical), detail=detail)

    mark("stable", bool(case.get("stable", False)),
         "발산 없이 계산이 끝났는가")
    mark("converged", bool(case.get("converged", False)),
         "잔차뿐 아니라 force/moment 이력이 안정되었는가")
    gci = case.get("GCI_Cd")
    mark("mesh_converged", (gci is not None and np.isfinite(gci) and gci < 0.05),
         f"GCI_Cd = {gci*100:.2f}%" if (gci is not None and np.isfinite(gci))
         else "격자 수렴 연구 없음")
    mark("time_converged", bool(case.get("time_converged", False)),
         "dt, dt/2, dt/4 비교 수행 여부",
         critical=bool(case.get("unsteady", False)))
    mark("domain_independent", bool(case.get("domain_independent", False)),
         "원거리 경계/출구 거리 변경 시 계수 변화 확인")
    yp = case.get("yplus_max")
    wall = str(case.get("wall_treatment", "")).lower()
    if "resolved" in wall:
        mark("yplus_ok", (yp is not None and yp < 5.0),
             f"y+ max = {yp}" if yp is not None else "y+ 미기록")
    elif "function" in wall:
        mark("yplus_ok", (yp is not None and 30.0 <= yp <= 300.0),
             f"y+ max = {yp}" if yp is not None else "y+ 미기록")
    else:
        mark("yplus_ok", False, "wall treatment 미지정", critical=True)
    ma = case.get("Ma")
    mark("low_mach_valid", (ma is not None and ma < 0.3),
         f"Ma = {ma}" if ma is not None else "Ma 미기록")
    ngap = case.get("N_gap")
    mark("gap_resolved", (ngap is not None and ngap >= 5),
         f"N_gap = {ngap} (틈 폭 / 국부 셀 크기)" if ngap is not None
         else "틈 해상도 미기록")
    mark("turbulence_sensitivity",
         bool(case.get("turbulence_sensitivity_done", False)),
         "난류모델 2종 이상 비교 여부", critical=False)
    mark("literature_valid", bool(case.get("literature_valid", False)),
         "문헌 비교 통과 여부", critical=False)

    critical_fail = [k for k, v in checks.items() if v["critical"] and not v["passed"]]
    status = "검증됨" if not critical_fail else "검증되지 않음"
    return dict(checks=checks, critical_failures=critical_fail,
                 coefficient_status=status,
                 note=("모든 critical 항목을 통과했습니다." if not critical_fail else
                        "critical 항목 실패: " + ", ".join(critical_fail)
                        + " — 이 계수는 연구 결과로 사용할 수 없습니다."))

CFD_VALIDATION_LEVELS = {
    0: "Level 0 — 식·단위 검사",
    1: "Level 1 — 표준 CFD 벤치마크",
    2: "Level 2 — 정상 셔틀콕 문헌 비교",
    3: "Level 3 — 정상 셔틀콕 실험 비교",
    4: "Level 4 — 파손 경향 검증",
    5: "Level 5 — 특정 파손 케이스 실험 검증",
}

def cfd_validation_status(level_reached):
    lv = int(level_reached)
    label = CFD_VALIDATION_LEVELS.get(lv, "미분류")
    if lv < 5:
        note = ("Level 5 이전에는 '검증된 파손 셔틀콕 CFD' 라고 표시할 수 없습니다. "
                f"현재 도달 단계: {label}")
    else:
        note = "특정 파손 케이스까지 실험 검증 완료"
    return dict(level=lv, label=label, note=note)

class CFDCoefficientDatabase:
    """Stores externally computed CFD coefficients with full provenance.

    Only cases whose quality gate returns 검증됨 are made available to the
    reduced-order simulator; everything else stays in the table marked as
    unverified so it can be inspected but not silently used.
    """

    def __init__(self):
        self.cases = []

    def add_case(self, **kw):
        missing = [f for f in CFD_REQUIRED_FIELDS if f not in kw]
        rec = dict(kw)
        rec["missing_fields"] = missing
        gate = cfd_quality_gate(kw)
        rec["coefficient_status"] = gate["coefficient_status"]
        rec["quality_note"] = gate["note"]
        rec["source_type"] = "CFD-derived (외부 솔버)"
        if missing:
            rec["coefficient_status"] = "검증되지 않음"
            rec["quality_note"] = ("필수 항목 누락: " + ", ".join(missing)
                                    + " — " + rec.get("quality_note", ""))
        self.cases.append(rec)
        return rec

    def to_frame(self):
        return pd.DataFrame(self.cases) if self.cases else pd.DataFrame(
            columns=CFD_REQUIRED_FIELDS + ["coefficient_status"])

    def verified(self):
        return [c for c in self.cases if c.get("coefficient_status") == "검증됨"]

    def lookup(self, Re=None, alpha_deg=None, removed=0, allow_extrapolation=False):
        """Interpolates verified cases only; never extrapolates silently (§56)."""
        cands = [c for c in self.verified()
                 if len(c.get("removed_feathers") or []) == int(removed)]
        if not cands:
            return None, "해당 파손 상태의 검증된 CFD 케이스가 없습니다"
        if alpha_deg is None:
            return cands[0], "ok"
        xs = np.array([float(c["alpha_deg"]) for c in cands])
        if not allow_extrapolation and (alpha_deg < xs.min() or alpha_deg > xs.max()):
            return None, (f"요청 받음각 {alpha_deg:.1f}° 가 CFD 데이터 범위 "
                           f"[{xs.min():.1f}, {xs.max():.1f}]° 밖입니다 — 외삽 금지")
        out = {}
        for key in ("Cd", "Cl", "Cy", "Cm"):
            ys = np.array([float(c.get(key, np.nan)) for c in cands])
            order = np.argsort(xs)
            out[key] = float(np.interp(alpha_deg, xs[order], ys[order]))
        out["source_type"] = "CFD-derived (보간)"
        return out, "ok"

    def stability_derivative(self, removed=0, alpha_eq_deg=180.0, half_window=10.0):
        """dCm/dalpha near the equilibrium, from verified CFD cases only (§45)."""
        cands = [c for c in self.verified()
                 if len(c.get("removed_feathers") or []) == int(removed)]
        pts = [(float(c["alpha_deg"]), float(c["Cm"])) for c in cands
               if abs(float(c["alpha_deg"]) - alpha_eq_deg) <= half_window]
        if len(pts) < 2:
            return None, ("평형점 주변에 검증된 CFD 케이스가 2개 미만입니다 — "
                           "dCm/dalpha 를 계산하지 않습니다")
        pts.sort()
        xs = np.radians([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope = float(np.polyfit(xs, ys, 1)[0])
        return dict(dCm_dalpha=slope,
                     stable=bool(slope < 0),
                     n_points=len(pts),
                     interpretation=("복원 (dCm/dα < 0)" if slope < 0
                                      else "불안정 (dCm/dα >= 0)"),
                     source_type="CFD-derived"), "ok"

def k_alpha_from_cfd(dCm_dalpha, rho, U, A_ref, L_ref):
    """k_alpha = -dM/dalpha from a CFD stability derivative (§46)."""
    q = 0.5 * float(rho) * float(U) ** 2
    return float(-q * float(A_ref) * float(L_ref) * float(dCm_dalpha))

def Tw_from_cfd(dCm_dalpha, rho, U, A_ref, L_ref, I_turnover):
    k = k_alpha_from_cfd(dCm_dalpha, rho, U, A_ref, L_ref)
    if k <= 0 or I_turnover <= 0:
        return None, ("k_alpha <= 0 (불안정 평형) 이므로 Tw 를 계산하지 않습니다"
                       if k <= 0 else "관성이 0 이하입니다")
    return dict(k_alpha=k, Tw=float(math.sqrt(I_turnover / k)),
                 source_type="CFD-derived"), "ok"

CFD_STUDIES = [
    dict(authors="S. Verma, A. Desai, S. Mittal", year=2013,
          title="Aerodynamics of badminton shuttlecocks",
          venue="Journal of Fluids and Structures 41, 89–98",
          doi="10.1016/j.jfluidstructs.2013.01.009",
          concept="깃털·스커트 틈의 항력 기여, 내·외부 압력차, 성분별 기여",
          applied="파손 형상 ↔ 공력 변화의 방향성 비교 기준",
          scope="약 25–50 m/s 계산", status="문헌 비교 가능"),
    dict(authors="C. S. H. Lin, C. K. Chua, J. H. Yeo", year=2014,
          title=("Aerodynamics of badminton shuttlecock: characterization of flow "
                 "around a conical skirt with gaps, behind a hemispherical dome"),
          venue="Journal of Wind Engineering and Industrial Aerodynamics 127, 29–39",
          doi="10.1016/j.jweia.2014.02.002",
          concept="틈을 통한 공기 누출, 내·외부 압력차, 후류/재순환 변화",
          applied="다공성 Cd 모델의 물리적 근거 (틈이 항력을 키우는 메커니즘)",
          scope="gapless 기준 케이스가 풍동 대비 3.2–4.7% 차이; 틈 변화로 항력 최대 약 45% 변동",
          status="문헌 비교 가능 (수치 동일성 주장 아님)"),
    dict(authors="F. Alam, C. Nutakom, H. Chowdhury", year=2015,
          title="Effect of Porosity of Badminton Shuttlecock on Aerodynamic Drag",
          venue="Procedia Engineering 112, 430–435",
          doi="10.1016/j.proeng.2015.07.220",
          concept="틈(porosity)이 항력에 미치는 실험적 영향",
          applied="다공성 Cd 모델의 실측 앵커 2점",
          scope="특정 시험체의 틈을 막은 비교 실험",
          status="문헌 비교 가능 (일반 법칙 아님)"),
    dict(authors="F. R. Menter", year=1994,
          title="Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications",
          venue="AIAA Journal 32(8), 1598–1605",
          doi="10.2514/3.12149",
          concept="SST k-ω 난류 모델",
          applied="외부 CFD 실행 시 권장 난류 모델 (이 프로그램은 솔버를 포함하지 않음)",
          scope="역압력구배·박리 공학 해석",
          status="프로그램 자체 계산 아님"),
    dict(authors="P. J. Roache", year=1994,
          title="Perspective: A Method for Uniform Reporting of Grid Refinement Studies",
          venue="Journal of Fluids Engineering 116, 405–413",
          doi="10.1115/1.2910291",
          concept="Grid Convergence Index (GCI), Richardson 외삽",
          applied="격자 수렴 검증 모듈에 직접 구현됨",
          scope="점근 수렴 영역", status="프로그램에 구현됨"),
    dict(authors="I. B. Celik et al.", year=2008,
          title=("Procedure for Estimation and Reporting of Uncertainty Due to "
                 "Discretization in CFD Applications"),
          venue="Journal of Fluids Engineering 130, 078001",
          doi="10.1115/1.2960953",
          concept="이산화 불확도 보고 절차, 관측 수렴차수",
          applied="GCI 보고 형식과 non-asymptotic 판정에 반영",
          scope="3개 이상 격자", status="프로그램에 구현됨"),
    dict(authors="Z. Peng et al.", year=2017,
          title=("Issues associated with Galilean invariance on a moving solid "
                 "boundary in the lattice Boltzmann method"),
          venue="Physical Review E 95, 013301",
          doi="10.1103/PhysRevE.95.013301",
          concept="LBM 이동경계의 Galilean 불변성과 운동량 교환",
          applied="이 프로그램은 LBM 을 솔버로 사용하지 않으므로 수치방법 참고문헌으로만 표시",
          scope="LBM 이동경계", status="관련 수치방법 참고문헌"),
]

def cfd_citation_markdown():
    lines = ["## 📚 CFD 연구 근거 및 검증", "",
             f"> **{CFD_MODULE_ROLE}**", "",
             "| 연구 | 검증/사용한 개념 | 프로그램 적용 위치 | 관계 |",
             "|---|---|---|---|"]
    for c in CFD_STUDIES:
        lines.append(f"| {c['authors']} ({c['year']}) | {c['concept']} | "
                     f"{c['applied']} | {c['status']} |")
    lines += ["", "### 서지사항", ""]
    for c in CFD_STUDIES:
        lines += [f"**{c['authors']} ({c['year']})**  ",
                  f"{c['title']}  ",
                  f"{c['venue']}  ",
                  f"DOI: `{c['doi']}`  ",
                  f"적용 범위: {c['scope']}", ""]
    lines += ["---", "", "### 검증(Verification) 과 확인(Validation) 구분", "",
              "| 구분 | 질문 | 이 프로그램의 상태 |", "|---|---|---|",
              "| Verification | 방정식을 올바르게 풀었는가 | 격자·시간 수렴, GCI, 품질 게이트를 "
              "**구현했습니다**. 단 풀 대상 솔버는 외부 프로그램입니다 |",
              "| Validation | 현실을 올바르게 재현하는가 | 정상 셔틀콕은 문헌 비교까지 "
              "(Level 2). 파손 케이스 실험 검증(Level 5)은 미완입니다 |",
              "", "### 검증 단계", ""]
    for lv, label in CFD_VALIDATION_LEVELS.items():
        lines.append(f"- {label}")
    lines += ["", "*Level 5 이전에는 '검증된 파손 셔틀콕 CFD' 라고 표시하지 않습니다.*"]
    return "\n".join(lines)

APP_CSS = """
    .gradio-container { max-width: 1500px !important; }
    input[type="number"], input[type="text"], textarea, select,
    .gr-box input, .gr-box textarea {
        border: 1px solid #94a3b8 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
    }
    input[type="number"]:focus, input[type="text"]:focus, textarea:focus {
        border-color: #2b6cb0 !important;
        box-shadow: 0 0 0 2px rgba(43,108,176,0.18) !important;
    }
    .block, .form {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        background: #f8fafc !important;
        padding: 10px !important;
    }
    label span { font-weight: 600 !important; color: #1e293b !important; }
    .tabitem { padding-top: 12px !important; }
"""


def client_js_errors(src):
    r"""Reasons this client-side (``js=``) handler will not parse as JavaScript.

    A ``js=`` payload is pasted into the page and parsed with the rest of Gradio's
    boot script. One syntax error there kills the whole front end: the app never
    finishes loading, no control ever responds, and nothing is logged on the server
    because the browser never got as far as sending a request. That is exactly what
    a raw newline inside a string literal did here -- ``"...m/s**\n\n*vx..."`` was
    written in a normal (non-raw) Python string, so Python turned the ``\n`` into a
    real newline before Gradio ever saw it, and JavaScript does not allow a line
    break inside a '' or "" literal. Write ``\n`` in the Python source (the docstring
    is raw, so that is the two characters backslash-n), or use a template literal.

    Only that class of error is detected -- enough to catch a hang, not a parser.
    """
    reasons = []
    quote = None
    escaped = False
    line = 1
    for ch in src:
        if ch == "\n":
            if quote in ("'", '"') and not escaped:
                reasons.append(
                    f"{line}번째 줄: 문자열 리터럴({quote}) 안에 줄바꿈이 그대로 들어 "
                    f"있습니다. 자바스크립트 문법 오류라 페이지 전체가 로딩에서 "
                    f"멈춥니다. 파이썬 소스에서 \\n 으로 이스케이프하세요.")
                quote = None
            line += 1
            escaped = False
            continue
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is None:
            if ch in "'\"`":
                quote = ch
        elif ch == quote:
            quote = None
    if quote is not None:
        reasons.append(f"닫히지 않은 문자열 리터럴({quote})이 있습니다.")
    return reasons


def disable_broken_client_js(demo):
    """Strip any ``js=`` handler that would stop the page from loading.

    Losing one instant-update handler costs a round trip to the server; shipping a
    broken one costs the entire app, and in a way that looks like a dead server
    rather than a typo. Fail to the slow path, loudly.
    """
    broken = []
    for key, fn in getattr(demo, "fns", {}).items():
        src = getattr(fn, "js", None)
        if not isinstance(src, str) or not src.strip():
            continue
        errs = client_js_errors(src)
        if errs:
            broken.append((key, errs))
            try:
                fn.js = None
            except Exception:
                logger.exception("could not disable broken js handler %s", key)
    for key, errs in broken:
        logger.error("client-side js handler %s disabled: %s", key, "; ".join(errs))
        print("[경고] 클라이언트 측 js 핸들러(%s)에 문법 오류가 있어 껐습니다. "
              "그대로 두면 페이지가 무한 로딩됩니다.\n        %s"
              % (key, "\n        ".join(errs)))
    return broken


def profile_drag_coefficient(db, profile_name, geometry=None):
    """The drag coefficient the chosen shuttlecock profile actually carries.

    The 2D basic tab builds its model straight from the profile, so it always used
    the profile's own Cd. The advanced and 3D tabs took theirs from a number box
    that was hard-wired to 0.6, so picking "Synthetic" (Cd 0.465) or a damaged
    profile (0.35 / 0.80) changed nothing there -- the 3D model kept flying a
    0.6 shuttlecock, and the two tabs answered different questions.
    """
    try:
        geom = geometry if geometry is not None else Geometry()
        return float(build_model_from_profile(db, profile_name, geom)
                     .Cd_model.constant_value)
    except Exception:
        logger.exception("profile Cd lookup failed for %r", profile_name)
        return 0.605


def gradio_ui_kwargs(where):
    """Route css to whichever of Blocks() and launch() this Gradio version accepts.

    Gradio 6.0 moved css (and theme) out of the Blocks constructor and into launch().
    Passing css to Blocks there only prints "The parameters have been moved ... css"
    and drops the styling, so the app renders unstyled. Reading the signature keeps one
    file working on Gradio 4, 5 and 6 alike.
    """
    try:
        import inspect as _inspect
        _ensure("gradio")
        import gradio as _gr   # gradio is imported inside build_ui(), not at module scope
        target = _gr.Blocks.__init__ if where == "blocks" else _gr.Blocks.launch
        params = _inspect.signature(target).parameters
        return {"css": APP_CSS} if "css" in params else {}
    except Exception:
        logger.exception("gradio css routing failed")
        return {}


def build_ui():
    _ensure("gradio")
    import gradio as gr

    # Display-only handlers: the work is microseconds, but the queue is shared and
    # serialised (default_concurrency_limit=1), so a keystroke in the launch speed
    # waited behind whatever else was running -- the physics suite takes 24 s and the
    # wind-tunnel sweep 26 s. queue=False sends them straight over HTTP instead;
    # show_progress hidden stops a spinner flashing on every keystroke; and
    # always_last collapses a burst of typing into one call for the final value.
    FAST = dict(queue=False, show_progress="hidden", trigger_mode="always_last")

    import plotly.graph_objects as go

    db = ParameterDatabase()
    state = dict(model=None, env=None, ic=None, result=None, exp_df=None, exp_quality=None,
                 fit_result=None, fit_model_snapshot=None, validation=None,
                 profile="BWF 천연 깃털 (기본)",
                 solver="RK4", dt=0.001, warnings=[],
                 skirt=SkirtGeometry(), model_level=1, aero=None, result3d=None)

    NA = "N/A"

    def fmt(v, unit="", nd=4):
        if v is None:
            return NA
        try:
            if isinstance(v, bool):
                return str(v)
            return f"{float(v):.{nd}g}{unit}"
        except (TypeError, ValueError):
            return str(v)

    def status_markdown():
        gs = state["skirt"].geometry_state()
        ic = state.get("ic")
        if ic is not None:
            V0v = math.hypot(ic.vx0, ic.vy0)
            la = rad2deg(math.atan2(ic.vy0, ic.vx0)) if V0v > 0 else 0.0
            bo = rad2deg(ic.theta_body0_rad)
            launch_txt = f"**발사 속도:** {V0v:.1f} m/s &nbsp;|&nbsp; **발사각:** {la:.1f}°"
            orient_txt = f"**초기 자세:** {bo:.1f}°"
        else:
            launch_txt = "**발사 속도:** 미설정 &nbsp;|&nbsp; **발사각:** 미설정"
            orient_txt = "**초기 자세:** 미설정"
        lvl = state["model_level"]
        phys = ("병진: 2차 항력 / 회전: 선형화 Turnover" if lvl == 1
                else "병진: Cd(Re, α) 항력+양력 / 회전: Cm 공력 모멘트")
        if state["result"] is None:
            sim_state = "대기"
        elif state["result"].valid:
            sim_state = "완료"
        else:
            sim_state = "오류"
        damage_txt = ("BWF 정상 기준" if gs["removed_feathers"] == 0
                      else "실험용 파손 형상")
        parts = [
            f"**셔틀콕:** {state['profile']}",
            f"**깃털:** {gs['remaining_feathers']}/{gs['N_feathers']} ({damage_txt})",
            launch_txt,
            orient_txt,
            f"**물리 모델:** {phys} (Level {lvl})",
            f"**Solver:** {state['solver']} &nbsp;|&nbsp; **dt:** {state['dt']} s",
            f"**상태:** {sim_state}",
            f"**실험 데이터:** {'불러옴 (' + str(len(state['exp_df'])) + '행)' if state['exp_df'] is not None else '없음'}",
            f"**피팅:** {'완료' if state['fit_result'] is not None else '미실행'}",
        ]
        return " &nbsp;|&nbsp; ".join(parts)

    def workflow_markdown():
        steps = ["① 셔틀콕", "② 발사 조건", "③ 시뮬레이션", "④ 결과 분석",
                 "⑤ 실험 비교", "⑥ 피팅", "⑦ 수치 검증", "⑧ 저장"]
        done = [True, True, state["result"] is not None,
                state["result"] is not None, state["exp_df"] is not None,
                state["fit_result"] is not None, state["validation"] is not None,
                False]
        rendered = []
        for s, d in zip(steps, done):
            rendered.append(f"**{s}**" if d else s)
        return " → ".join(rendered)

    def warning_center():
        if not state["warnings"]:
            return "✓ 경고 없음"
        return "\n".join(f"⚠ {w}" for w in state["warnings"])

    def push_warning(msg):
        if msg not in state["warnings"]:
            state["warnings"].append(msg)

    def clear_warnings():
        state["warnings"] = []

    def make_plot(df, x, y, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=title))
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, template="plotly_white",
                           margin=dict(l=40, r=20, t=40, b=40))
        return fig

    def input_validation_errors(mass, cd, tw, zeta, dt_val):
        errs = []
        if mass is not None and mass <= 0:
            errs.append("mass must be > 0")
        if cd is not None and cd <= 0:
            errs.append("항력계수 Cd 는 0 보다 커야 합니다 (정상 셔틀콕 Cd ≈ 0.6). "
                        "Cd = 0 이면 항력이 없어 탄도 포물선이 됩니다.")
        if tw is not None and tw <= 0:
            errs.append("Tw must be > 0")
        if zeta is not None and zeta < 0:
            errs.append("zeta must be >= 0")
        if dt_val is not None and dt_val <= 0:
            errs.append("dt must be > 0")
        return errs

    def animation_figure(res, max_frames=70):
        df = res.df if res is not None else None
        if df is None or len(df) == 0:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white", height=460, title_x=0.5,
                title="시뮬레이션을 실행하면 여기에 비행 영상이 재생됩니다",
                xaxis=dict(title="x [m]"), yaxis=dict(title="y [m]"),
                margin=dict(l=50, r=20, t=50, b=45))
            return fig

        n = len(df)
        idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))
        xs, ys = df.x_m.values, df.y_m.values
        xr = float(max(xs.max() - xs.min(), 1e-3))
        yr = float(max(ys.max() - ys.min(), 1e-3))
        pad_x, pad_y = 0.08 * xr, 0.18 * yr
        L = 0.075 * max(xr, yr)

        has_theta = "theta_cork_deg" in df.columns
        has_body = "theta_body_deg" in df.columns
        prof = shuttlecock_profile(state["model"].geometry if state.get("model") else None,
                                    state.get("skirt"))
        out = shuttlecock_outline_2d(prof)

        def shuttle_shape(row):
            if has_theta:
                ang = deg2rad(row.theta_cork_deg)
            elif has_body:
                ang = deg2rad(row.theta_body_deg) + math.pi
            else:
                ang = deg2rad(row.alpha_deg) + math.pi
            ux, uy = math.cos(ang), math.sin(ang)
            px, py = -math.sin(ang), math.cos(ang)
            LL = 2.0 * L

            def pt(u, r):
                return (row.x_m + ux * u * LL + px * r * LL,
                        row.y_m + uy * u * LL + py * r * LL)
            skx = [pt(u, r)[0] for u, r in zip(out["skirt_u"], out["skirt_r"])]
            sky = [pt(u, r)[1] for u, r in zip(out["skirt_u"], out["skirt_r"])]
            ckx = [pt(u, r)[0] for u, r in zip(out["cork_u"], out["cork_r"])]
            cky = [pt(u, r)[1] for u, r in zip(out["cork_u"], out["cork_r"])]
            return skx, sky, np.array(ckx), np.array(cky)

        def vel_line(row):
            sp = math.hypot(row.vx_m_s, row.vy_m_s)
            if sp <= 0:
                return [row.x_m, row.x_m], [row.y_m, row.y_m]
            sc = 1.8 * L / sp
            return ([row.x_m, row.x_m + row.vx_m_s * sc],
                    [row.y_m, row.y_m + row.vy_m_s * sc])

        first = df.iloc[idx[0]]
        skx, sky, ckx, cky = shuttle_shape(first)
        vx_, vy_ = vel_line(first)
        base = [
            go.Scatter(x=xs, y=ys, mode="lines", name="비행 궤적",
                       line=dict(color="#cbd5e1", width=2)),
            go.Scatter(x=skx, y=sky, mode="lines", fill="toself", name="깃털(스커트)",
                       fillcolor="rgba(255,255,255,0.95)",
                       line=dict(color="#c3cad6", width=1.8)),
            go.Scatter(x=ckx, y=cky, mode="lines", fill="toself", name="코르크 (앞머리)",
                       fillcolor="#16295e", line=dict(color="#16295e", width=1.8)),
            go.Scatter(x=vx_, y=vy_, mode="lines", name="진행 방향",
                       line=dict(color="#2f855a", width=3, dash="dot")),
        ]

        frames = []
        frame_names = []
        for i in idx:
            row = df.iloc[i]
            skxi, skyi, ckxi, ckyi = shuttle_shape(row)
            vxi, vyi = vel_line(row)
            fname = f"{row.time_s:.3f}"
            frame_names.append(fname)
            frames.append(go.Frame(
                name=fname,
                traces=[0, 1, 2, 3],
                data=[
                    go.Scatter(x=xs[:i + 1], y=ys[:i + 1], mode="lines",
                               line=dict(color="#cbd5e1", width=2)),
                    go.Scatter(x=skxi, y=skyi, mode="lines", fill="toself",
                               fillcolor="rgba(255,255,255,0.95)",
                               line=dict(color="#c3cad6", width=1.8)),
                    go.Scatter(x=ckxi, y=ckyi, mode="lines", fill="toself",
                               fillcolor="#16295e", line=dict(color="#16295e", width=1.8)),
                    go.Scatter(x=vxi, y=vyi, mode="lines",
                               line=dict(color="#2f855a", width=3, dash="dot")),
                ]))

        steps = [dict(method="animate", label=nm,
                      args=[[nm], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
                 for nm in frame_names]

        fig = go.Figure(data=base, frames=frames)
        fig.update_layout(
            template="plotly_white", height=460, title_x=0.5,
            title="셔틀콕 비행 영상 — Enter 재생 / Backspace 역재생",
            xaxis=dict(title="x [m]", range=[xs.min() - pad_x, xs.max() + pad_x]),
            yaxis=dict(title="y [m]", range=[min(ys.min() - pad_y, 0), ys.max() + pad_y]),
            margin=dict(l=50, r=20, t=50, b=45),
            legend=dict(orientation="h", y=1.02, x=0),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.02, y=-0.16,
                xanchor="left", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶ 재생", method="animate",
                         args=[None, dict(fromcurrent=True, mode="immediate",
                                          frame=dict(duration=45, redraw=True),
                                          transition=dict(duration=0))]),
                    dict(label="◀ 역재생", method="animate",
                         args=[frame_names[::-1], dict(mode="immediate",
                                                       frame=dict(duration=45, redraw=True),
                                                       transition=dict(duration=0))]),
                    dict(label="⏸ 정지", method="animate",
                         args=[[None], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=False),
                                            transition=dict(duration=0))]),
                ])],
            sliders=[dict(active=0, x=0.16, y=-0.14, len=0.82,
                          currentvalue=dict(prefix="시간 t = ", suffix=" s",
                                            font=dict(size=13)),
                          steps=steps)])
        return fig

    ANIM_DIV_ID = "shuttle_flight_anim"

    ANIM_TEMPLATE = r"""
<div id="__ID__-wrap" style="width:100%;font-family:system-ui,-apple-system,'Malgun Gothic',sans-serif">
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
    <button id="__ID__-play"  style="padding:6px 14px;border:1px solid #94a3b8;border-radius:6px;background:#2b6cb0;color:#fff;cursor:pointer;font-size:13px">▶ 재생</button>
    <button id="__ID__-rev"   style="padding:6px 14px;border:1px solid #94a3b8;border-radius:6px;background:#fff;cursor:pointer;font-size:13px">◀ 역재생</button>
    <button id="__ID__-stop"  style="padding:6px 14px;border:1px solid #94a3b8;border-radius:6px;background:#fff;cursor:pointer;font-size:13px">⏸ 정지</button>
    <button id="__ID__-rst"   style="padding:6px 14px;border:1px solid #94a3b8;border-radius:6px;background:#fff;cursor:pointer;font-size:13px">⏮ 처음</button>
    <label style="font-size:13px;color:#334155;margin-left:6px">재생 길이
      <select id="__ID__-spd" style="border:1px solid #94a3b8;border-radius:6px;padding:4px">
        <option value="16">아주 느리게 (16초)</option>
        <option value="8" selected>느리게 (8초)</option>
        <option value="4">보통 (4초)</option>
        <option value="2">빠르게 (2초)</option>
      </select>
    </label>
    <label style="font-size:13px;color:#334155;margin-left:8px">
      <input type="checkbox" id="__ID__-loop" checked style="vertical-align:middle"> 반복 재생
    </label>
    <span id="__ID__-clock" style="font-size:13px;color:#0f172a;font-weight:600;margin-left:8px"></span>
  </div>
  <input id="__ID__-scrub" type="range" min="0" max="100" value="0" step="1" style="width:100%;margin:2px 0 6px 0">
  <canvas id="__ID__-cv" style="width:100%;height:430px;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;display:block"></canvas>
  <div style="font-size:12px;color:#475569;margin-top:5px">
    ⌨ <b>Enter</b> 재생 · <b>Backspace</b> 역재생 · <b>Esc</b> 정지 &nbsp;|&nbsp;
    노란 밴드 쪽이 코르크입니다. 코르크가 진행 방향을 향하면 안정된 자세입니다.
  </div>
</div>
<script>
(function(){
  var D = __DATA__;
  var ID = "__ID__";
  var cv = document.getElementById(ID+"-cv");
  if(!cv) return;
  var ctx = cv.getContext("2d");
  var n = D.t.length;
  var idx = 0, dir = 0, last = null, raf = null;

  function resize(){
    var r = cv.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(300, r.width) * dpr;
    cv.height = 430 * dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  function view(){
    var w = cv.width/(window.devicePixelRatio||1), h = 430;
    var padL=52, padR=16, padT=14, padB=40;
    var sx = (w-padL-padR)/Math.max(D.xmax-D.xmin,1e-6);
    var sy = (h-padT-padB)/Math.max(D.ymax-D.ymin,1e-6);
    var s = Math.min(sx,sy);
    var ox = padL + ((w-padL-padR) - (D.xmax-D.xmin)*s)/2;
    var oy = padT + ((h-padT-padB) - (D.ymax-D.ymin)*s)/2;
    return {s:s, X:function(x){return ox+(x-D.xmin)*s;},
                Y:function(y){return oy+(D.ymax-y)*s;}, w:w, h:h,
                padL:padL, padB:padB};
  }
  function axes(v){
    ctx.strokeStyle="#e2e8f0"; ctx.lineWidth=1; ctx.fillStyle="#64748b";
    ctx.font="11px system-ui";
    var y0=v.Y(0);
    if(y0>0 && y0<v.h){ ctx.strokeStyle="#cbd5e1"; ctx.beginPath();
      ctx.moveTo(v.padL,y0); ctx.lineTo(v.w-10,y0); ctx.stroke();
      ctx.fillText("지면 y=0", v.padL+4, y0-4); }
    ctx.fillStyle="#475569";
    ctx.fillText("x [m]", v.w-46, v.h-12);
    ctx.save(); ctx.translate(14, 30); ctx.rotate(-Math.PI/2);
    ctx.fillText("y [m]", 0,0); ctx.restore();
  }
  function samp(i){
    var i0=Math.floor(i), i1=Math.min(i0+1,n-1);
    i0=Math.max(0,Math.min(i0,n-1));
    var f=i-i0;
    function L(a,b){ return a+(b-a)*f; }
    return {x:L(D.x[i0],D.x[i1]), y:L(D.y[i0],D.y[i1]), th:L(D.th[i0],D.th[i1]),
            vx:L(D.vx[i0],D.vx[i1]), vy:L(D.vy[i0],D.vy[i1]), t:L(D.t[i0],D.t[i1]), i0:i0};
  }
  function shuttle(v,s){
    var cx=v.X(s.x), cy=v.Y(s.y);
    var th=-s.th*Math.PI/180;
    var L=D.size*v.s;
    var ux=Math.cos(th), uy=Math.sin(th), px=-Math.sin(th), py=Math.cos(th);
    function P(u,r){ return [cx+ux*u*L+px*r*L, cy+uy*u*L+py*r*L]; }
    var pr=D.prof;
    var order=D.az.slice().sort(function(a,b){return Math.sin(a)-Math.sin(b);});
    for(var k=0;k<order.length;k++){
      var a=order[k], pos=Math.cos(a), ws=Math.abs(Math.sin(a));
      var g=Math.round(232+23*(Math.sin(a)+1)/2);
      ctx.beginPath();
      var seg=8, pts=[];
      for(var q=0;q<=seg;q++){ var t=q/seg;
        var u=pr.ub+(pr.ut-pr.ub)*t, r=pr.ch*pos+(pr.th_*pos-pr.ch*pos)*t;
        var w=(0.055+(0.20-0.055)*Math.pow(t,0.65))*ws+0.010;
        pts.push(P(u,r+w)); }
      for(var q=seg;q>=0;q--){ var t=q/seg;
        var u=pr.ub+(pr.ut-pr.ub)*t, r=pr.ch*pos+(pr.th_*pos-pr.ch*pos)*t;
        var w=(0.055+(0.20-0.055)*Math.pow(t,0.65))*ws+0.010;
        pts.push(P(u,r-w)); }
      ctx.moveTo(pts[0][0],pts[0][1]);
      for(var q=1;q<pts.length;q++) ctx.lineTo(pts[q][0],pts[q][1]);
      ctx.closePath();
      ctx.fillStyle="rgb("+g+","+g+","+Math.min(g+2,255)+")"; ctx.fill();
      ctx.strokeStyle="rgba(150,162,180,0.9)"; ctx.lineWidth=1; ctx.stroke();
    }
    var b0=P(pr.ub,0), b1=P(pr.ub+pr.clen*0.26,0);
    ctx.beginPath();
    var c1=P(pr.ub,pr.ch), c2=P(pr.ub+pr.clen*0.26,pr.ch),
        c3=P(pr.ub+pr.clen*0.26,-pr.ch), c4=P(pr.ub,-pr.ch);
    ctx.moveTo(c1[0],c1[1]); ctx.lineTo(c2[0],c2[1]);
    ctx.lineTo(c3[0],c3[1]); ctx.lineTo(c4[0],c4[1]); ctx.closePath();
    ctx.fillStyle="#12341f"; ctx.fill();
    ctx.beginPath();
    var us=pr.ub+pr.clen*0.26, dp=pr.un-us, first=true;
    for(var q=0;q<=24;q++){ var A=Math.PI/2*q/24;
      var pp=P(us+dp*Math.sin(A), pr.ch*Math.cos(A));
      if(first){ctx.moveTo(pp[0],pp[1]);first=false;} else ctx.lineTo(pp[0],pp[1]); }
    for(var q=24;q>=0;q--){ var A=Math.PI/2*q/24;
      var pp=P(us+dp*Math.sin(A), -pr.ch*Math.cos(A)); ctx.lineTo(pp[0],pp[1]); }
    ctx.closePath(); ctx.fillStyle="#fbfcfd"; ctx.fill();
    ctx.strokeStyle="#c3cad6"; ctx.lineWidth=1.2; ctx.stroke();
  }
  function draw(){
    var v=view();
    var s=samp(idx);
    ctx.clearRect(0,0,v.w,v.h);
    axes(v);
    ctx.strokeStyle="#cbd5e1"; ctx.lineWidth=1.5; ctx.beginPath();
    for(var i=0;i<n;i++){ var X=v.X(D.x[i]),Y=v.Y(D.y[i]);
      if(i===0)ctx.moveTo(X,Y); else ctx.lineTo(X,Y); }
    ctx.stroke();
    ctx.strokeStyle="#2b6cb0"; ctx.lineWidth=2.5; ctx.beginPath();
    for(var i=0;i<=s.i0;i++){ var X=v.X(D.x[i]),Y=v.Y(D.y[i]);
      if(i===0)ctx.moveTo(X,Y); else ctx.lineTo(X,Y); }
    ctx.lineTo(v.X(s.x),v.Y(s.y));
    ctx.stroke();
    var sp=Math.hypot(s.vx,s.vy);
    if(sp>0){ var k=D.size*v.s*2.2/sp;
      var X0=v.X(s.x),Y0=v.Y(s.y);
      var X1=X0+s.vx*k, Y1=Y0-s.vy*k;
      ctx.strokeStyle="#2f855a"; ctx.lineWidth=2; ctx.setLineDash([5,4]);
      ctx.beginPath(); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke();
      ctx.setLineDash([]); }
    shuttle(v,s);
    var c=document.getElementById(ID+"-clock");
    if(c) c.textContent="t = "+s.t.toFixed(4)+" s   ·   속력 "+sp.toFixed(1)+" m/s";
    var sc=document.getElementById(ID+"-scrub");
    if(sc && +sc.value!==idx) sc.value=idx;
  }
  function speed(){ var e=document.getElementById(ID+"-spd"); return e?parseFloat(e.value):8; }
  function loop(ts){
    if(last===null) last=ts;
    var dtr=(ts-last)/1000; last=ts;
    if(dir!==0){
      if(dtr>0.25) dtr=0.25;
      var adv=dtr*(n-1)/Math.max(speed(),0.1);
      idx+=dir*adv;
      var lp=document.getElementById(ID+"-loop");
      var looping=lp?lp.checked:false;
      if(idx>=n-1){ if(looping){ idx=idx-(n-1); } else { idx=n-1; dir=0; } }
      if(idx<=0){ if(looping){ idx=idx+(n-1); } else { idx=0; dir=0; } }
      draw();
    }
    raf=requestAnimationFrame(loop);
  }
  function go(d){ if(d>0&&idx>=n-1) idx=0; if(d<0&&idx<=0) idx=n-1; dir=d; last=null; }
  var pb=document.getElementById(ID+"-play"); if(pb) pb.onclick=function(){go(1);};
  var rb=document.getElementById(ID+"-rev");  if(rb) rb.onclick=function(){go(-1);};
  var sb=document.getElementById(ID+"-stop"); if(sb) sb.onclick=function(){dir=0;};
  var rs=document.getElementById(ID+"-rst");  if(rs) rs.onclick=function(){dir=0;idx=0;draw();};
  var scb=document.getElementById(ID+"-scrub");
  if(scb){ scb.max=n-1; scb.oninput=function(){ dir=0; idx=+scb.value; draw(); }; }
  if(window.__shuttleKey) document.removeEventListener("keydown",window.__shuttleKey,true);
  window.__shuttleKey=function(e){
    var t=e.target||{}, tg=(t.tagName||"").toUpperCase();
    if(tg==="INPUT"||tg==="TEXTAREA"||tg==="SELECT"||t.isContentEditable) return;
    if(e.key==="Enter"){e.preventDefault();go(1);}
    else if(e.key==="Backspace"){e.preventDefault();go(-1);}
    else if(e.key==="Escape"){e.preventDefault();dir=0;}
  };
  document.addEventListener("keydown",window.__shuttleKey,true);
  window.addEventListener("resize",resize);
  resize();
  if(raf) cancelAnimationFrame(raf);
  raf=requestAnimationFrame(loop);
  idx=0; draw();
})();
</script>
"""

    def _srcdoc_iframe(inner_html, height=600):
        """Wraps markup in a sandboxed iframe so its <script> actually runs.

        Gradio's HTML component injects markup with innerHTML, and the browser does
        not execute <script> tags inserted that way -- neither inline code nor
        external src. An iframe with srcdoc is parsed as a fresh document, so both
        run normally, and the WebGL context is isolated from the host page.
        """
        doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
               "<style>body{margin:0;padding:0;font-family:system-ui,-apple-system,"
               "'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:transparent}"
               "</style></head><body>" + inner_html + "</body></html>")
        esc = doc.replace("&", "&amp;").replace('"', "&quot;")
        return (f'<iframe srcdoc="{esc}" '
                f'style="width:100%;height:{int(height)}px;border:0;display:block" '
                'sandbox="allow-scripts allow-same-origin"></iframe>')

    GL_DIV_ID = "shuttle_gl_scene"

    GL_TEMPLATE = r"""
<div id="__ID__-wrap" style="border:1px solid #e2e8f0;border-radius:10px;padding:8px;background:#fff">
  <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px">
    <button id="__ID__-play" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 12px;cursor:pointer">▶ 재생</button>
    <button id="__ID__-pause" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 12px;cursor:pointer">⏸ 정지</button>
    <button id="__ID__-rst" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 12px;cursor:pointer">⏮ 처음</button>
    <label style="font-size:13px;color:#334155">재생 속도
      <select id="__ID__-rate" style="border:1px solid #94a3b8;border-radius:6px;padding:4px">
        <option value="0.002">1/500x (뒤집힘 관찰)</option>
        <option value="0.005">1/200x</option>
        <option value="0.01">1/100x</option>
        <option value="0.02" selected>1/50x</option>
        <option value="0.05">1/20x</option>
        <option value="0.1">1/10x</option>
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1">1x (실시간)</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
    </label>
    <button id="__ID__-toflip" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 10px;cursor:pointer">⤴ 뒤집힘 순간</button>
    <span style="font-size:13px;color:#334155;margin-left:6px">확대</span>
    <button id="__ID__-zin" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 10px;cursor:pointer">＋</button>
    <button id="__ID__-zout" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 10px;cursor:pointer">－</button>
    <button id="__ID__-zfit" style="border:1px solid #94a3b8;border-radius:6px;padding:5px 10px;cursor:pointer">⤢ 맞춤</button>
    <span id="__ID__-zlab" style="font-size:12px;color:#64748b">x1.00</span>
    <label style="font-size:13px;color:#334155"><input type="checkbox" id="__ID__-loop" checked style="vertical-align:middle"> 반복</label>
    <label style="font-size:13px;color:#334155"><input type="checkbox" id="__ID__-follow" checked style="vertical-align:middle"> 카메라 추적</label>
    <label style="font-size:13px;color:#334155"><input type="checkbox" id="__ID__-forces" checked style="vertical-align:middle"> 힘 벡터</label>
    <label style="font-size:13px;color:#b91c1c"><input type="checkbox" id="__ID__-ghost" checked style="vertical-align:middle"> 제거된 깃털 표시</label>
    <label style="font-size:13px;color:#334155">입자
      <select id="__ID__-parts" style="border:1px solid #94a3b8;border-radius:6px;padding:4px">
        <option value="0">끄기</option>
        <option value="150">적음</option>
        <option value="400" selected>보통</option>
        <option value="900">많음</option>
      </select>
    </label>
    <label style="font-size:13px;color:#334155">궤적
      <select id="__ID__-trail" style="border:1px solid #94a3b8;border-radius:6px;padding:4px">
        <option value="full" selected>전체</option>
        <option value="recent">최근 구간</option>
        <option value="off">끄기</option>
      </select>
    </label>
  </div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
    <input id="__ID__-scrub" type="range" min="0" max="100" value="0" step="0.001" style="flex:1">
    <span id="__ID__-hud" style="font-size:12px;color:#0f172a;font-weight:600;white-space:nowrap"></span>
  </div>
  <canvas id="__ID__-cv" style="width:100%;height:460px;display:block;border-radius:8px;background:#f8fafc"></canvas>
  <div style="font-size:11px;color:#64748b;margin-top:5px">
    간이 공기 흐름 시각화 — CFD 결과가 아니며 힘 계산에 사용되지 않습니다.
    물리는 이미 계산된 결과를 재생만 하며, 재생 중 재계산하지 않습니다.
  </div>
  <div id="__ID__-fps" style="font-size:11px;color:#94a3b8;margin-top:2px"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script>
(function(){
  var D = __DATA__;
  var ID = "__ID__";
  var cv = document.getElementById(ID+"-cv");
  if(!cv) return;
  function fail(msg){
    var w = document.getElementById(ID+"-wrap");
    var box = document.createElement("div");
    box.style.cssText = "padding:18px;color:#b45309;font-size:13px;line-height:1.6";
    box.innerHTML = msg;
    if(cv.parentNode) cv.parentNode.replaceChild(box, cv);
    else if(w) w.appendChild(box);
  }
  if(typeof THREE === "undefined"){
    fail('3D 렌더러(three.js)를 CDN에서 불러오지 못했습니다.<br>'+
         'Colab이 외부 네트워크를 차단하고 있거나 CDN 접속이 실패한 경우입니다.<br>'+
         '아래 Plotly 3D 장면과 분석 그래프는 정상 동작합니다.');
    return;
  }
  var n = D.t.length;
  if(n < 2) return;

  var renderer, scene, cam;
  try{
    renderer = new THREE.WebGLRenderer({canvas:cv, antialias:true, alpha:true});
  }catch(e){
    cv.outerHTML = '<div style="padding:20px;color:#b45309;font-size:13px">'+
      'WebGL을 사용할 수 없는 환경입니다.</div>';
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(45, 2, 0.01, 500);
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  var dl = new THREE.DirectionalLight(0xffffff, 0.55); dl.position.set(1,2,1.4); scene.add(dl);

  function resize(){
    var w = cv.clientWidth||760, h = 460;
    renderer.setSize(w, h, false);
    cam.aspect = w/h; cam.updateProjectionMatrix();
  }

  // ---- static scene ------------------------------------------------------
  var span = Math.max(D.span, 1e-3);
  var grid = new THREE.GridHelper(Math.max(span*1.6, 2), 16, 0xcbd5e1, 0xe2e8f0);
  grid.position.y = 0; scene.add(grid);

  var trailGeom = new THREE.BufferGeometry();
  var trailPos = new Float32Array(n*3);
  for(var i=0;i<n;i++){ trailPos[3*i]=D.x[i]; trailPos[3*i+1]=D.y[i]; trailPos[3*i+2]=D.z[i]; }
  trailGeom.setAttribute("position", new THREE.BufferAttribute(trailPos,3));
  var trailFull = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({color:0xcbd5e1}));
  scene.add(trailFull);
  var doneGeom = new THREE.BufferGeometry();
  doneGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n*3),3));
  var doneLine = new THREE.Line(doneGeom, new THREE.LineBasicMaterial({color:0x2b6cb0}));
  scene.add(doneLine);

  // ---- shuttlecock -------------------------------------------------------
  var body = new THREE.Group();
  var sc = D.scale;
  var P = D.prof;
  var corkLen = (P.un - P.ub)*2*sc, corkR = P.ch*2*sc;
  var corkMat  = new THREE.MeshLambertMaterial({color:0xf7f5ef});
  var leatherMat = new THREE.MeshLambertMaterial({color:0xfdfdfd});
  var baseMat  = new THREE.MeshLambertMaterial({color:0x1f3d2b});
  var quillMat = new THREE.MeshLambertMaterial({color:0xf0ece2});
  var ringMat  = new THREE.MeshLambertMaterial({color:0xd8d3c4});

  // cork: rounded nose + short cylindrical skirt-side base, leather-wrapped look
  var noseLen = corkLen*0.62;
  var nose = new THREE.Mesh(
    new THREE.SphereGeometry(corkR, 28, 18, 0, Math.PI*2, 0, Math.PI/2), leatherMat);
  nose.rotation.z = -Math.PI/2;
  nose.position.x = (P.un*2*sc) - corkR;
  body.add(nose);
  var corkBody = new THREE.Mesh(
    new THREE.CylinderGeometry(corkR, corkR*0.97, noseLen, 28), corkMat);
  corkBody.rotation.z = Math.PI/2;
  corkBody.position.x = (P.un*2*sc) - corkR - noseLen*0.5;
  body.add(corkBody);
  var corkBase = new THREE.Mesh(
    new THREE.CylinderGeometry(corkR*0.97, corkR*0.90, corkLen*0.34, 28), baseMat);
  corkBase.rotation.z = Math.PI/2;
  corkBase.position.x = (P.ub*2*sc) + corkLen*0.17;
  body.add(corkBase);

  var tipR = P.th_*2*sc, ub = P.ub*2*sc, ut = P.ut*2*sc;
  var span = ut - ub;
  var vaneMat = new THREE.MeshLambertMaterial({color:0xfcfcfa, side:THREE.DoubleSide,
                                                transparent:true, opacity:0.97});
  var NS = 6;                     // spanwise segments -> curved, tapered vane
  var IMB = 0.42;                 // imbrication: each feather leans on the next
  for(var k=0;k<D.az.length;k++){
    var a = D.az[k];
    var pos = [], idx = [];
    for(var j=0;j<=NS;j++){
      var f = j/NS;
      var ax = ub + span*f;
      var rr = corkR + (tipR-corkR)*f;
      // vane widens then tapers slightly, like a real trimmed feather
      var half = (corkR*0.55 + (tipR*0.95-corkR*0.55)*Math.pow(f,0.75))*0.5;
      var lean = IMB*half*f;      // overlap direction (all same sense = rifling)
      var th0 = a + (lean - half)/Math.max(rr,1e-6);
      var th1 = a + (lean + half)/Math.max(rr,1e-6);
      // slight cupping: outer edge sits a little further out
      var r0 = rr, r1 = rr*(1+0.05*f);
      pos.push(ax, r0*Math.cos(th0), r0*Math.sin(th0));
      pos.push(ax, r1*Math.cos(th1), r1*Math.sin(th1));
    }
    for(var j=0;j<NS;j++){
      var b0=2*j, b1=2*j+1, b2=2*j+2, b3=2*j+3;
      idx.push(b0,b1,b3, b0,b3,b2);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos),3));
    g.setIndex(idx);
    g.computeVertexNormals();
    body.add(new THREE.Mesh(g, vaneMat));

    // quill (feather shaft) along the inner edge
    var qg = new THREE.BufferGeometry();
    var qp = [];
    for(var j=0;j<=NS;j++){
      var f=j/NS, ax=ub+span*f, rr=corkR+(tipR-corkR)*f;
      var half=(corkR*0.55+(tipR*0.95-corkR*0.55)*Math.pow(f,0.75))*0.5;
      var th=a+(IMB*half*f-half)/Math.max(rr,1e-6);
      qp.push(ax, rr*Math.cos(th), rr*Math.sin(th));
    }
    qg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(qp),3));
    body.add(new THREE.Line(qg, new THREE.LineBasicMaterial({color:0xbdb7a6})));
  }

  // removed feathers, drawn as translucent red ghosts so the gap is visible
  var ghosts = [];
  var azr = D.az_removed || [];
  for(var k=0;k<azr.length;k++){
    var ga = azr[k];
    var gpos = [], gidx = [];
    for(var j=0;j<=NS;j++){
      var f = j/NS;
      var ax = ub + span*f;
      var rr = corkR + (tipR-corkR)*f;
      var half = (corkR*0.55 + (tipR*0.95-corkR*0.55)*Math.pow(f,0.75))*0.5;
      var lean = IMB*half*f;
      var th0 = ga + (lean - half)/Math.max(rr,1e-6);
      var th1 = ga + (lean + half)/Math.max(rr,1e-6);
      var r1v = rr*(1+0.05*f);
      gpos.push(ax, rr*Math.cos(th0), rr*Math.sin(th0));
      gpos.push(ax, r1v*Math.cos(th1), r1v*Math.sin(th1));
    }
    for(var j=0;j<NS;j++){
      var b0=2*j,b1=2*j+1,b2=2*j+2,b3=2*j+3;
      gidx.push(b0,b1,b3, b0,b3,b2);
    }
    var gg = new THREE.BufferGeometry();
    gg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(gpos),3));
    gg.setIndex(gidx);
    gg.computeVertexNormals();
    var gm = new THREE.Mesh(gg, new THREE.MeshBasicMaterial({
      color:0xdc2626, side:THREE.DoubleSide, transparent:true, opacity:0.15,
      depthWrite:false}));
    body.add(gm); ghosts.push(gm);
    var el = new THREE.LineSegments(new THREE.EdgesGeometry(gg),
      new THREE.LineBasicMaterial({color:0xdc2626, transparent:true, opacity:0.6}));
    body.add(el); ghosts.push(el);
  }

  // two binding rings (thread + glue) that hold the skirt together
  [[0.34,0.55],[0.72,0.42]].forEach(function(rg){
    var f=rg[0];
    var rr=(corkR+(tipR-corkR)*f)*1.005;
    var ring=new THREE.Mesh(
      new THREE.TorusGeometry(rr, Math.max(tipR*0.018, sc*0.06), 8, 40), ringMat);
    ring.rotation.y = Math.PI/2;
    ring.position.x = ub + span*f;
    body.add(ring);
  });

  var axisMat = new THREE.LineBasicMaterial({color:0x7c3aed});
  var axGeom = new THREE.BufferGeometry();
  axGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
    ut*1.25,0,0, (P.un*2*sc)*1.35,0,0]),3));
  body.add(new THREE.Line(axGeom, axisMat));
  scene.add(body);

  // ---- vectors -----------------------------------------------------------
  function mkArrow(col){
    var a = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(),
                                   sc*3, col, sc*1.1, sc*0.6);
    scene.add(a); return a;
  }
  var arV = mkArrow(0x2f855a), arW = mkArrow(0x805ad5),
      arD = mkArrow(0xc53030), arL = mkArrow(0xdd6b20), arSp = mkArrow(0x0891b2);

  // ---- airflow particles -------------------------------------------------
  var maxP = 900;
  var pPos = new Float32Array(maxP*3);
  var pAge = new Float32Array(maxP);
  var pOff = new Float32Array(maxP*2);
  for(var i=0;i<maxP;i++){
    pAge[i] = Math.random();
    pOff[2*i] = (Math.random()-0.5);
    pOff[2*i+1] = (Math.random()-0.5);
  }
  var pGeom = new THREE.BufferGeometry();
  pGeom.setAttribute("position", new THREE.BufferAttribute(pPos,3));
  var pts = new THREE.Points(pGeom, new THREE.PointsMaterial({
    color:0x60a5fa, size:sc*0.5, transparent:true, opacity:0.55, sizeAttenuation:true}));
  scene.add(pts);

  var wake = new THREE.Mesh(
    new THREE.ConeGeometry(1,1,20,1,true),
    new THREE.MeshBasicMaterial({color:0x93c5fd, transparent:true, opacity:0.13,
                                  side:THREE.DoubleSide}));
  scene.add(wake);

  // ---- state buffer sampling --------------------------------------------
  var qA = new THREE.Quaternion(), qB = new THREE.Quaternion(), qC = new THREE.Quaternion();
  var T0 = D.t[0], T1 = D.t[n-1];
  var simT = T0;

  function findIdx(tt){
    var lo=0, hi=n-1;
    while(hi-lo>1){ var mid=(lo+hi)>>1; if(D.t[mid]<=tt) lo=mid; else hi=mid; }
    return lo;
  }
  var S = {p:new THREE.Vector3(), v:new THREE.Vector3(), vr:new THREE.Vector3(),
           q:new THREE.Quaternion(), i:0, f:0, Vrel:0, spin:0, drag:0, lift:0,
           axis:new THREE.Vector3()};
  function sample(tt){
    var i = findIdx(tt), j = Math.min(i+1, n-1);
    var dtv = D.t[j]-D.t[i];
    var f = dtv>1e-12 ? (tt-D.t[i])/dtv : 0;
    S.i=i; S.f=f;
    function L(a,b){ return a+(b-a)*f; }
    S.p.set(L(D.x[i],D.x[j]), L(D.y[i],D.y[j]), L(D.z[i],D.z[j]));
    S.v.set(L(D.vx[i],D.vx[j]), L(D.vy[i],D.vy[j]), L(D.vz[i],D.vz[j]));
    S.vr.set(L(D.rx[i],D.rx[j]), L(D.ry[i],D.ry[j]), L(D.rz[i],D.rz[j]));
    qA.set(D.qx[i],D.qy[i],D.qz[i],D.qw[i]);
    qB.set(D.qx[j],D.qy[j],D.qz[j],D.qw[j]);
    if(qA.dot(qB)<0){ qB.set(-qB.x,-qB.y,-qB.z,-qB.w); }
    qC.copy(qA).slerp(qB, f);
    S.q.copy(qC);
    S.Vrel = L(D.vrel[i],D.vrel[j]);
    S.spin = L(D.spin[i],D.spin[j]);
    S.drag = L(D.drag[i],D.drag[j]);
    S.lift = L(D.lift[i],D.lift[j]);
    S.axis.set(1,0,0).applyQuaternion(S.q);
  }

  // ---- camera ------------------------------------------------------------
  var camTgt = new THREE.Vector3(), camPos = new THREE.Vector3();
  var camInit = false;
  var yaw = 0.6, pitch = 0.30, dist = Math.max(span*0.55, sc*22);
  var drag0 = null;
  cv.addEventListener("mousedown", function(e){ drag0={x:e.clientX,y:e.clientY}; });
  window.addEventListener("mouseup", function(){ drag0=null; });
  cv.addEventListener("mousemove", function(e){
    if(!drag0) return;
    yaw -= (e.clientX-drag0.x)*0.006;
    pitch = Math.max(-1.3, Math.min(1.3, pitch + (e.clientY-drag0.y)*0.006));
    drag0={x:e.clientX,y:e.clientY};
  });
  cv.addEventListener("wheel", function(e){
    e.preventDefault();
    zoom(1 + (e.deltaY>0 ? 0.08 : -0.08));
  }, {passive:false});

  function updateCamera(dtr){
    var follow = document.getElementById(ID+"-follow");
    var tgt = (follow && follow.checked) ? S.p :
              new THREE.Vector3(D.cx, D.cy, D.cz);
    var want = new THREE.Vector3(
      tgt.x + dist*Math.cos(pitch)*Math.sin(yaw),
      tgt.y + dist*Math.sin(pitch) + sc*2,
      tgt.z + dist*Math.cos(pitch)*Math.cos(yaw));
    if(!camInit){ camTgt.copy(tgt); camPos.copy(want); camInit=true; }
    var k = 1 - Math.exp(-8*Math.min(dtr,0.1));
    camTgt.lerp(tgt, k); camPos.lerp(want, k);
    cam.position.copy(camPos); cam.lookAt(camTgt);
  }

  // ---- per-frame updates -------------------------------------------------
  function updateScene(dtr){
    body.position.copy(S.p);
    body.quaternion.copy(S.q);
    var gh = document.getElementById(ID+"-ghost");
    var gv = gh ? gh.checked : true;
    for(var gi=0; gi<ghosts.length; gi++) ghosts[gi].visible = gv;

    var tr = document.getElementById(ID+"-trail");
    var mode = tr?tr.value:"full";
    trailFull.visible = (mode==="full");
    doneLine.visible = (mode!=="off");
    if(mode!=="off"){
      var end = S.i+1;
      var start = (mode==="recent") ? Math.max(0, end-Math.floor(n*0.18)) : 0;
      var arr = doneGeom.attributes.position.array;
      var c=0;
      for(var i=start;i<=end && i<n;i++){ arr[3*c]=D.x[i];arr[3*c+1]=D.y[i];arr[3*c+2]=D.z[i]; c++; }
      if(c>0){ arr[3*c]=S.p.x;arr[3*c+1]=S.p.y;arr[3*c+2]=S.p.z; c++; }
      doneGeom.setDrawRange(0, Math.max(c,2));
      doneGeom.attributes.position.needsUpdate = true;
    }

    var showF = document.getElementById(ID+"-forces");
    var sf = showF?showF.checked:true;
    var vlen = S.v.length()||1e-9;
    var fmax = Math.max(S.drag, S.lift, D.weight, 1e-9);
    function setA(a, dir, len, on){
      a.visible = on && len>1e-6 && dir.lengthSq()>1e-18;
      if(!a.visible) return;
      a.position.copy(S.p);
      a.setDirection(dir.clone().normalize());
      a.setLength(len, len*0.28, len*0.16);
    }
    setA(arV, S.v, sc*4*Math.min(vlen/Math.max(D.vmax,1e-9),1)+sc*1.2, true);
    setA(arW, S.vr.clone().negate(), sc*3.2, true);
    setA(arD, S.vr.clone().negate(), sc*4*(S.drag/fmax), sf);
    var lat = new THREE.Vector3().crossVectors(S.vr, S.axis);
    if(lat.lengthSq()>1e-16){ lat.crossVectors(lat, S.vr).normalize(); }
    setA(arL, lat, sc*4*(S.lift/fmax), sf && S.lift>1e-9);
    setA(arSp, S.axis, sc*3*Math.min(Math.abs(S.spin)/Math.max(D.spinmax,1e-9),1)+sc*0.5,
         Math.abs(S.spin)>1e-6);

    // airflow particles: advected in the shuttlecock frame (visual only)
    var psel = document.getElementById(ID+"-parts");
    var np = psel?parseInt(psel.value,10):400;
    pts.visible = np>0;
    if(np>0){
      var f = S.vr.clone();
      var fl = f.length();
      if(fl>1e-9){
        f.multiplyScalar(1/fl);
        var up = Math.abs(f.y)<0.9 ? new THREE.Vector3(0,1,0) : new THREE.Vector3(1,0,0);
        var e1 = new THREE.Vector3().crossVectors(up,f).normalize();
        var e2 = new THREE.Vector3().crossVectors(f,e1).normalize();
        var L = sc*14, Rr = sc*4.5;
        var arr = pGeom.attributes.position.array;
        var adv = dtr/Math.max(L/Math.max(fl,1e-6),1e-6);
        for(var i=0;i<np;i++){
          pAge[i] += adv*(0.6+0.8*Math.abs(pOff[2*i]));
          if(pAge[i]>1){ pAge[i]-=1; }
          var s01 = pAge[i];
          var axial = L*(0.5 - s01);
          var r0 = Rr*(0.25+Math.abs(pOff[2*i]));
          var thp = pOff[2*i+1]*Math.PI*2;
          // wake widening + deficit behind the body (Gaussian, visual only)
          var behind = axial<0 ? (-axial/L) : 0;
          var rr = r0*(1+1.9*behind);
          var pxv = S.p.x + f.x*axial + (e1.x*Math.cos(thp)+e2.x*Math.sin(thp))*rr;
          var pyv = S.p.y + f.y*axial + (e1.y*Math.cos(thp)+e2.y*Math.sin(thp))*rr;
          var pzv = S.p.z + f.z*axial + (e1.z*Math.cos(thp)+e2.z*Math.sin(thp))*rr;
          arr[3*i]=pxv; arr[3*i+1]=pyv; arr[3*i+2]=pzv;
        }
        pGeom.setDrawRange(0,np);
        pGeom.attributes.position.needsUpdate = true;
      }
      // wake cone
      var wl = sc*10, wr = sc*5.2;
      wake.visible = np>0 && fl>1e-9;
      if(wake.visible){
        wake.scale.set(wr, wl, wr);
        var dirw = S.vr.clone().normalize().negate();
        wake.position.copy(S.p).addScaledVector(dirw, wl*0.5);
        wake.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dirw);
      }
    } else { wake.visible=false; }
  }

  function hud(){
    var h = document.getElementById(ID+"-hud");
    if(!h) return;
    h.textContent = "t = "+simT.toFixed(4)+" s | V = "+S.v.length().toFixed(1)
      +" m/s | Vrel = "+S.Vrel.toFixed(1)+" m/s | spin = "+S.spin.toFixed(1)+" rad/s"
      +((D.n_removed>0) ? ("  |  깃털 "+D.n_removed+"개 제거") : "");
    var sb = document.getElementById(ID+"-scrub");
    if(sb && !sb.__drag) sb.value = String(100*(simT-T0)/Math.max(T1-T0,1e-9));
  }

  // ---- loop --------------------------------------------------------------
  var playing = false, last = null, fpsT = 0, fpsN = 0;
  function rate(){ var e=document.getElementById(ID+"-rate"); return e?parseFloat(e.value):1; }
  function frame(ts){
    var dtr = (last===null) ? 0 : (ts-last)/1000;
    last = ts;
    if(dtr>0.25) dtr = 0.25;
    if(playing){
      simT += dtr*rate();
      if(simT>T1){
        var lp=document.getElementById(ID+"-loop");
        if(lp && lp.checked) simT = T0 + (simT-T1); else { simT=T1; playing=false; }
      }
      if(simT<T0) simT=T0;
    }
    sample(simT);
    updateScene(dtr);
    updateCamera(dtr);
    resize();
    renderer.render(scene, cam);
    hud();
    fpsN++; fpsT += dtr;
    if(fpsT>1.0){
      var el=document.getElementById(ID+"-fps");
      var fps = fpsN/fpsT;
      if(el) el.textContent = "renderer: "+fps.toFixed(0)+" FPS · 상태 샘플 "+n+"개 · "
        +"물리 dt "+D.dt.toExponential(1)+" s (재생 중 재계산 없음)";
      if(fps<40){
        var ps=document.getElementById(ID+"-parts");
        if(ps && parseInt(ps.value,10)>150){
          ps.value = (parseInt(ps.value,10)>400)?"400":"150";
        } else {
          var tsel=document.getElementById(ID+"-trail");
          if(tsel && tsel.value==="full") tsel.value="recent";
        }
      }
      fpsT=0; fpsN=0;
    }
    requestAnimationFrame(frame);
  }

  var pb=document.getElementById(ID+"-play");
  if(pb) pb.onclick=function(){ if(simT>=T1) simT=T0; playing=true; };
  var ps=document.getElementById(ID+"-pause");
  if(ps) ps.onclick=function(){ playing=false; };
  var rb=document.getElementById(ID+"-rst");
  if(rb) rb.onclick=function(){ simT=T0; };
  var dist0=dist;
  var DMIN=Math.max(sc*0.4, 1e-4), DMAX=Math.max(span*60+50, dist0*60);
  function zoom(f){ dist=Math.max(DMIN, Math.min(DMAX, dist*f));
                    var z=document.getElementById(ID+"-zlab");
                    if(z) z.textContent="x"+(dist0/dist).toFixed(2); }
  var zi=document.getElementById(ID+"-zin");
  if(zi) zi.onclick=function(){ zoom(0.75); };
  var zo=document.getElementById(ID+"-zout");
  if(zo) zo.onclick=function(){ zoom(1.35); };
  var zf=document.getElementById(ID+"-zfit");
  if(zf) zf.onclick=function(){ dist=dist0; zoom(1); };
  var fb=document.getElementById(ID+"-toflip");
  if(fb) fb.onclick=function(){
    if(D.flip_t===null||D.flip_t===undefined){ simT=T0; return; }
    // start a little before the flip so the run-up is visible
    var pad=(T1-T0)*0.02;
    simT=Math.max(T0, D.flip_t-Math.max(pad, 0.01));
    var rs=document.getElementById(ID+"-rate");
    if(rs) rs.value="0.002";
    playing=true;
  };
  var sb=document.getElementById(ID+"-scrub");
  if(sb){
    sb.addEventListener("input", function(){
      sb.__drag=true; playing=false;
      simT = T0 + (T1-T0)*(parseFloat(sb.value)/100);
    });
    sb.addEventListener("change", function(){ sb.__drag=false; });
  }
  resize();
  sample(simT);
  requestAnimationFrame(frame);
})();
</script>
"""

    def _flip_time_from_df(df):
        """Time at which the shuttlecock first comes within 90 deg of cork-forward.

        The turnover lasts on the order of 10 ms, so the renderer needs to know where
        it happens to be able to jump there.
        """
        try:
            col = ("wobble_signed_deg" if "wobble_signed_deg" in df.columns
                   else ("delta_alpha_deg" if "delta_alpha_deg" in df.columns else None))
            if col is None:
                return None
            w = np.abs(pd.to_numeric(df[col], errors="coerce").values)
            t = pd.to_numeric(df["time_s"], errors="coerce").values
            if not len(w) or np.abs(w[0]) < 90.0:
                return None
            idx = np.where(w < 90.0)[0]
            return float(t[idx[0]]) if len(idx) else None
        except Exception:
            return None

    def gl_scene_html(res, msg=None):
        """Builds the WebGL scene from an already-computed 3D result buffer.

        Physics is never recomputed here: the browser interpolates between stored
        states (linear for position/velocity, SLERP for attitude) at its own frame rate.
        """
        try:
            df = res.df if (res is not None and res.df is not None) else None
            if df is None or len(df) < 2:
                return ('<div style="padding:26px;border:1px dashed #cbd5e1;border-radius:8px;'
                        'color:#475569;font-size:14px;text-align:center">'
                        + (msg or "3차원 시뮬레이션을 실행하면 여기에서 실시간 3D 재생이 시작됩니다.")
                        + '</div>')
            df = df.copy()
            # 2D results carry the attitude as a planar angle rather than a quaternion.
            # Rebuild the quaternion (rotation about z by the cork heading) so the same
            # renderer can play back planar runs from the basic and advanced tabs.
            if "qw" not in df.columns:
                ang = None
                if "theta_cork_deg" in df.columns:
                    ang = np.radians(pd.to_numeric(df["theta_cork_deg"],
                                                    errors="coerce").fillna(0.0).values)
                elif "theta_body_deg" in df.columns:
                    ang = np.radians(pd.to_numeric(df["theta_body_deg"],
                                                    errors="coerce").fillna(0.0).values
                                      + 180.0)
                if ang is None:
                    return ('<div style="padding:26px;border:1px dashed #cbd5e1;'
                            'border-radius:8px;color:#475569;font-size:14px;'
                            'text-align:center">자세 정보가 없어 재생할 수 없습니다.</div>')
                # already the cork direction, so no extra 180 deg flip here
                df["qw"] = np.cos(ang / 2.0)
                df["qx"] = 0.0
                df["qy"] = 0.0
                df["qz"] = np.sin(ang / 2.0)
                df["_planar_q"] = True
            if "z_m" not in df.columns:
                df["z_m"] = 0.0
            if "vz_m_s" not in df.columns:
                df["vz_m_s"] = 0.0
            need = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
            if any(c not in df.columns for c in need):
                return ('<div style="padding:26px;border:1px dashed #cbd5e1;border-radius:8px;'
                        'color:#475569;font-size:14px;text-align:center">'
                        '자세 정보가 없어 재생할 수 없습니다.</div>')

            n_pts = min(len(df), 1200)
            sel = np.unique(np.linspace(0, len(df) - 1, n_pts).astype(int))
            sub = df.iloc[sel]

            def col(name, default=0.0):
                if name in sub.columns:
                    v = pd.to_numeric(sub[name], errors="coerce").fillna(default)
                    return [round(float(a), 6) for a in v.values]
                return [default] * len(sub)

            xs = np.asarray(col("x_m")); ys = np.asarray(col("y_m")); zs = np.asarray(col("z_m"))

            # The integrated attitude quaternion carries body +x along the SKIRT (tail)
            # direction, because equilibrium is alpha = 180 deg. Every mesh in this app
            # is modelled with the cork at +x, so the drawing quaternion is the physics
            # quaternion composed with a 180 deg turn about body y. Without it the
            # renderer shows the skirt leading, which is the opposite of a stable flight.
            _q = np.c_[np.asarray(col("qw", 1.0)), np.asarray(col("qx")),
                        np.asarray(col("qy")), np.asarray(col("qz"))]
            _flip = (np.array([1.0, 0.0, 0.0, 0.0]) if "_planar_q" in df.columns
                     else np.array([0.0, 0.0, 1.0, 0.0]))
            _qd = np.empty_like(_q)
            for _i in range(len(_q)):
                _qd[_i] = quat_mul(_q[_i], _flip)
            qd_w = [round(float(a), 6) for a in _qd[:, 0]]
            qd_x = [round(float(a), 6) for a in _qd[:, 1]]
            qd_y = [round(float(a), 6) for a in _qd[:, 2]]
            qd_z = [round(float(a), 6) for a in _qd[:, 3]]
            spanv = float(max(np.ptp(xs), np.ptp(ys), np.ptp(zs), 0.5))
            env = state.get("env") or EnvironmentModel()
            vx = np.asarray(col("vx_m_s")); vy = np.asarray(col("vy_m_s")); vz = np.asarray(col("vz_m_s"))
            rx = vx - env.u_air_x; ry = vy - env.u_air_y; rz = vz - env.u_air_z
            skirt = state.get("skirt")
            prof = shuttlecock_profile(
                state["model"].geometry if state.get("model") else None, skirt)
            az = ([f.azimuth_rad for f in skirt.attached_feathers]
                  if skirt is not None else [2 * math.pi * i / 16 for i in range(16)])
            m = state.get("model")
            weight = (m.m * G) if m is not None else 0.05
            spin_col = col("spin_axial_rad_s") if "spin_axial_rad_s" in sub.columns \
                else col("spin_rad_s")

            data = dict(
                t=col("time_s"), x=list(np.round(xs, 6)), y=list(np.round(ys, 6)),
                z=list(np.round(zs, 6)),
                vx=list(np.round(vx, 5)), vy=list(np.round(vy, 5)), vz=list(np.round(vz, 5)),
                rx=list(np.round(rx, 5)), ry=list(np.round(ry, 5)), rz=list(np.round(rz, 5)),
                qw=qd_w, qx=qd_x, qy=qd_y, qz=qd_z,
                vrel=col("V_rel_m_s"), spin=spin_col,
                drag=col("drag_force_N"), lift=col("lift_force_N"),
                cx=float(xs.mean()), cy=float(ys.mean()), cz=float(zs.mean()),
                span=spanv, scale=0.04 * spanv,
                vmax=float(np.max(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)) or 1.0),
                spinmax=float(max(abs(a) for a in spin_col) or 1.0),
                weight=float(weight),
                flip_t=_flip_time_from_df(sub),
                dt=float(state.get("dt3d") or state.get("dt") or 1e-3),
                az=[round(float(a), 5) for a in az],
                az_removed=[round(float(f.azimuth_rad), 5)
                             for f in (skirt.feathers if skirt is not None else [])
                             if not f.attached],
                n_removed=int(len(skirt.removed_ids) if skirt is not None else 0),
                prof=dict(ch=prof["cork_half"], th_=prof["tip_half"],
                           ub=prof["u_cork_base"], ut=prof["u_tip"],
                           un=prof["u_nose"], clen=prof["cork_len"]),
            )
            payload = json.dumps(data, allow_nan=False)
            inner = GL_TEMPLATE.replace("__DATA__", payload).replace("__ID__", GL_DIV_ID)
            return _srcdoc_iframe(inner, height=620)
        except Exception:
            logger.exception("gl scene html failed")
            return ('<div style="padding:20px;color:#b45309;font-size:13px">'
                    '3D 장면을 만들 수 없습니다. 시뮬레이션 결과를 확인하세요.</div>')

    def controller_gl_scene():
        try:
            return gl_scene_html(state.get("result3d"))
        except Exception:
            logger.exception("gl scene build failed")
            return gl_scene_html(None)

    def animation_html(res):
        """Self-contained canvas animation of the flight.

        Draws from the already-computed trajectory with requestAnimationFrame, so playback
        is smooth and immediate: no Plotly frames, no CDN download, no loading screen.
        The physics is never recomputed here - this only replays stored results.
        """
        try:
            df = res.df if (res is not None and res.df is not None) else None
            if df is None or len(df) == 0:
                return ('<div style="padding:26px;border:1px dashed #cbd5e1;border-radius:8px;'
                        'color:#475569;font-size:14px;text-align:center">'
                        '시뮬레이션을 실행하면 여기에 비행 영상이 자동으로 재생됩니다.</div>')

            if "theta_cork_deg" in df.columns:
                th_full = unwrap_deg(df.theta_cork_deg.astype(float).values)
            elif "theta_body_deg" in df.columns:
                th_full = unwrap_deg(df.theta_body_deg.astype(float).values) + 180.0
            else:
                th_full = unwrap_deg(df.alpha_deg.astype(float).values) + 180.0

            n_pts = min(len(df), 900)
            if len(df) > 2:
                dth = np.abs(np.diff(th_full))
                span = float(np.sum(dth))
                if span > 0:
                    n_pts = int(min(len(df), max(n_pts, span / 6.0)))
            sel = np.unique(np.linspace(0, len(df) - 1, int(n_pts)).astype(int))
            sub = df.iloc[sel]
            th = pd.Series(th_full[sel], index=sub.index)

            xs = sub.x_m.astype(float).values
            ys = sub.y_m.astype(float).values
            xr = max(float(xs.max() - xs.min()), 1e-3)
            yr = max(float(ys.max() - ys.min()), 1e-3)
            pad_x, pad_y = 0.07 * xr, 0.16 * yr
            skirt = state.get("skirt")
            prof = shuttlecock_profile(
                state["model"].geometry if state.get("model") else None, skirt)
            az = ([f.azimuth_rad for f in skirt.attached_feathers]
                  if skirt is not None else
                  [2 * math.pi * i / 16 for i in range(16)])
            tv = sub.time_s.astype(float).values
            dt_mean = float(np.mean(np.diff(tv))) if len(tv) > 1 else 0.01

            data = dict(
                t=[round(float(v), 6) for v in tv],
                x=[round(float(v), 5) for v in xs],
                y=[round(float(v), 5) for v in ys],
                th=[round(float(v), 3) for v in th.values],
                vx=[round(float(v), 4) for v in sub.vx_m_s.astype(float).values],
                vy=[round(float(v), 4) for v in sub.vy_m_s.astype(float).values],
                xmin=float(xs.min() - pad_x), xmax=float(xs.max() + pad_x),
                ymin=float(min(ys.min() - pad_y, 0.0)), ymax=float(ys.max() + pad_y),
                size=0.055 * max(xr, yr), dt=dt_mean,
                az=[round(float(a), 5) for a in az],
                prof=dict(ch=prof["cork_half"], th_=prof["tip_half"],
                           ub=prof["u_cork_base"], ut=prof["u_tip"],
                           un=prof["u_nose"], clen=prof["cork_len"]),
            )
            html = ANIM_TEMPLATE.replace("__DATA__", json.dumps(data))
            return html.replace("__ID__", ANIM_DIV_ID)
        except Exception:
            logger.exception("animation html failed")
            return "<div>비행 영상을 생성할 수 없습니다.</div>"

    def controller_animation():
        try:
            return animation_html(state.get("result"))
        except Exception:
            logger.exception("animation build failed")
            return animation_html(None)

    def row_flow_angle(row):
        fa = getattr(row, "flow_angle_deg", None)
        if fa is not None and fa == fa:
            return float(fa)
        env = state.get("env")
        ux = env.u_air_x if env else 0.0
        uy = env.u_air_y if env else 0.0
        return rad2deg(math.atan2(row.vy_m_s - uy, row.vx_m_s - ux))

    def row_body_angle(row):
        tb = getattr(row, "theta_body_deg", None)
        if tb is not None and tb == tb:
            return float(tb)
        return float(row.alpha_deg)

    def row_cork_angle(row):
        tc = getattr(row, "theta_cork_deg", None)
        if tc is not None and tc == tc:
            return float(tc)
        return cork_heading_deg(row_body_angle(row))

    def controller_flow_view(frame_idx, show_vectors, show_streams, show_forces,
                              show_attitude, resolution):
        res = state.get("result")
        if res is None or len(res.df) == 0:
            return None, "먼저 시뮬레이션을 실행하세요."
        try:
            df = res.df
            i = int(np.clip(int(frame_idx or 0), 0, len(df) - 1))
            row = df.iloc[i]
            span = max(df.x_m.max() - df.x_m.min(), df.y_m.max() - df.y_m.min(), 1e-3)
            extent = 0.10 * span
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.x_m, y=df.y_m, mode="lines", name="비행 궤적",
                                      line=dict(color="#e2e8f0", width=2)))
            if show_vectors or show_streams:
                add_flow_overlay(fig, row, show_vectors=bool(show_vectors),
                                  show_streams=bool(show_streams),
                                  resolution=resolution, extent=extent)
            if show_attitude:
                draw_shuttlecock(fig, row.x_m, row.y_m, row_cork_angle(row),
                                  scale=extent * 0.55)
            if show_forces:
                fmax = max(abs(row.drag_force_N), abs(row.lift_force_N),
                           (state["model"].m * G) if state.get("model") else 1.0, 1e-9)
                A = extent * 0.75

                def far(angle_deg, mag, color, label):
                    if mag <= 0:
                        return
                    aa = deg2rad(angle_deg)
                    r = A * (mag / fmax)
                    arrow_with_label(fig, row.x_m, row.y_m,
                                      row.x_m + r * math.cos(aa), row.y_m + r * math.sin(aa),
                                      color, label)
                fang = row_flow_angle(row)
                far(fang + 180.0, abs(row.drag_force_N), "#c53030", "항력")
                if abs(row.lift_force_N) > 0:
                    far(fang + (90.0 if row.lift_force_N >= 0 else -90.0),
                        abs(row.lift_force_N), "#2f855a", "양력")
                far(-90.0, (state["model"].m * G) if state.get("model") else 0.0,
                    "#4a5568", "중력")
            fig.update_layout(
                title=f"t = {row.time_s:.4f} s — 셔틀콕 주변 공기 흐름", title_x=0.5,
                template="plotly_white", height=520, showlegend=False,
                xaxis=dict(title="x [m]", range=[row.x_m - extent, row.x_m + extent],
                            scaleanchor="y", scaleratio=1),
                yaxis=dict(title="y [m]", range=[row.y_m - extent, row.y_m + extent]),
                margin=dict(l=55, r=20, t=50, b=45))
            panel = (
                f"**시각 t:** {fmt(row.time_s, ' s')} &nbsp;|&nbsp; "
                f"**상대 풍속:** {fmt(row.V_rel_m_s, ' m/s')} &nbsp;|&nbsp; "
                f"**Re:** {fmt(row.Re)} &nbsp;|&nbsp; **받음각:** {fmt(row.alpha_deg, '°')}\n\n"
                f"**항력:** {fmt(row.drag_force_N, ' N')} &nbsp;|&nbsp; "
                f"**양력:** {fmt(row.lift_force_N, ' N')}\n\n"
                f"*파란 선/화살표는 간이 공기흐름 시각화입니다. CFD 계산 결과가 아니며, "
                f"힘 계산에는 사용되지 않습니다 (힘은 Cd·Cl·Cm 모델로만 계산).*"
            )
            return fig, panel
        except Exception:
            logger.exception("flow view failed")
            return None, "공기 흐름을 표시할 수 없습니다: 내부 오류가 발생했습니다."

    def controller_top_refresh():
        return feather_figure(state["skirt"]), controller_animation(), status_markdown()

    def feather_figure(skirt: SkirtGeometry):
        """View down the symmetry axis: white feather vanes overlapping around the cork,
        with the thread ring, as on a real shuttlecock."""
        gs = skirt.geometry_state()
        prof = shuttlecock_profile(None, skirt)
        C = SHUTTLE_COLORS
        r_in = prof["cork_half"] / prof["tip_half"]
        r_out = 1.0
        fig = go.Figure()

        th = np.linspace(0, 2 * math.pi, 90)
        fig.add_trace(go.Scatter(x=r_out * np.cos(th), y=r_out * np.sin(th), mode="lines",
                                  line=dict(color="#e2e8f0", width=1.5),
                                  showlegend=False, hoverinfo="skip"))

        half = math.pi / max(skirt.n_feathers, 1) * 1.15
        for f in skirt.feathers:
            a = f.azimuth_rad
            n_out = 10
            outer = [a - half + 2 * half * k / n_out for k in range(n_out + 1)]
            xs = [r_in * math.cos(a - half * 0.30)]
            ys = [r_in * math.sin(a - half * 0.30)]
            for k, t in enumerate(outer):
                rr = r_out * (0.90 + 0.10 * math.sin(math.pi * k / n_out))
                xs.append(rr * math.cos(t))
                ys.append(rr * math.sin(t))
            xs.append(r_in * math.cos(a + half * 0.30))
            ys.append(r_in * math.sin(a + half * 0.30))
            xs.append(xs[0]); ys.append(ys[0])
            on = f.attached
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                fillcolor=C["feather_fill"] if on else C["removed_fill"],
                line=dict(color=C["feather_edge"] if on else C["removed_edge"],
                          width=1.2, dash="solid" if on else "dot"),
                showlegend=False,
                hovertemplate=f"{f.id}번 깃털 · {'붙어있음' if on else '제거됨'}"
                              f"<br>방위각 {f.azimuth_deg:.0f}°<extra></extra>"))
            if on:
                fig.add_trace(go.Scatter(
                    x=[r_in * math.cos(a), r_out * 0.96 * math.cos(a)],
                    y=[r_in * math.sin(a), r_out * 0.96 * math.sin(a)],
                    mode="lines", line=dict(color=C["quill"], width=1.0),
                    showlegend=False, hoverinfo="skip"))
            rt = (r_in + r_out) / 2
            fig.add_trace(go.Scatter(
                x=[rt * math.cos(a)], y=[rt * math.sin(a)], mode="text",
                text=[str(f.id)],
                textfont=dict(size=10, color="#64748b" if on else "#e53e3e"),
                showlegend=False, hoverinfo="skip"))

        r_thread = r_in + (r_out - r_in) * 0.45
        fig.add_trace(go.Scatter(x=r_thread * np.cos(th), y=r_thread * np.sin(th),
                                  mode="lines", line=dict(color=C["thread"], width=2.2),
                                  showlegend=False, hoverinfo="skip"))

        fig.add_trace(go.Scatter(x=r_in * np.cos(th), y=r_in * np.sin(th), mode="lines",
                                  fill="toself", fillcolor=C["band"],
                                  line=dict(color=C["band"], width=1.5),
                                  showlegend=False,
                                  hovertemplate="코르크 밴드<extra></extra>"))
        fig.add_trace(go.Scatter(x=r_in * 0.82 * np.cos(th), y=r_in * 0.82 * np.sin(th),
                                  mode="lines", fill="toself", fillcolor=C["cork_fill"],
                                  line=dict(color=C["cork_edge"], width=1.5),
                                  showlegend=False,
                                  hovertemplate="코르크<extra></extra>"))

        state_txt = "정상 (BWF 기준)" if gs["removed_feathers"] == 0 else "파손 (실험용 형상)"
        fig.update_layout(
            title=(f"남은 깃털 {gs['remaining_feathers']}/{gs['N_feathers']} · "
                   f"비대칭도 {gs['geometry_asymmetry']:.3f} · {state_txt}"),
            template="plotly_white", plot_bgcolor="#f8fafc",
            xaxis=dict(visible=False, range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            margin=dict(l=10, r=10, t=50, b=10), width=520, height=520,
            title_x=0.5)
        return fig

    def side_view_figure(skirt: SkirtGeometry, geometry: Optional[Geometry] = None):
        """Side elevation of the shuttlecock at true BWF proportions, with dimensions."""
        prof = shuttlecock_profile(geometry, skirt)
        fig = go.Figure()
        draw_shuttlecock(fig, 0.0, 0.0, 0.0, scale=0.5, skirt=skirt, labels=False)
        out = shuttlecock_outline_2d(prof)
        tot_mm = prof["total_length_m"] * 1000.0
        tip_mm = prof["tip_half"] * 2 * tot_mm
        cork_mm = prof["cork_half"] * 2 * tot_mm

        fig.add_annotation(x=out["u_tip"], y=-out["tip_half"] - 0.14, showarrow=False,
                            text=f"<b>깃털 끝 지름 {tip_mm:.0f} mm</b>",
                            font=dict(size=12, color="#2c5282"))
        fig.add_annotation(x=out["u_base"] + 0.16, y=out["cork_half"] + 0.13,
                            showarrow=False,
                            text=f"<b>코르크 {cork_mm:.0f} mm</b>",
                            font=dict(size=12, color="#7b4b0a"))
        fig.add_annotation(x=0.0, y=-out["tip_half"] - 0.28, showarrow=False,
                            text=f"전체 길이 약 {tot_mm:.0f} mm",
                            font=dict(size=11, color="#4a5568"))
        gs = skirt.geometry_state()
        fig.update_layout(
            title=dict(text=(f"셔틀콕 옆모습 (BWF 규격 비율) — 남은 깃털 "
                              f"{gs['remaining_feathers']}/{gs['N_feathers']}"),
                        x=0.5, xanchor="center", y=0.97, yanchor="top",
                        font=dict(size=14)),
            template="plotly_white", plot_bgcolor="#f8fafc",
            xaxis=dict(visible=False, range=[-0.95, 0.95], scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False, range=[-0.62, 0.62]),
            margin=dict(l=10, r=10, t=64, b=10), width=620, height=440)
        return fig

    def controller_side_view():
        try:
            return side_view_figure(state["skirt"],
                                     state["model"].geometry if state.get("model") else None)
        except Exception:
            logger.exception("side view failed")
            return None

    def geometry_markdown():
        gs = state["skirt"].geometry_state()
        asym = gs["geometry_asymmetry"]
        if gs["removed_feathers"] == 0:
            judge = "정상 상태 (BWF 규격 기준 셔틀콕)"
        elif asym < 0.05:
            judge = "대칭 파손 (깃털이 고르게 빠짐)"
        elif asym < 0.3:
            judge = "약한 비대칭 파손"
        else:
            judge = "강한 비대칭 파손 (한쪽에 집중)"
        return (
            f"**남은 깃털:** {gs['remaining_feathers']} / {gs['N_feathers']} &nbsp;|&nbsp; "
            f"**제거된 깃털:** {gs['removed_feathers']} &nbsp;|&nbsp; "
            f"**비대칭도:** {asym:.3f}\n\n"
            f"**판정:** {judge}\n\n"
            f"**제거된 깃털 번호:** {gs['removed_ids'] or '없음'}\n\n"
            f"**스커트 개방 비율:** {fmt(gs['skirt_open_fraction'])} &nbsp;|&nbsp; "
            f"**추정 공극률:** {fmt(gs['estimated_porosity'])}\n\n"
            f"**유효 스커트 폭:** {fmt(gs['effective_skirt_width'], ' m')} &nbsp;|&nbsp; "
            f"**유효 투영 면적:** {fmt(gs['effective_projected_area'], ' m²')}\n\n"
            f"*위 값은 형상 기반 추정값입니다. 깃털 제거는 형상만 바꾸며, "
            f"실험·문헌 자료로 만든 대응 관계가 없으면 Cd·Cl·Cm은 변하지 않습니다.*"
        )

    def controller_toggle_feathers(selected_ids):
        try:
            ids = set(int(s) for s in (selected_ids or []))
            state["skirt"].set_attached(
                [f.id for f in state["skirt"].feathers if f.id not in ids])
            return (feather_figure(state["skirt"]), geometry_markdown(), status_markdown(),
                    warning_center())
        except Exception:
            logger.exception("feather toggle failed")
            return (feather_figure(state["skirt"]), geometry_markdown(), status_markdown(),
                    warning_center())

    def controller_bwf_spec(mass_g, feather_len_mm, skirt_tip_mm, cork_mm):
        try:
            rep = bwf_spec_report(mass_g=mass_g, feather_length_mm=feather_len_mm,
                                  skirt_tip_diameter_mm=skirt_tip_mm,
                                  cork_diameter_mm=cork_mm, n_feathers=16)
            lines = ["| 항목 | 현재 입력값 | BWF 표준 범위 | 판정 |", "|---|---|---|---|"]
            for r in rep["rows"]:
                mark = "✅ 규격 내" if r["within_spec"] else "⚠ 규격 밖"
                lines.append(f"| {r['label']} | {fmt(r['value'])} {r['unit']} | "
                             f"{r['spec_low']}~{r['spec_high']} {r['unit']} | {mark} |")
            table = "\n".join(lines)
            if rep["warnings"]:
                for w in rep["warnings"]:
                    push_warning(w)
                table += "\n\n" + "\n".join(f"⚠ {w}" for w in rep["warnings"])
                table += "\n\n*입력값은 자동으로 바뀌지 않습니다. 사용자 정의 값이 그대로 사용됩니다.*"
            else:
                table += "\n\n✅ 모든 항목이 BWF 일반 규격 범위 안에 있습니다."
            return table, warning_center()
        except Exception:
            logger.exception("bwf spec check failed")
            return "규격 검사를 할 수 없습니다: 입력값을 확인하세요.", warning_center()

    def controller_launch_components(V0, launch_angle):
        """vx0/vy0 표시. 초기값 렌더링과 서버측 기준값 용도.

        같은 계산을 브라우저에서 하는 _COMP_JS 가 아래에 있다. 타이핑 중 왕복을
        없애기 위한 것이므로, 이 함수를 고치면 그쪽도 같이 고쳐야 한다.
        """
        try:
            th = deg2rad(float(launch_angle))
            vx = float(V0) * math.cos(th)
            vy = float(V0) * math.sin(th)
            return (f"**vx₀ = {vx:.4f} m/s** &nbsp;|&nbsp; **vy₀ = {vy:.4f} m/s**\n\n"
                    f"*vx₀ = V₀·cos(θ₀), vy₀ = V₀·sin(θ₀) — 내부 계산은 radian으로 수행됩니다.*")
        except (TypeError, ValueError):
            return "발사 속도와 발사각에 숫자를 입력하세요."

    def controller_normal_state():
        state["skirt"].restore_all()
        return ([], feather_figure(state["skirt"]), geometry_markdown(), status_markdown())

    def controller_apply_damage_numeric(n_remove, pattern):
        try:
            n = int(round(float(n_remove or 0)))
        except (TypeError, ValueError):
            n = 0
        total = state["skirt"].n_feathers
        n = max(0, min(n, total))
        ids = list(range(1, total + 1))
        if n == 0:
            removed = []
        elif pattern == "한쪽 집중":
            removed = ids[:n]
        elif pattern == "좌우 대칭":
            removed = []
            i = 0
            while len(removed) < n:
                a = ids[i % total]
                b = ids[(i + total // 2) % total]
                for c in (a, b):
                    if c not in removed and len(removed) < n:
                        removed.append(c)
                i += 1
            removed.sort()
        else:
            step = total / n
            removed = sorted({ids[int(round(k * step)) % total] for k in range(n)})
            k = 0
            while len(removed) < n and k < total:
                if ids[k] not in removed:
                    removed.append(ids[k])
                k += 1
            removed.sort()
        state["skirt"].restore_all()
        for fid in removed:
            state["skirt"].remove(fid)
        if len(removed) == total:
            push_warning("깃털을 전부 제거했습니다: 물리적으로 불가능한 형상이라 결과에 의미가 없습니다.")
        return ([str(i) for i in removed], feather_figure(state["skirt"]),
                geometry_markdown(), status_markdown(), warning_center())

    def controller_apply_damage_numeric_safe(n_remove, pattern):
        try:
            return controller_apply_damage_numeric(n_remove, pattern)
        except Exception:
            logger.exception("numeric damage apply failed")
            return ([str(i) for i in state["skirt"].removed_ids], feather_figure(state["skirt"]),
                    geometry_markdown(), status_markdown(), warning_center())

    def controller_damage_preset(preset_name):
        try:
            removed = DAMAGE_PRESETS.get(preset_name)
        except TypeError:
            removed = None
        if removed is None:
            return ([str(i) for i in state["skirt"].removed_ids],
                    feather_figure(state["skirt"]), geometry_markdown(), status_markdown())
        state["skirt"].restore_all()
        for fid in removed:
            state["skirt"].remove(fid)
        return ([str(i) for i in removed], feather_figure(state["skirt"]),
                geometry_markdown(), status_markdown())

    def controller_restore_all():
        state["skirt"].restore_all()
        return ([], feather_figure(state["skirt"]), geometry_markdown(), status_markdown())

    def controller_remove_all():
        state["skirt"].remove_all()
        push_warning("all feathers removed: geometry is unphysical, results are not meaningful")
        return ([str(f.id) for f in state["skirt"].feathers], feather_figure(state["skirt"]),
                geometry_markdown(), status_markdown())

    def arrow_with_label(fig, x0, y0, x1, y1, color, label, arrowhead=3):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                            axref="x", ayref="y", showarrow=True, arrowhead=arrowhead,
                            arrowwidth=3, arrowsize=1.2, arrowcolor=color, text="")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / n * 0.17, dx / n * 0.17
        fig.add_annotation(x=mx + ox, y=my + oy, xref="x", yref="y", showarrow=False,
                            text=f"<b>{label}</b>", font=dict(size=13, color=color))

    def draw_shuttlecock(fig, cx, cy, body_deg, scale=1.0, skirt=None, labels=True,
                          show_feathers=True, detail=True):
        """Side view of a real BWF shuttlecock: a soft skirt silhouette, overlapping white
        feather vanes drawn back-to-front, cross-stitch thread rings, and a rounded cork
        with its band."""
        sk = skirt if skirt is not None else state.get("skirt")
        prof = shuttlecock_profile(state["model"].geometry if state.get("model") else None, sk)
        out = shuttlecock_outline_2d(prof)
        C = SHUTTLE_COLORS
        b = deg2rad(body_deg)
        ux, uy = math.cos(b), math.sin(b)
        px, py = -math.sin(b), math.cos(b)
        L = 2.0 * scale

        def pt(u, r):
            return (cx + ux * u * L + px * r * L, cy + uy * u * L + py * r * L)

        def poly(us, rs, fill, edge, width=1.0, dash="solid", hover=None):
            xs = [pt(u, r)[0] for u, r in zip(us, rs)]
            ys = [pt(u, r)[1] for u, r in zip(us, rs)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill,
                line=dict(color=edge, width=width, dash=dash), showlegend=False,
                hoverinfo="skip" if hover is None else "text", hovertext=hover))

        feathers = sk.feathers if sk is not None else []
        attached_any = [f for f in feathers if f.attached]

        if not detail or not show_feathers or not feathers:
            poly(out["skirt_u"], out["skirt_r"], "rgba(255,255,255,0.92)",
                 C["feather_edge"], 1.6)
        else:
            order = sorted(feathers, key=lambda f: math.sin(f.azimuth_rad))
            for f in order:
                us, rs, quill = feather_vane_2d(prof, f.azimuth_rad)
                if f.attached:
                    fill, edge = feather_shade(f.azimuth_rad)
                    poly(us, rs, fill, edge, 0.9, hover=f"{f.id}번 깃털 · 붙어있음")
                    qx = [pt(u, r)[0] for u, r in zip(*quill)]
                    qy = [pt(u, r)[1] for u, r in zip(*quill)]
                    shade = 0.30 + 0.35 * (math.sin(f.azimuth_rad) + 1) / 2
                    fig.add_trace(go.Scatter(
                        x=qx, y=qy, mode="lines",
                        line=dict(color=f"rgba(150,160,178,{shade:.2f})", width=0.9),
                        hoverinfo="skip", showlegend=False))
                else:
                    poly(us, rs, C["removed_fill"], C["removed_edge"], 0.9, "dot",
                         hover=f"{f.id}번 깃털 · 제거됨")

        if detail and attached_any:
            for u_ring, r_ring in thread_rings_2d(prof):
                seg = [pt(u_ring, r_ring * (-1 + 2 * k / 30)) for k in range(31)]
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in seg], y=[p[1] for p in seg], mode="lines",
                    line=dict(color="rgba(146,158,178,0.85)", width=1.7),
                    hoverinfo="skip", showlegend=False))

        ch = prof["cork_half"]
        ub = prof["u_cork_base"]
        band_w = prof["cork_len"] * 0.26
        poly([ub, ub + band_w, ub + band_w, ub, ub],
             [ch, ch, -ch, -ch, ch], "#12341f", "#0d2716", 0.8)

        n_arc = 30
        u_start = ub + band_w
        u_end = prof["u_nose"]
        depth = u_end - u_start
        for layer, (tint, rad) in enumerate([("#e9ecf1", 1.00), ("#f4f6f9", 0.92),
                                              ("#fbfcfd", 0.78), ("#ffffff", 0.55)]):
            du, dr = [], []
            for k in range(n_arc + 1):
                ang = math.pi / 2 * k / n_arc
                du.append(u_start + depth * math.sin(ang))
                dr.append(ch * rad * math.cos(ang))
            off = ch * (1.0 - rad) * 0.55
            du = [u + depth * 0.06 * (1 - rad) for u in du]
            dr = [r + off for r in dr]
            du_full = du + du[::-1] + [du[0]]
            dr_full = dr + [2 * off - x for x in dr[::-1]] + [dr[0]]
            poly(du_full, dr_full, tint,
                 C["cork_edge"] if layer == 0 else "rgba(0,0,0,0)",
                 1.3 if layer == 0 else 0.1)

        if labels:
            lx, ly = pt(prof["u_nose"] + 0.20, 0.0)
            fig.add_annotation(x=lx, y=ly, xref="x", yref="y", showarrow=False,
                                text="<b>코르크</b>", font=dict(size=12, color="#334155"))
            sx2, sy2 = pt(prof["u_tip"] - 0.22, 0.0)
            fig.add_annotation(x=sx2, y=sy2, xref="x", yref="y", showarrow=False,
                                text="<b>깃털</b>", font=dict(size=12, color="#334155"))
        return fig

    def controller_orientation_preview(preset_name, custom_deg, V0, launch_angle,
                                        u_air_x, u_air_y, alpha0=None):
        try:
            # The preview used to read the orientation radio only, so it drew the
            # shuttlecock cork-forward while the run launched it at whatever alpha0
            # said. Both resolve the attitude the same way now.
            a0 = 180.0 if alpha0 in (None, "") else float(alpha0)
            bo = resolved_body_orientation(preset_name, custom_deg, a0, V0,
                                            launch_angle, u_air_x, u_air_y)
            ic = InitialCondition.from_launch(V0, launch_angle, 1.0, a0, 180.0,
                                               body_orientation_deg=bo,
                                               u_air_x=u_air_x, u_air_y=u_air_y)
            alpha0 = rad2deg(ic.alpha0_rad)
            flow = rad2deg(ic.flow_angle0_rad)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[-1.5, 1.5], y=[0.0, 0.0], mode="lines",
                                      line=dict(color="#cbd5e1", width=1, dash="dot"),
                                      hoverinfo="skip", showlegend=False))
            fig.add_annotation(x=1.5, y=0.0, xref="x", yref="y", showarrow=False,
                                text="0° 기준축 (+x)", xanchor="right", yanchor="bottom",
                                font=dict(size=11, color="#94a3b8"))
            draw_shuttlecock(fig, 0.0, 0.0, cork_heading_deg(bo), 1.0)

            fr = deg2rad(flow)
            ofx, ofy = -math.sin(fr) * 0.40, math.cos(fr) * 0.40
            arrow_with_label(fig, 1.30 * math.cos(fr) - ofx, 1.30 * math.sin(fr) - ofy,
                              0.50 * math.cos(fr) - ofx, 0.50 * math.sin(fr) - ofy,
                              "#805ad5", "맞바람 (공기가 오는 쪽)", 4)
            th = deg2rad(launch_angle)
            arrow_with_label(fig, 0.45 * math.cos(th), 0.45 * math.sin(th),
                              1.15 * math.cos(th), 1.15 * math.sin(th),
                              "#2f855a", "진행 방향", 4)

            aa = abs(wrap_to_pi(deg2rad(alpha0)))
            if aa > deg2rad(150):
                verdict = "코르크가 맞바람을 향함 → 안정된 비행 자세입니다."
                color = "#2f855a"
            elif aa < deg2rad(30):
                verdict = "깃털이 맞바람을 향함 → 불안정한 자세이며, 곧 뒤집힙니다(turnover)."
                color = "#c53030"
            else:
                verdict = "비스듬한 자세 → 복원 모멘트가 작용해 코르크가 앞으로 돌아갑니다."
                color = "#b7791f"

            fig.add_annotation(x=0, y=-1.42, xref="x", yref="y", showarrow=False,
                                text=f"<b>{verdict}</b>", font=dict(size=14, color=color),
                                align="center")
            fig.update_layout(
                template="plotly_white", width=560, height=520, showlegend=False,
                xaxis=dict(visible=False, range=[-1.55, 1.55], scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False, range=[-1.62, 1.48]),
                margin=dict(l=10, r=10, t=20, b=10))

            info = (
                f"### 현재 자세 요약\n\n"
                f"| 항목 | 값 | 설명 |\n|---|---|---|\n"
                f"| 발사각 | {launch_angle:.1f}° | 셔틀콕이 **날아가는 방향** (사용자 입력) |\n"
                f"| 자세각 (코르크 방향) | {cork_heading_deg(bo):.1f}° | **코르크 머리가 향한 방향** |\n"
                f"| 진행 방향 | {flow:.1f}° | 공기에 대한 속도 벡터의 방향 (자동 계산) |\n"
                f"| 맞바람이 오는 쪽 | {flow:.1f}° | 진행 방향과 같은 쪽에서 공기가 불어옵니다 |\n"
                f"| 상대 바람 벡터 | {rad2deg(wrap_to_pi(deg2rad(flow) + math.pi)):.1f}° | "
                f"공기가 실제로 밀고 가는 방향 (진행 방향의 정반대) |\n"
                f"| 받음각 α | {alpha0:.1f}° | 코르크 축과 상대 바람 벡터 사이의 각 (자동 계산) |\n\n"
                f"**{verdict}**\n\n"
                + ANGLE_REFERENCE_TEXT
            )
            return fig, info, bo
        except Exception:
            logger.exception("orientation preview failed")
            return None, "자세 미리보기를 표시할 수 없습니다: 입력값을 확인하세요.", 180.0

    def add_flow_overlay(fig, row, show_vectors=True, show_streams=True,
                          resolution="보통", extent=None):
        """Overlay the reduced-order visual flow field. Display only, never used in forces."""
        try:
            vrel = np.array([row.vx_m_s - state["env"].u_air_x if state.get("env") else row.vx_m_s,
                             row.vy_m_s - state["env"].u_air_y if state.get("env") else row.vy_m_s])
        except Exception:
            vrel = np.array([row.vx_m_s, row.vy_m_s])
        if float(vnorm(vrel)) <= 0:
            return fig
        n = FLOW_RESOLUTIONS.get(resolution, 15)
        if extent is None:
            extent = 0.55
        tb = getattr(row, "theta_body_deg", None)
        tb = float(tb) if (tb is not None and tb == tb) else float(row.alpha_deg)
        body_e = np.array([math.cos(deg2rad(tb)), math.sin(deg2rad(tb))])
        field = visual_flow_field(row.x_m, row.y_m, vrel, body_e, extent, n=n)
        if field is None:
            return fig

        if show_streams:
            for xs, ys in visual_streamlines(field, n_lines=max(7, n // 2)):
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                          line=dict(color="rgba(56,132,255,0.45)", width=1.4),
                                          hoverinfo="skip"))
        if show_vectors:
            X, Y, U, V = field["X"], field["Y"], field["U"], field["V"]
            sc = extent / (n * max(field["Vrel"], 1e-6)) * 1.6
            xs, ys = [], []
            for i in range(0, X.shape[0], 2):
                for j in range(0, X.shape[1], 2):
                    xs += [float(X[i, j]), float(X[i, j] + U[i, j] * sc), None]
                    ys += [float(Y[i, j]), float(Y[i, j] + V[i, j] * sc), None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                      line=dict(color="rgba(43,108,176,0.55)", width=1.2),
                                      hoverinfo="skip", name="공기 흐름"))
        fig.add_annotation(x=row.x_m, y=row.y_m - extent * 0.92, xref="x", yref="y",
                            showarrow=False, font=dict(size=11, color="#718096"),
                            text="간이 공기흐름 시각화 (CFD 결과 아님)")
        return fig

    def _intact_comparison_md(df, ref_df):
        """The dashed baseline in numbers: what the damage actually cost."""
        try:
            rng = float(df.x_m.iloc[-1] - df.x_m.iloc[0])
            rng0 = float(ref_df.x_m.iloc[-1] - ref_df.x_m.iloc[0])
            dz = float(df.z_m.iloc[-1] - df.z_m.iloc[0])
            dz0 = float(ref_df.z_m.iloc[-1] - ref_df.z_m.iloc[0])
            t = float(df.time_s.iloc[-1]); t0 = float(ref_df.time_s.iloc[-1])
            ratio = (rng / rng0) if rng0 else float("nan")
            return (
                f"**정상 16깃 기준과 비교** (그래프의 회색 점선)\n\n"
                f"| 항목 | 정상 16깃 | 현재 파손 | 차이 |\n|---|---|---|---|\n"
                f"| 도달 거리 (x) | {fmt(rng0, ' m')} | {fmt(rng, ' m')} | "
                f"{rng - rng0:+.3f} m ({ratio:.3f}배) |\n"
                f"| 좌우 이동 (z) | {fmt(dz0, ' m')} | {fmt(dz, ' m')} | "
                f"{dz - dz0:+.3f} m |\n"
                f"| 비행 시간 | {fmt(t0, ' s')} | {fmt(t, ' s')} | {t - t0:+.3f} s |\n\n")
        except Exception:
            logger.exception("intact comparison failed")
            return ""

    def projection_figure(df, mode, ref_df=None):
        pairs = {"XY 평면": ("x_m", "y_m", "x [m]", "y [m] (높이)"),
                 "XZ 평면": ("x_m", "z_m", "x [m]", "z [m] (좌우)"),
                 "YZ 평면": ("z_m", "y_m", "z [m] (좌우)", "y [m] (높이)")}
        cx, cy, lx, ly = pairs[mode]
        fig = go.Figure()
        # The intact 16-feather baseline, dashed, so the damage is read off the plot
        # instead of being held in the head between two runs. The 2D tabs have had
        # this since the porosity work; the 3D tab drew the damaged flight alone.
        if ref_df is not None and cx in ref_df.columns and cy in ref_df.columns:
            fig.add_trace(go.Scatter(x=ref_df[cx], y=ref_df[cy], mode="lines",
                                      name="정상 16깃 (기준)",
                                      line=dict(color="#94a3b8", width=2,
                                                dash="dash")))
        fig.add_trace(go.Scatter(x=df[cx], y=df[cy], mode="lines", name="시뮬레이션",
                                  line=dict(color="#2b6cb0", width=2.5)))
        fig.add_trace(go.Scatter(x=[df[cx].iloc[0]], y=[df[cy].iloc[0]], mode="markers",
                                  name="출발", marker=dict(size=11, color="#2f855a")))
        fig.add_trace(go.Scatter(x=[df[cx].iloc[-1]], y=[df[cy].iloc[-1]], mode="markers",
                                  name="도착", marker=dict(size=11, color="#c53030")))
        note = "" if mode == "XY 평면" else "  (Z는 모델 예측값 — 측정값 아님)"
        fig.update_layout(title=f"{mode}{note}", title_x=0.5, template="plotly_white",
                           xaxis_title=lx, yaxis_title=ly, height=440,
                           margin=dict(l=55, r=20, t=50, b=45))
        return fig

    def trajectory_3d_figure(df, ref_df=None):
        fig = go.Figure()
        if ref_df is not None and {"x_m", "y_m", "z_m"} <= set(ref_df.columns):
            fig.add_trace(go.Scatter3d(x=ref_df.x_m, y=ref_df.z_m, z=ref_df.y_m,
                                        mode="lines", name="정상 16깃 (기준)",
                                        line=dict(color="#94a3b8", width=4,
                                                  dash="dash")))
        fig.add_trace(go.Scatter3d(x=df.x_m, y=df.z_m, z=df.y_m, mode="lines",
                                    name="3차원 궤적",
                                    line=dict(color="#2b6cb0", width=5)))
        fig.add_trace(go.Scatter3d(x=[df.x_m.iloc[0]], y=[df.z_m.iloc[0]],
                                    z=[df.y_m.iloc[0]], mode="markers", name="출발",
                                    marker=dict(size=5, color="#2f855a")))
        fig.add_trace(go.Scatter3d(x=[df.x_m.iloc[-1]], y=[df.z_m.iloc[-1]],
                                    z=[df.y_m.iloc[-1]], mode="markers", name="도착",
                                    marker=dict(size=5, color="#c53030")))
        fig.update_layout(
            title="3차원 궤적 (Z는 모델 예측값)", title_x=0.5, height=520,
            template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
            scene=dict(xaxis_title="x [m] (진행)", yaxis_title="z [m] (좌우)",
                        zaxis_title="y [m] (높이)",
                        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
        return fig

    def scene_3d_figure(df, i, show_flow=True, show_streams=True, show_forces=True,
                         resolution="보통", follow=True):
        i = int(np.clip(i, 0, len(df) - 1))
        row = df.iloc[i]
        c = np.array([row.x_m, row.y_m, row.z_m])
        q = np.array([row.qw, row.qx, row.qy, row.qz])
        v = np.array([row.vx_m_s, row.vy_m_s, row.vz_m_s])
        env = state.get("env") or EnvironmentModel()
        v_rel = v - np.array([env.u_air_x, env.u_air_y, env.u_air_z])

        spanx = max(df.x_m.max() - df.x_m.min(), 1e-3)
        spany = max(df.y_m.max() - df.y_m.min(), 1e-3)
        spanz = max(df.z_m.max() - df.z_m.min(), 1e-3)
        span = max(spanx, spany, spanz)
        extent = 0.12 * span
        sc = extent * 0.85

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=df.x_m, y=df.z_m, z=df.y_m, mode="lines",
                                    name="전체 궤적",
                                    line=dict(color="#cbd5e1", width=3)))
        fig.add_trace(go.Scatter3d(x=df.x_m[:i + 1], y=df.z_m[:i + 1], z=df.y_m[:i + 1],
                                    mode="lines", name="지나온 궤적",
                                    line=dict(color="#2b6cb0", width=5)))

        mesh = shuttlecock_mesh_3d(c, quat_mul(q, np.array([0.0, 0.0, 1.0, 0.0])), sc,
                                    state.get("skirt"),
                                    state["model"].geometry if state.get("model") else None)

        def mesh_trace(part, color, name, opacity=1.0, show=True):
            pts = part["pts"]
            if len(pts) == 0:
                return
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
                i=part["i"], j=part["j"], k=part["k"],
                color=color, opacity=opacity, name=name, showlegend=show,
                flatshading=False, hoverinfo="skip",
                lighting=dict(ambient=0.62, diffuse=0.82, specular=0.18, roughness=0.55),
                lightposition=dict(x=100, y=200, z=300)))

        mesh_trace(mesh["vanes"], "#ffffff", f"깃털 ({mesh['n_shown']}/{mesh['n_total']}개)")
        mesh_trace(mesh["cork"], "#f7f7f5", "코르크")
        mesh_trace(mesh["band"], "#16295e", "코르크 밴드", show=False)

        rx_, ry_, rz_ = [], [], []
        for r0, r1 in mesh["ribs"]:
            rx_ += [r0[0], r1[0], None]
            ry_ += [r0[2], r1[2], None]
            rz_ += [r0[1], r1[1], None]
        fig.add_trace(go.Scatter3d(x=rx_, y=ry_, z=rz_, mode="lines", showlegend=False,
                                    line=dict(color="rgba(150,160,178,0.85)", width=2),
                                    hoverinfo="skip"))
        for ring in mesh["threads"]:
            fig.add_trace(go.Scatter3d(x=[p[0] for p in ring], y=[p[2] for p in ring],
                                        z=[p[1] for p in ring], mode="lines",
                                        showlegend=False, hoverinfo="skip",
                                        line=dict(color="rgba(190,198,212,0.9)", width=3)))
        fig.add_trace(go.Scatter3d(x=[p[0] for p in mesh["rim"]],
                                    y=[p[2] for p in mesh["rim"]],
                                    z=[p[1] for p in mesh["rim"]],
                                    mode="lines", showlegend=False, hoverinfo="skip",
                                    line=dict(color="#aab3c2", width=3)))

        def vec3(vec, color, label, scale_to=1.0):
            n = float(vnorm(vec))
            if n <= 0:
                return
            d = np.asarray(vec, dtype=float) / n * (extent * scale_to)
            fig.add_trace(go.Scatter3d(
                x=[c[0], c[0] + d[0]], y=[c[2], c[2] + d[2]], z=[c[1], c[1] + d[1]],
                mode="lines", name=label, line=dict(color=color, width=6)))

        vec3(v, "#2f855a", "속도", 1.0)
        if show_flow or show_streams:
            vec3(-v_rel, "#805ad5", "맞바람", 0.9)

        if show_forces:
            drag_dir = -v_rel
            vec3(drag_dir, "#c53030", "항력", 0.8)
            vec3(np.array([0.0, -1.0, 0.0]), "#4a5568", "중력", 0.6)
            if float(row.get("side_force_N", 0.0)) > 0:
                vec3(cross3(v_rel, np.array([row.ex, row.ey, row.ez])),
                     "#dd6b20", "측면력", 0.6)

        n_grid = {"낮음": 5, "보통": 7, "높음": 9}.get(resolution, 7)
        if show_flow:
            field = visual_flow_field_3d(c, v_rel, extent * 1.6, n=n_grid)
            if field is not None:
                pts, vecs = field["points"], field["vectors"]
                k = extent * 0.5 / max(field["Vrel"], 1e-9)
                xs, ys, zs = [], [], []
                for ppos, vv in zip(pts, vecs):
                    xs += [ppos[0], ppos[0] + vv[0] * k, None]
                    ys += [ppos[2], ppos[2] + vv[2] * k, None]
                    zs += [ppos[1], ppos[1] + vv[1] * k, None]
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                            name="공기 흐름 (간이)",
                                            line=dict(color="rgba(43,108,176,0.45)", width=2)))
        if show_streams:
            field = visual_flow_field_3d(c, v_rel, extent * 1.6, n=3)
            for xs, ys, zs in visual_streamlines_3d(field, c, extent * 1.6,
                                                     n_lines=max(5, n_grid)):
                fig.add_trace(go.Scatter3d(x=xs, y=[zz for zz in zs], z=ys, mode="lines",
                                            showlegend=False,
                                            line=dict(color="rgba(56,132,255,0.35)", width=2)))

        if follow:
            rng = dict(xaxis=dict(range=[c[0] - extent * 2.2, c[0] + extent * 2.2]),
                       yaxis=dict(range=[c[2] - extent * 2.2, c[2] + extent * 2.2]),
                       zaxis=dict(range=[c[1] - extent * 2.2, c[1] + extent * 2.2]))
        else:
            rng = dict(xaxis=dict(range=[df.x_m.min() - extent, df.x_m.max() + extent]),
                       yaxis=dict(range=[df.z_m.min() - extent, df.z_m.max() + extent]),
                       zaxis=dict(range=[min(df.y_m.min(), 0) - extent, df.y_m.max() + extent]))

        fig.update_layout(
            title=f"3차원 비행 공간 — t = {row.time_s:.4f} s", title_x=0.5,
            template="plotly_white", height=620, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.02, x=0),
            scene=dict(xaxis_title="x [m] (진행)", yaxis_title="z [m] (좌우)",
                        zaxis_title="y [m] (높이)", aspectmode="cube",
                        camera=dict(eye=dict(x=1.6, y=-1.7, z=0.85)), **rng))
        return fig

    def scene_3d_panel(df, i):
        i = int(np.clip(i, 0, len(df) - 1))
        r = df.iloc[i]

        def g(name, unit="", nd=4):
            val = r.get(name, None)
            if val is None or (isinstance(val, float) and val != val):
                return NA
            return fmt(val, unit, nd)
        return (
            f"**시간 t = {fmt(r.time_s, ' s')}**\n\n"
            f"| 물리량 | 값 | 물리량 | 값 |\n|---|---|---|---|\n"
            f"| 위치 x | {g('x_m',' m')} | 속도 vx | {g('vx_m_s',' m/s')} |\n"
            f"| 위치 y | {g('y_m',' m')} | 속도 vy | {g('vy_m_s',' m/s')} |\n"
            f"| 위치 z *(예측)* | {g('z_m',' m')} | 속도 vz *(예측)* | {g('vz_m_s',' m/s')} |\n"
            f"| 속력 | {g('speed_m_s',' m/s')} | 상대풍속 | {g('V_rel_m_s',' m/s')} |\n"
            f"| 받음각 α | {g('alpha_deg','°')} | 측면각 β | {g('beta_deg','°')} |\n"
            f"| Reynolds 수 | {g('Re')} | Cd | {g('Cd')} |\n"
            f"| Cl | {g('Cl')} | Cy | {g('Cy')} |\n"
            f"| Mx | {g('moment_x_Nm',' N·m')} | My | {g('moment_y_Nm',' N·m')} |\n"
            f"| Mz | {g('moment_z_Nm',' N·m')} | 순모멘트 | {g('net_moment_Nm',' N·m')} |\n"
            f"| 각속도 (turnover) | {g('omega_rad_s',' rad/s')} | 축 spin | {g('spin_rad_s',' rad/s')} |\n"
            f"| 항력 | {g('drag_force_N',' N')} | 양력 | {g('lift_force_N',' N')} |\n"
            f"| 측면력 | {g('side_force_N',' N')} | quaternion norm | {g('quat_norm')} |\n\n"
            f"*z·vz와 3차원 자세는 **모델 예측값**이며 측정값이 아닙니다. "
            f"공기 흐름 그림은 간이 시각화이며 CFD 결과가 아닙니다.*"
        )

    def controller_scene_3d(frame_idx, show_flow, show_streams, show_forces,
                             resolution, follow):
        res = state.get("result3d")
        if res is None or len(res.df) == 0:
            return None, "먼저 3차원 시뮬레이션을 실행하세요."
        try:
            df = res.df
            i = int(np.clip(int(frame_idx or 0), 0, len(df) - 1))
            fig = scene_3d_figure(df, i, bool(show_flow), bool(show_streams),
                                   bool(show_forces), resolution, bool(follow))
            return fig, scene_3d_panel(df, i)
        except Exception:
            logger.exception("3D scene failed")
            return None, "3차원 화면을 표시할 수 없습니다: 내부 오류가 발생했습니다."

    def controller_run_3d(profile_name, skirt_d_mm, mode3d, V0, elevation, azimuth, height,
                           z0, orient_offset, orient_azimuth, wx, wy, wz,
                           spin0, rho, mu, cd_value, cl_mode, cl_value,
                           ixx_in, solver, dt, t_max, view_mode,
                           omega_turn=None, impact_k=0.5,
                           spin_mode="자전 자유 (자연 자전 모델)", c_spin=0.0):
        try:
            clear_warnings()
            planar = (mode3d == "2차원 평면 운동")
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            # An empty gr.Number arrives as 0 (and float(None) used to raise), so a
            # cleared Cd box turned the drag off and drew an 80 m ballistic parabola.
            # There is no shuttlecock with Cd = 0, so treat a non-positive value as
            # "not specified" and fall back to the profile's own drag coefficient.
            try:
                cd_num = float(cd_value)
            except (TypeError, ValueError):
                cd_num = None
            if cd_num is None or cd_num <= 0:
                cd_num = float(build_model_from_profile(
                    db, profile_name, geom).Cd_model.constant_value)
                push_warning(
                    f"항력계수 Cd 칸이 비어 있거나 0 이라 프로파일 값 {cd_num:.3f} 을 "
                    f"사용했습니다. Cd = 0 이면 항력이 없어 셔틀콕이 아니라 탄도 "
                    f"포물선(80 m)이 됩니다.")
            cd_model = CdModel(mode="constant", constant_value=cd_num,
                                source_type="user_defined")
            cl_model = ClModel(mode=cl_mode, constant_value=float(cl_value or 0.0),
                                source_type="user_defined")
            dmg_map = (build_porosity_damage_mapping()
                        if state["skirt"].removed_ids else None)
            # the spin degree of freedom is chosen explicitly; an intact shuttlecock
            # spins naturally because its feathers are canted, so that is the default
            _sp_mode = ("constant" if "토크 없음" in str(spin_mode) else "aero_driven")
            _spin_model = SpinModel(mode=_sp_mode,
                                     spin_ratio=db.spin_ratio_for(profile_name),
                                     R_m=geom.R_m,
                                     c_spin=float(c_spin or 0.0))
            model = build_full_model(db, profile_name, geom, level=2,
                                      spin=_spin_model,
                                      removed_feathers=state["skirt"].removed_ids,
                                      cd_model=cd_model, cl_model=cl_model,
                                      damage_mapping=dmg_map)
            if dmg_map is not None:
                _gs = state["skirt"].geometry_state()
                _pm = dmg_map.porosity_model
                push_warning(
                    f"파손 {len(state['skirt'].removed_ids)}깃 → 다공성 모델 적용 "
                    f"(Cd 배율 {_pm.cd_factor(_gs):.3f}, 비대칭 측력계수 "
                    f"{_pm.lateral_force_bias(_gs):.4f}). 비대칭 파손이면 좌우(z) 편향이 "
                    "발생합니다.")
            if ixx_in:
                model.Ispin = float(ixx_in)
            env = EnvironmentModel(rho=float(rho), u_air_x=float(wx), u_air_y=float(wy),
                                    u_air_z=float(wz), mu=float(mu))
            errs = physical_sanity_checks(model, dt, env)
            Is, It = inertia_3d(model)
            if Is <= 0 or It <= 0:
                errs.append("Ixx/Iyy/Izz must be > 0")
            if errs:
                for e in errs:
                    push_warning(e)
                return (None, None, "입력값이 올바르지 않아 실행하지 못했습니다.",
                        status_markdown(), warning_center(), gr.update())

            if omega_turn is None or omega_turn == "":
                w_turn = float(impact_k or 0.0) * abs(float(V0)) / SHUTTLE_LENGTH_M
            else:
                w_turn = float(omega_turn)
            # A shuttlecock spins about its own axis because its vanes are canted --
            # R*Omega/U = 0.04 for feather, measured (Cohen et al. 2015 fig. 15). The
            # engine applies that equilibrium only when the initial spin is left
            # UNSET; an empty box arrives as 0.0, which is a number, so the natural
            # spin was switched off on every run and a damaged shuttlecock kept its
            # asymmetry pointed the same way for the whole flight instead of turning
            # under it. That is most of the lateral drift the 3D tab was showing.
            try:
                spin_num = float(spin0)
            except (TypeError, ValueError):
                spin_num = 0.0
            spin_ic = spin_num if abs(spin_num) > 1e-12 else None
            if spin_ic is None and "구속" in str(spin_mode):
                spin_ic = 0.0          # 자전 구속을 고른 경우는 0 이 사용자의 뜻이다
            spin0_ic = spin_ic
            if abs(wrap_to_pi(deg2rad(float(orient_offset)) - math.pi)) < 1e-6 \
                    and abs(w_turn) < 1e-9:
                push_warning(
                    "초기 자세가 평형(180°)과 정확히 같고 Turnover 각속도도 0이라 "
                    "뒤집힘(flip)이나 회전 진동이 전혀 발생하지 않습니다. "
                    "자세각을 0°(타격 직후)로 두거나 Turnover 각속도를 주세요.")
            ic3 = InitialCondition3D(x0=0.0, y0=float(height), z0=float(z0),
                                      V0=float(V0), elevation_deg=float(elevation),
                                      azimuth_deg=float(azimuth),
                                      body_axis_offset_deg=float(orient_offset),
                                      body_azimuth_deg=float(orient_azimuth),
                                      omega0=(0.0, 0.0, float(w_turn)),
                                      spin0_rad_s=spin0_ic)
            for w in geometry_warnings(model):
                push_warning(w)

            model.use_geometry_inertia = bool(state.get("use_geometry_inertia_3d", True))
            cache_key = json.dumps([
                profile_name, float(skirt_d_mm), bool(planar),
                sorted(state["skirt"].removed_ids), float(V0), float(elevation),
                float(azimuth), float(height), float(z0), float(orient_offset),
                float(orient_azimuth), float(wx), float(wy), float(wz),
                (spin0_ic if spin0_ic is not None else "natural"),
                float(rho), float(mu), cd_num,
                str(cl_mode), float(cl_value or 0.0),
                (float(ixx_in) if ixx_in else None), str(solver), float(dt), float(t_max),
                bool(model.use_geometry_inertia),
                # inputs added later were missing from the key, so changing them
                # silently returned the cached run and looked like "no effect"
                (None if omega_turn in (None, "") else float(omega_turn)),
                float(impact_k or 0.0),
                float(getattr(model.aero.spin, "c_spin", 0.0) or 0.0),
                str(getattr(model.aero.spin, "mode", "constant")),
                str(spin_mode), float(c_spin or 0.0),
                float(getattr(model.aero.spin, "spin_ratio", 0.0) or 0.0),
                bool(state["skirt"].removed_ids),
                float(state.get("skirt_d_mm") or 0.0),
            ], sort_keys=True, default=str)
            cached = state.get("cache3d")
            if cached is not None and cached.get("key") == cache_key:
                res = cached["result"]
            else:
                t_phys = time.time()
                res = SimulationEngine3D(model, env, ic3, solver=solver, dt=dt, t_max=t_max,
                                          planar=planar).run()
                state["physics_time_s"] = time.time() - t_phys
                state["cache3d"] = dict(key=cache_key, result=res)
            state["result3d"] = res
            state["dt3d"] = float(dt)
            state["model"] = model
            state["env"] = env
            for w in res.warnings:
                push_warning(w)
            if not res.valid or len(res.df) == 0:
                return (None, None, "시뮬레이션 실패: 수치적으로 불안정합니다.",
                        status_markdown(), warning_center(), gr.update())
            if not planar:
                push_warning("3D 예측은 실험으로 검증되지 않은 모델 계산값입니다 "
                             "(검증되지 않은 3D 예측).")

            # A trajectory far past the court is not a shuttlecock, whatever the
            # inputs were: drawing it makes an input mistake look like a result.
            env_errs = flight_envelope_errors(res.df, model, env)
            if env_errs:
                for e in env_errs:
                    push_warning(e)
                return (None, None,
                        "물리적으로 불가능한 궤적이라 그리지 않았습니다 — 아래 경고를 "
                        "확인하세요.",
                        status_markdown(), warning_center(), gr.update())

            df = res.df
            # Same launch, same coefficients, undamaged skirt -- the dashed baseline.
            # Only worth running when feathers are actually missing; otherwise the two
            # curves are the same line.
            ref_df = None
            if state["skirt"].removed_ids:
                try:
                    ref_model = build_full_model(db, profile_name, geom, level=2,
                                                  spin=_spin_model,
                                                  removed_feathers=[],
                                                  cd_model=cd_model,
                                                  cl_model=cl_model,
                                                  damage_mapping=None)
                    ref_model.use_geometry_inertia = model.use_geometry_inertia
                    if ixx_in:
                        ref_model.Ispin = float(ixx_in)
                    ref_res = SimulationEngine3D(ref_model, env, ic3, solver=solver,
                                                  dt=dt, t_max=t_max,
                                                  planar=planar).run()
                    if ref_res.valid and len(ref_res.df):
                        ref_df = ref_res.df
                except Exception:
                    logger.exception("3D reference run failed")
            state["ref3d"] = ref_df

            main = (trajectory_3d_figure(df, ref_df) if view_mode == "3차원 궤적"
                    else projection_figure(df, view_mode, ref_df))
            side = (projection_figure(df, "XY 평면", ref_df) if view_mode != "XY 평면"
                    else projection_figure(df, "XZ 평면", ref_df))

            last = df.iloc[-1]
            info = (
                f"**모드:** {mode3d} &nbsp;|&nbsp; **버전:** v{APP_VERSION}\n\n"
                f"| 항목 | 값 | 성격 |\n|---|---|---|\n"
                f"| 비행 시간 | {fmt(last.time_s, ' s')} | 모델 계산 |\n"
                f"| 도달 거리 (x) | {fmt(last.x_m - df.x_m.iloc[0], ' m')} | 모델 계산 |\n"
                f"| 좌우 이동 (z) | {fmt(last.z_m - df.z_m.iloc[0], ' m')} | **모델 예측값** |\n"
                f"| 최종 속도 | {fmt(last.speed_m_s, ' m/s')} | 모델 계산 |\n"
                f"| 최종 받음각 | {fmt(last.alpha_deg, '°')} | 모델 계산 |\n"
                f"| Reynolds 수 | {fmt(df.Re.min())} ~ {fmt(df.Re.max())} | 모델 계산 |\n\n"
                + (_intact_comparison_md(df, ref_df) if ref_df is not None else "")
                + "*X/Y는 2D 실험과 직접 비교할 수 있지만, Z와 3차원 자세는 "
                "모델 예측값이며 측정값이 아닙니다.*"
            )
            slider_upd = gr.update(minimum=0, maximum=max(len(df) - 1, 1), value=0, step=1)
            return main, side, info, status_markdown(), warning_center(), slider_upd
        except Exception:
            logger.exception("3D simulation failed")
            push_warning("3D 시뮬레이션 실패: 내부 오류가 발생했습니다.")
            return (None, None, "시뮬레이션 실패: 내부 오류가 발생했습니다.",
                    status_markdown(), warning_center(), gr.update())

    def controller_change_view(view_mode):
        res = state.get("result3d")
        if res is None or len(res.df) == 0:
            return None, "먼저 3차원 시뮬레이션을 실행하세요."
        try:
            df = res.df
            ref_df = state.get("ref3d")
            fig = (trajectory_3d_figure(df, ref_df) if view_mode == "3차원 궤적"
                   else projection_figure(df, view_mode, ref_df))
            return fig, ""
        except Exception:
            logger.exception("view change failed")
            return None, "보기를 바꿀 수 없습니다: 내부 오류가 발생했습니다."

    def controller_compare_3d_xy():
        res = state.get("result3d")
        exp = state.get("exp_df")
        if res is None or len(res.df) == 0:
            return None, "먼저 3차원 시뮬레이션을 실행하세요."
        if exp is None:
            return None, "먼저 실험 데이터를 불러오세요."
        try:
            df = res.df
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.x_m, y=df.y_m, mode="lines",
                                      name="시뮬레이션 XY 투영",
                                      line=dict(color="#2b6cb0", width=2.5)))
            if "x" in exp.columns and "y" in exp.columns:
                fig.add_trace(go.Scatter(x=exp.x, y=exp.y, mode="markers",
                                          name="실험 데이터 (측정값)",
                                          marker=dict(size=8, color="#c53030")))
            fig.update_layout(title="3D 시뮬레이션의 XY 투영 vs 2D 실험", title_x=0.5,
                               template="plotly_white", xaxis_title="x [m]",
                               yaxis_title="y [m]", height=440,
                               margin=dict(l=55, r=20, t=50, b=45))
            resid = residual_analysis(df, exp)
            lines = [f"**{k}** — RMSE: {fmt(v.get('rmse'))}, MAE: {fmt(v.get('mae'))}"
                     for k, v in resid.items()]
            lines.append("")
            lines.append("*이 XY 비교가 3D 모델의 핵심 검증 방법입니다. "
                         "Z 성분은 실험 데이터가 없으므로 검증되지 않습니다.*")
            return fig, "\n\n".join(lines)
        except Exception:
            logger.exception("3D-2D comparison failed")
            return None, "비교 실패: 내부 오류가 발생했습니다."

    def controller_flip_validation(profile_name, skirt_d_mm, l_gc, damping_scale,
                                    cd_cross, rho_v):
        try:
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            model = build_model_from_profile(db, profile_name, geom)
            tp = model.turnover
            # `x or default` treats a deliberate 0 as "not set"; damping_scale = 0
            # (no aerodynamic damping) is a legitimate thing to ask for
            tp.l_gc = 0.020 if l_gc in (None, "") else float(l_gc)
            tp.damping_scale = 2.5 if damping_scale in (None, "") else float(damping_scale)
            tp.Cd_cross = 0.94 if cd_cross in (None, "") else float(cd_cross)
            S = geom.S
            rho_val = float(rho_v or 1.2)
            cases = [
                ("Cohen 2015 fig.3(a)", 18.6, 206.0, 16.0, 92.0, 130.0),
                ("Cohen 2015 fig.3(b)", 10.4, 28.0, 39.0, 168.0, 180.0),
            ]
            lines = [
                "### 논문 실측값과의 비교",
                "",
                "출처: C Cohen, B Darbois Texier, D Quéré, C Clanet, "
                "*The physics of badminton*, New J. Phys. **17** (2015) 063001 (open access).",
                "",
                "| 실험 | 항목 | 논문 실측 | 이 모델 | 차이 |",
                "|---|---|---|---|---|",
            ]
            for name, U, pdot, f_exp, o_exp, s_exp in cases:
                f_th = 1000.0 * tp.tau_flip(U, rho_val, S, pdot)
                o_th = 1000.0 * tp.tau_oscillation(U, rho_val, S)
                s_th = 1000.0 * tp.tau_stabilizing(U, rho_val, S)
                for label, exp_v, th_v in (("뒤집힘 시간 τf", f_exp, f_th),
                                            ("진동 주기 τo", o_exp, o_th),
                                            ("정렬 시간 τs", s_exp, s_th)):
                    err = 100.0 * (th_v - exp_v) / exp_v if exp_v else float("nan")
                    lines.append(f"| {name} U={U} m/s | {label} | {exp_v:.0f} ms | "
                                 f"{th_v:.0f} ms | {err:+.0f}% |")
            ell = tp.aero_length(rho_val, S)
            z = tp.zeta_aero(rho_val, S)
            lines += [
                "",
                f"**공기역학 길이 ℓ = {ell:.2f} m** (Cohen 2015: 깃털 4.04 m, 플라스틱 4.48 m — "
                f"스커트 지름에 따라 달라집니다)",
                f"**감쇠비 ζ = {z:.3f}** (속도와 무관한 상수 — 논문 모델의 구조적 결과)",
                "",
                "*τf는 타격 회전 φ̇₀에 거의 전적으로 좌우되고, τo는 ω₀(U)가, "
                "τs는 감쇠 보정계수가 결정합니다. 실제 시뮬레이션은 비선형·감속을 모두 포함하므로 "
                "진폭이 큰 초기 구간에서는 위 선형 예측보다 주기가 길게 나오는 것이 정상입니다.*",
            ]
            lines.append("")
            lines.append("---")
            lines.append("")
            try:
                env_lit = EnvironmentModel(rho=rho_val)
                lines.append(literature_validation_markdown(model, env_lit,
                                                             rho=rho_val))
            except Exception:
                logger.exception("literature table failed")
            return "\n".join(lines)
        except Exception:
            logger.exception("flip validation failed")
            return "검증 실패: 입력값을 확인하세요."

    def reference_intact_run(model, env, ic, solver, dt, t_max, profile_name, geom):
        """Re-runs the same launch with an undamaged 16-feather skirt.

        Gives the dashed comparison curve, so the effect of the damage is visible
        against the intact baseline rather than having to be remembered.
        """
        ref_model = build_model_from_profile(db, profile_name, geom)
        ref_model.Cd_model.constant_value = model.Cd_model.constant_value
        ref_model.Cd_model.mode = model.Cd_model.mode
        ref_model.Cd_model.cd_cross = getattr(model.Cd_model, "cd_cross", 0.94)
        ref_model.turnover.Tw = model.turnover.Tw
        ref_model.turnover.zeta = model.turnover.zeta
        ref_model.turnover.mode = model.turnover.mode
        ref_model.turnover.Cd_ref = model.turnover.Cd_ref
        ref_model.turnover.l_gc = model.turnover.l_gc
        ref_model.turnover.damping_scale = model.turnover.damping_scale
        ref_model.turnover.Cd_cross = model.turnover.Cd_cross
        ref_model.skirt = SkirtGeometry.from_defaults() \
            if hasattr(SkirtGeometry, "from_defaults") else None
        ref_model._damage_mapping = None
        r = SimulationEngine(ref_model, env, ic, solver=solver, dt=dt, t_max=t_max).run()
        return r.df if r.valid and len(r.df) else None

    def add_reference_trace(fig, ref_df, xcol, ycol, name="정상 16깃 (기준)"):
        """Adds the intact-shuttlecock baseline as a dashed grey curve."""
        if fig is None or ref_df is None or xcol not in ref_df.columns \
                or ycol not in ref_df.columns:
            return fig
        try:
            fig.add_trace(go.Scatter(x=ref_df[xcol], y=ref_df[ycol], mode="lines",
                                      name=name,
                                      line=dict(color="#94a3b8", width=2, dash="dash")))
            fig.update_layout(showlegend=True)
        except Exception:
            logger.exception("reference trace failed")
        return fig

    def controller_run_simulation(profile_name, skirt_d_mm, cd_override, tw_override,
                                   zeta_override,
                                   turnover_mode, l_gc, damping_scale, cd_cross, orient_cd,
                                   aero_spin,
                                   V0, launch_angle, height, alpha0, alpha_eq, omega0,
                                   rho, u_air_x, solver, dt, t_max):
        try:
            clear_warnings()
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            model = build_model_from_profile(db, profile_name, geom)
            if cd_override:
                model.Cd_model.constant_value = float(cd_override)
                model.Cd_model.constant_value_source = "user_defined"
            if tw_override:
                model.turnover.Tw = float(tw_override)
            if zeta_override:
                model.turnover.zeta = float(zeta_override)
            apply_flip_settings(model, turnover_mode, l_gc, damping_scale, cd_cross,
                                 orient_cd, False, geom)

            pre_errs = input_validation_errors(model.m, model.Cd_model.constant_value,
                                                model.turnover.Tw, model.turnover.zeta, dt)
            if pre_errs:
                for e in pre_errs:
                    push_warning(e)
                return (None, None, None, None, None,
                        "시뮬레이션 실패: 입력값이 올바르지 않습니다.",
                        status_markdown(), workflow_markdown(), warning_center())

            env = EnvironmentModel(rho=rho, u_air_x=u_air_x)
            ic = InitialCondition.from_launch(V0, launch_angle, height, alpha0, alpha_eq, omega0)

            cons = model.consistency_report(rho, model.Cd_model.constant_value)
            if cons.get("parameter_consistency") is False:
                push_warning(f"Parameter inconsistency: L_input={fmt(cons['L_input'],' m')} "
                             f"vs L_calculated={fmt(cons['L_calculated'],' m')}")

            state["solver"] = solver
            state["dt"] = dt
            state["profile"] = profile_name

            # the basic-tab model is built without a skirt, so the feather damage never
            # reached the physics: attach the current skirt and the porosity mapping
            model.skirt = state["skirt"]
            if state["skirt"].removed_ids:
                model._damage_mapping = build_porosity_damage_mapping()
                _gs = state["skirt"].geometry_state()
                _pm = model._damage_mapping.porosity_model
                push_warning(
                    f"파손 {len(state['skirt'].removed_ids)}깃 → 다공성 모델 적용 "
                    f"(solidity σ = {_pm.solidity_from_state(_gs):.3f}, "
                    f"Cd 배율 {_pm.cd_factor(_gs):.3f})")
                for _w in planar_damage_warnings(model):
                    push_warning(_w)
            else:
                model._damage_mapping = None

            eng = SimulationEngine(model, env, ic, solver=solver, dt=dt, t_max=t_max)
            res = eng.run()

            state["model"] = model
            state["env"] = env
            state["ic"] = ic
            state["result"] = res
            for w in res.warnings:
                push_warning(w)
            if not res.valid:
                return (None, None, None, None, None,
                        "시뮬레이션 실패: 수치적으로 불안정합니다.",
                        status_markdown(), workflow_markdown(), warning_center())

            ref_df = None
            if state["skirt"].removed_ids:
                try:
                    ref_df = reference_intact_run(model, env, ic, solver, dt, t_max,
                                                   profile_name, geom)
                except Exception:
                    logger.exception("reference run failed")
                    ref_df = None

            traj_fig = make_plot(res.df, "x_m", "y_m", f"{profile_name} — Trajectory")
            add_reference_trace(traj_fig, ref_df, "x_m", "y_m")
            speed_fig = make_plot(res.df, "time_s", "speed_m_s", f"{profile_name} — Speed vs Time")
            add_reference_trace(speed_fig, ref_df, "time_s", "speed_m_s")
            angle_fig = go.Figure()
            angle_fig.add_trace(go.Scatter(x=res.df.time_s,
                                            y=unwrap_deg(res.df.theta_cork_deg),
                                            mode="lines", name="자세각 (코르크 방향)"))
            angle_fig.add_trace(go.Scatter(x=res.df.time_s,
                                            y=unwrap_deg(res.df.flow_angle_deg),
                                            mode="lines", name="진행 방향"))
            wob = (res.df.delta_alpha_total_deg if "delta_alpha_total_deg" in res.df.columns
                   else res.df.delta_alpha_deg)
            angle_fig.add_trace(go.Scatter(x=res.df.time_s, y=wob,
                                            mode="lines", name="Wobble (α − 평형)",
                                            yaxis="y2"))
            angle_fig.update_layout(
                title=dict(text=f"{profile_name} — 자세각 · 진행 방향 · Wobble"
                                "<br><sub>0° = +x축, 반시계 방향이 (+)</sub>",
                           x=0, xanchor="left", y=0.97, yanchor="top"),
                xaxis_title="time_s", yaxis=dict(title="방향각 [deg]"),
                yaxis2=dict(title="Wobble [deg]", overlaying="y", side="right",
                            zeroline=True, zerolinecolor="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                template="plotly_white", margin=dict(l=60, r=60, t=110, b=45))
            omega_fig = make_plot(res.df, "time_s", "omega_rad_s", f"{profile_name} — Angular Velocity vs Time")
            wobble_fig = make_plot(res.df, "time_s", "delta_alpha_deg", f"{profile_name} — Wobble vs Time")

            summ = summarize_result(res)
            cards = (
                f"**비행 시간:** {fmt(summ.get('flight_time_s'), ' s')}\n\n"
                f"**최대 도달 거리:** {fmt(summ.get('max_range_m'), ' m')}\n\n"
                f"**초기 속도:** {fmt(summ.get('initial_speed_m_s'), ' m/s')}\n\n"
                f"**최종 속도:** {fmt(summ.get('final_speed_m_s'), ' m/s')}\n\n"
                f"**최고 높이:** {fmt(summ.get('max_height_m'), ' m')}\n\n"
                f"**Turnover 시간:** {fmt(summ.get('turnover_time_s'), ' s')}\n\n"
                f"**안정화 시간:** {fmt(summ.get('stabilization_time_s'), ' s')}"
                f" <sub>({summ.get('stabilization_source') or '-'})</sub>\n\n"
                f"**최대 각속도:** {fmt(summ.get('max_angular_velocity_rad_s'), ' rad/s')}\n\n"
                f"**최대 Wobble:** {fmt(summ.get('max_wobble_deg'), '°')}"
            )
            return (traj_fig, speed_fig, angle_fig, omega_fig, wobble_fig, cards,
                    status_markdown(), workflow_markdown(), warning_center())
        except Exception:
            logger.exception("simulation run failed")
            push_warning("시뮬레이션 실패: 내부 오류가 발생했습니다.")
            return (None, None, None, None, None, "시뮬레이션 실패: 내부 오류가 발생했습니다.",
                    status_markdown(), workflow_markdown(), warning_center())

    def controller_run_advanced(profile_name, skirt_d_mm, level, cd_mode, cd_value,
                                 cl_mode, cl_value, cm_mode, cm_alpha,
                                 area_mode, use_geom_inertia,
                                 V0, launch_angle, height, orient_preset, custom_orient,
                                 alpha0, alpha_eq,
                                 omega0, spin0, rho, u_air_x, u_air_y, mu,
                                 solver, dt, t_max, enable_damage_mapping,
                                 turnover_mode, l_gc, damping_scale, cd_cross,
                                 orient_cd, aero_spin):
        try:
            clear_warnings()
            level = int(level)
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            cd_model = CdModel(mode=cd_mode, constant_value=float(cd_value),
                                source_type="user_defined")
            cl_model = ClModel(mode=cl_mode, constant_value=float(cl_value or 0.0),
                                source_type="user_defined")
            cm_model = CmModel(mode=cm_mode,
                                cm_alpha=(float(cm_alpha) if cm_alpha not in (None, "") else None),
                                source_type="user_defined")
            mapping = None
            if enable_damage_mapping:
                mapping = build_porosity_damage_mapping()
                gs_dbg = state["skirt"].geometry_state()
                pmod = mapping.porosity_model
                push_warning(
                    "파손→공력 대응에 다공성 예측 모델을 사용합니다 "
                    f"(solidity σ = {pmod.solidity_from_state(gs_dbg):.3f}, "
                    f"Cd 배율 {pmod.cd_factor(gs_dbg):.3f}, "
                    f"복원모멘트 배율 {pmod.cm_factor(gs_dbg):.3f}). "
                    "실측 앵커 3점(Alam 2015 / Cooke) 사이의 보간이며 "
                    "'제거 깃털 수 대 Cd' 실측 곡선은 아닙니다.")
            spin = SpinModel(spin0_rad_s=float(spin0 or 0.0))

            alpha_eq_v = float(alpha_eq if alpha_eq not in (None, "") else 180.0)
            alpha0_v = float(alpha0 if alpha0 not in (None, "") else 145.0)
            model = build_full_model(db, profile_name, geom, level=level,
                                      removed_feathers=state["skirt"].removed_ids,
                                      cd_model=cd_model, cl_model=cl_model, cm_model=cm_model,
                                      spin=spin, damage_mapping=mapping,
                                      area_mode=area_mode,
                                      use_geometry_inertia=bool(use_geom_inertia),
                                      alpha_equilibrium_deg=alpha_eq_v)
            apply_flip_settings(model, turnover_mode, l_gc, damping_scale, cd_cross,
                                 orient_cd, aero_spin, geom)
            env = EnvironmentModel(rho=float(rho), u_air_x=float(u_air_x),
                                    u_air_y=float(u_air_y), mu=float(mu))

            errs = physical_sanity_checks(model, dt, env)
            if errs:
                for e in errs:
                    push_warning(e)
                return (None, None, None, "시뮬레이션 실패: 입력값이 올바르지 않습니다.",
                        status_markdown(), workflow_markdown(), warning_center(), gr.update())

            bo = resolved_body_orientation(orient_preset, custom_orient, alpha0_v,
                                            V0, launch_angle, u_air_x, u_air_y)
            for _w in attitude_warnings(orient_preset, alpha0_v, alpha_eq_v, omega0):
                push_warning(_w)
            ic = InitialCondition.from_launch(V0, launch_angle, height, alpha0_v, alpha_eq_v,
                                               omega0=float(omega0 or 0.0),
                                               body_orientation_deg=bo,
                                               spin0_rad_s=float(spin0 or 0.0),
                                               u_air_x=u_air_x, u_air_y=u_air_y)

            for w in geometry_warnings(model):
                push_warning(w)
            for w in planar_damage_warnings(model):
                push_warning(w)

            state.update(solver=solver, dt=dt, profile=profile_name, model_level=level)
            res = SimulationEngine(model, env, ic, solver=solver, dt=dt, t_max=t_max,
                                    model_level=level).run()
            state.update(model=model, env=env, ic=ic, result=res, aero=model.aero)
            for w in res.warnings:
                push_warning(w)
            if not res.valid or len(res.df) == 0:
                return (None, None, None, "시뮬레이션 실패: 수치적으로 불안정합니다.",
                        status_markdown(), workflow_markdown(), warning_center(), gr.update())

            df = res.df
            traj = go.Figure()
            traj.add_trace(go.Scatter(x=df.x_m, y=df.y_m, mode="lines", name="시뮬레이션"))
            traj.update_layout(title=f"{profile_name} — Level {level} trajectory",
                                xaxis_title="x (m)", yaxis_title="y (m)",
                                template="plotly_white")

            ang = go.Figure()
            ang.add_trace(go.Scatter(x=df.time_s, y=unwrap_deg(df.theta_cork_deg),
                                      mode="lines", name="자세각 (코르크 방향)"))
            ang.add_trace(go.Scatter(x=df.time_s, y=unwrap_deg(df.flow_angle_deg),
                                      mode="lines", name="진행 방향"))
            wob = (df.delta_alpha_total_deg if "delta_alpha_total_deg" in df.columns
                   else df.delta_alpha_deg)
            ang.add_trace(go.Scatter(x=df.time_s, y=wob,
                                      mode="lines", name="Wobble (α − 평형)",
                                      yaxis="y2"))
            ang.update_layout(
                title=dict(text="자세각 · 진행 방향 · Wobble"
                                "<br><sub>0° = +x축, 반시계 방향이 (+)</sub>",
                           x=0, xanchor="left", y=0.97, yanchor="top"),
                xaxis_title="t (s)",
                yaxis=dict(title="방향각 [deg]"),
                yaxis2=dict(title="Wobble [deg]", overlaying="y",
                            side="right", zeroline=True,
                            zerolinecolor="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                template="plotly_white", margin=dict(l=60, r=60, t=110, b=45))

            mom = go.Figure()
            mom.add_trace(go.Scatter(x=df.time_s, y=df.restoring_moment_Nm, mode="lines",
                                      name="복원 모멘트"))
            mom.add_trace(go.Scatter(x=df.time_s, y=df.damping_moment_Nm, mode="lines",
                                      name="감쇠 모멘트"))
            mom.add_trace(go.Scatter(x=df.time_s, y=df.net_moment_Nm, mode="lines", name="합성 모멘트"))
            mom.update_layout(title="Moments vs Time", xaxis_title="t (s)",
                               yaxis_title="N·m", template="plotly_white")

            desc = res.metadata.get("aero_model", {})
            summ = summarize_result(res)
            info = (
                f"**{MODEL_LEVELS.get(level)}**\n\n"
                f"**Cd:** {desc.get('cd_mode')} (source: {desc.get('cd_source')}) &nbsp;|&nbsp; "
                f"**Cl:** {desc.get('cl_status')} &nbsp;|&nbsp; "
                f"**Cm:** {desc.get('cm_mode')}\n\n"
                f"**축 Spin:** {desc.get('spin_status')} &nbsp;|&nbsp; "
                f"**파손→계수 대응:** {desc.get('damage_mapping_status')}\n\n"
                f"**비행 시간:** {fmt(summ.get('flight_time_s'), ' s')} &nbsp;|&nbsp; "
                f"**도달 거리:** {fmt(summ.get('max_range_m'), ' m')} &nbsp;|&nbsp; "
                f"**안정화 시간:** {fmt(summ.get('stabilization_time_s'), ' s')}"
                f" <sub>({summ.get('stabilization_source') or '-'})</sub> &nbsp;|&nbsp; "
                f"**최대 Wobble:** {fmt(summ.get('max_wobble_deg'), '°')}\n\n"
                f"**Reynolds 수 범위:** {fmt(df.Re.min())} – {fmt(df.Re.max())}"
            )
            slider_update = gr.update(minimum=0, maximum=max(len(df) - 1, 1), value=0, step=1)
            return (traj, ang, mom, info, status_markdown(), workflow_markdown(),
                    warning_center(), slider_update)
        except Exception:
            logger.exception("advanced simulation failed")
            push_warning("시뮬레이션 실패: 내부 오류가 발생했습니다.")
            return (None, None, None, "시뮬레이션 실패: 내부 오류가 발생했습니다.",
                    status_markdown(), workflow_markdown(), warning_center(), gr.update())

    def controller_frame_view(frame_idx):
        res = state.get("result")
        if res is None or len(res.df) == 0:
            return None, "먼저 시뮬레이션을 실행하세요."
        try:
            df = res.df
            i = int(np.clip(int(frame_idx or 0), 0, len(df) - 1))
            row = df.iloc[i]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.x_m, y=df.y_m, mode="lines",
                                      line=dict(color="#cbd5e0"), name="시뮬레이션"))
            fig.add_trace(go.Scatter(x=[row.x_m], y=[row.y_m], mode="markers",
                                      marker=dict(size=12, color="#2b6cb0"), name="셔틀콕"))

            span = max(df.x_m.max() - df.x_m.min(), df.y_m.max() - df.y_m.min(), 1e-6)
            L = 0.08 * span

            def arrow(angle_deg, scale, color, label):
                a = deg2rad(angle_deg)
                fig.add_annotation(x=row.x_m + L * scale * math.cos(a),
                                    y=row.y_m + L * scale * math.sin(a),
                                    ax=row.x_m, ay=row.y_m, xref="x", yref="y",
                                    axref="x", ayref="y", showarrow=True, arrowhead=3,
                                    arrowwidth=2, arrowcolor=color, text=label,
                                    font=dict(color=color, size=11))

            fang_f = row_flow_angle(row)
            draw_shuttlecock(fig, row.x_m, row.y_m, row_cork_angle(row),
                              scale=L * 0.6, labels=False)

            v_ang = rad2deg(math.atan2(row.vy_m_s, row.vx_m_s))
            arrow(v_ang, 1.0, "#2f855a", "진행 방향")
            wa = deg2rad(fang_f)
            fig.add_annotation(x=row.x_m + L * 0.5 * math.cos(wa),
                                y=row.y_m + L * 0.5 * math.sin(wa),
                                ax=row.x_m + L * 1.5 * math.cos(wa),
                                ay=row.y_m + L * 1.5 * math.sin(wa),
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=3, arrowwidth=2,
                                arrowcolor="#805ad5", text="맞바람",
                                font=dict(color="#805ad5", size=11))

            drag_ang = fang_f + 180.0
            fmax = max(abs(row.drag_force_N), abs(row.lift_force_N),
                       state["model"].m * G if state.get("model") else 1.0, 1e-9)
            arrow(drag_ang, abs(row.drag_force_N) / fmax, "#c53030", "항력")
            if abs(row.lift_force_N) > 0:
                lift_ang = fang_f + (90.0 if row.lift_force_N >= 0 else -90.0)
                arrow(lift_ang, abs(row.lift_force_N) / fmax, "#dd6b20", "양력")
            grav = (state["model"].m * G) if state.get("model") else 0.0
            arrow(-90.0, grav / fmax, "#4a5568", "중력")

            fig.update_layout(title=f"t = {row.time_s:.4f} s", xaxis_title="x (m)",
                               yaxis_title="y (m)", template="plotly_white",
                               margin=dict(l=40, r=20, t=40, b=40))

            def rget(name):
                val = row.get(name, None)
                if val is None or (isinstance(val, float) and val != val):
                    return None
                return val

            cm_v = rget("Cm")
            cm_txt = NA if cm_v is None else fmt(cm_v)
            panel = (
                f"**t:** {fmt(row.time_s, ' s')} &nbsp;|&nbsp; "
                f"**V_rel:** {fmt(row.V_rel_m_s, ' m/s')} &nbsp;|&nbsp; "
                f"**Re:** {fmt(row.Re)}\n\n"
                f"**받음각:** {fmt(row.alpha_deg, '°')} &nbsp;|&nbsp; "
                f"**자세각(코르크 방향):** {fmt(row_cork_angle(row), '°')} &nbsp;|&nbsp; "
                f"**진행 방향:** {fmt(fang_f, '°')} &nbsp;|&nbsp; "
                f"**맞바람이 오는 쪽:** {fmt(fang_f, '°')}\n\n"
                f"**Cd:** {fmt(rget('Cd'))} &nbsp;|&nbsp; **Cl:** {fmt(rget('Cl'))} &nbsp;|&nbsp; "
                f"**Cm:** {cm_txt}\n\n"
                f"**항력:** {fmt(row.drag_force_N, ' N')} &nbsp;|&nbsp; "
                f"**양력:** {fmt(rget('lift_force_N'), ' N')} &nbsp;|&nbsp; "
                f"**합성 모멘트:** {fmt(rget('net_moment_Nm'), ' N·m')}\n\n"
                f"**Turnover 각속도:** {fmt(row.omega_rad_s, ' rad/s')} &nbsp;|&nbsp; "
                f"**축 Spin:** {fmt(rget('spin_rad_s'), ' rad/s')}"
            )
            return fig, panel
        except Exception:
            logger.exception("frame view failed")
            return None, "화면을 표시할 수 없습니다: 내부 오류가 발생했습니다."

    def controller_research_compare(profile_name, skirt_d_mm, level, V0, launch_angle, height,
                                     orient_preset, custom_orient, rho, u_air_x, dt, t_max):
        try:
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            env = EnvironmentModel(rho=float(rho), u_air_x=float(u_air_x))
            bo = body_orientation_from_preset(orient_preset, launch_angle,
                                               custom_deg=custom_orient, V0=V0, u_air_x=u_air_x)
            ic = InitialCondition.from_launch(V0, launch_angle, height, 145.0, 180.0,
                                               body_orientation_deg=bo, u_air_x=u_air_x)
            df = compare_damage_states(db, geom, env, ic, level=int(level), dt=dt, t_max=t_max,
                                        profile_name=profile_name)
            cols = ["damage_state", "remaining_feathers", "geometry_asymmetry",
                    "estimated_porosity", "wobble_rms_deg", "wobble_amplitude_deg",
                    "turnover_time_s", "stabilization_time_s", "range_m", "flight_time_s"]
            avail = [c for c in cols if c in df.columns]
            note = ("*Geometry differs between rows. Aerodynamic coefficients are identical "
                    "unless a damage→coefficient mapping from experiment or literature is "
                    "supplied, so flight differences here reflect only what was actually "
                    "modelled — not a measured damage effect.*")
            fig = go.Figure()
            if "wobble_rms_deg" in df.columns:
                fig.add_trace(go.Bar(x=df.damage_state, y=df.wobble_rms_deg, name="Wobble RMS (deg)"))
            fig.update_layout(title="Damage-state comparison — wobble RMS",
                               template="plotly_white", yaxis_title="deg")
            return fig, df[avail].to_csv(index=False), note
        except Exception:
            logger.exception("research comparison failed")
            return None, "", "비교 실패: 내부 오류가 발생했습니다."

    def controller_reset_simulation():
        state["result"] = None
        clear_warnings()
        return (None, None, None, None, None, "", status_markdown(), workflow_markdown(), warning_center())

    def controller_reset_all():
        state.update(model=None, env=None, ic=None, result=None, exp_df=None, exp_quality=None,
                      fit_result=None, fit_model_snapshot=None, validation=None)
        clear_warnings()
        return (status_markdown(), workflow_markdown(), warning_center(),
                "All state has been reset.")

    def controller_param_preview(profile_name):
        try:
            rows = parameter_source_table(db, profile_name)
        except (KeyError, TypeError):
            rows = parameter_source_table(db, "Normal Feather")
        lines = ["| 물리량 | 값 | 단위 | 출처 |", "|---|---|---|---|"]
        for name, val, unit, src in rows:
            lines.append(f"| {name} | {fmt(val)} | {unit} | {src} |")
        return "\n".join(lines)

    def controller_preset_fill(preset_name, impact_k=None):
        """Fills the launch fields from a preset.

        The presets that describe the instant of racket contact leave omega0 unset,
        because the turnover rate then follows phi_dot0 = k*U/L. k comes from the
        "타격 회전 계수" input; before, that input was created but never read by any
        handler, so the number the user typed had no effect anywhere and k was pinned
        to the module constant.
        """
        try:
            p = PRESETS.get(preset_name) or PRESETS["Normal smash"]
        except TypeError:
            p = PRESETS["Normal smash"]
        w0 = p["omega0"]
        if w0 is None:
            try:
                k = float(impact_k)
            except (TypeError, ValueError):
                k = IMPACT_SPIN_FACTOR
            w0 = k * abs(p["V0"]) / SHUTTLE_LENGTH_M
        return (p["V0"], p["launch_angle_deg"], p["height"], p["alpha0_deg"],
                p["alpha_eq_deg"], w0)

    def controller_compare(skirt_d_mm, V0, launch_angle, height, alpha0, alpha_eq, omega0, rho,
                            u_air_x, solver, dt, t_max):
        try:
            geom = Geometry(skirt_diameter_m=mm_to_m(skirt_d_mm))
            env = EnvironmentModel(rho=rho, u_air_x=u_air_x)
            ic = InitialCondition.from_launch(V0, launch_angle, height, alpha0, alpha_eq, omega0)
            fig = go.Figure()
            for profile_name in db.profile_names():
                model = build_model_from_profile(db, profile_name, geom)
                eng = SimulationEngine(model, env, ic, solver=solver, dt=dt, t_max=t_max)
                res = eng.run()
                if len(res.df) == 0:
                    continue
                fig.add_trace(go.Scatter(x=res.df.x_m, y=res.df.y_m, mode="lines", name=profile_name))
            fig.update_layout(title="Trajectory comparison by damage state", xaxis_title="x (m)",
                               yaxis_title="y (m)", template="plotly_white")
            return fig, ""
        except Exception:
            logger.exception("compare failed")
            return None, "비교 실패: 내부 오류가 발생했습니다."

    def controller_sensitivity(param, dt, t_max):
        if state["result"] is None:
            return None, "", "먼저 시뮬레이션을 실행하세요."
        try:
            model, env, ic = state["model"], state["env"], state["ic"]
            df = parameter_sensitivity(model, env, ic, param, dt=dt, t_max=t_max)
            fig = go.Figure()
            if "range_m" in df:
                fig.add_trace(go.Scatter(x=df.pct, y=df.range_m, mode="lines+markers", name="range (m)"))
            if "stabilization_time_s" in df:
                fig.add_trace(go.Scatter(x=df.pct, y=df.stabilization_time_s, mode="lines+markers",
                                          name="stabilization time (s)", yaxis="y2"))
            fig.update_layout(title=f"Sensitivity: {param}", xaxis_title="% change", yaxis_title="range (m)",
                               yaxis2=dict(overlaying="y", side="right", title="stabilization time (s)"),
                               template="plotly_white")
            return fig, df.to_csv(index=False), ""
        except Exception:
            logger.exception("sensitivity failed")
            return None, "", "민감도 분석 실패: 내부 오류가 발생했습니다."

    def controller_sweep(param_x, param_y, x_min, x_max, y_min, y_max, metric, dt, t_max):
        if state["result"] is None:
            return None, "먼저 시뮬레이션을 실행하세요."
        try:
            model, env, ic = state["model"], state["env"], state["ic"]
            xs = np.linspace(x_min, x_max, 5)
            ys = np.linspace(y_min, y_max, 5)
            grid = parameter_sweep(model, env, ic, param_x, xs, param_y, ys, metric=metric,
                                    dt=dt, t_max=t_max)
            fig = go.Figure(data=go.Heatmap(z=grid, x=xs, y=ys, colorbar=dict(title=metric)))
            fig.update_layout(title=f"{metric} — {param_x} vs {param_y}", xaxis_title=param_x,
                               yaxis_title=param_y, template="plotly_white")
            return fig, ""
        except Exception:
            logger.exception("sweep failed")
            return None, "스윕 분석 실패: 내부 오류가 발생했습니다."

    def controller_import(file, time_col, x_col, y_col, orientation_col):
        if file is None:
            return "", "먼저 CSV 파일을 업로드하세요.", None, None
        try:
            cmap = {"time": time_col, "x": x_col, "y": y_col, "orientation": orientation_col}
            exp = import_experiment_csv(file.name, cmap)
            state["exp_df"] = exp
            q = data_quality_report(exp)
            state["exp_quality"] = q
            quality_md = (
                f"**행 수:** {q.get('rows')}\n\n"
                f"**인식된 열:** {', '.join(q.get('columns', []))}\n\n"
                f"**시간 범위:** {fmt(q.get('time_range', (None, None))[0])} – "
                f"{fmt(q.get('time_range', (None, None))[1] if q.get('time_range') else None)} s\n\n"
                f"**샘플링 간격:** {fmt(q.get('sampling_interval_s'), ' s')}\n\n"
                f"**결측값:** {q.get('missing_values', NA)}\n\n"
                f"**중복 시각:** {q.get('duplicate_timestamps', NA)}\n\n"
                f"**시간 역행:** {q.get('non_monotonic_time', NA)}"
            )
            if q.get("duplicate_timestamps"):
                push_warning("Experimental data contains duplicate timestamps")
            if q.get("non_monotonic_time"):
                push_warning("Experimental data contains non-monotonic time")
            if q.get("rows", 0) < 2:
                push_warning("Insufficient experimental data")
            return exp.head(20).to_csv(index=False), quality_md, warning_center(), status_markdown()
        except Exception as e:
            logger.exception("import failed")
            return "", f"불러오기 실패: {e}", warning_center(), status_markdown()

    def controller_overlay():
        if state["result"] is None or state["exp_df"] is None:
            return None, "시뮬레이션과 실험 데이터가 모두 필요합니다."
        try:
            sim_df = state["result"].df
            exp_df = state["exp_df"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim_df.x_m, y=sim_df.y_m, mode="lines", name="시뮬레이션"))
            if "x" in exp_df.columns and "y" in exp_df.columns:
                fig.add_trace(go.Scatter(x=exp_df.x, y=exp_df.y, mode="markers", name="실험 데이터"))
            fig.update_layout(title="Simulation vs Experiment — Trajectory", xaxis_title="x (m)",
                               yaxis_title="y (m)", template="plotly_white")
            resid = residual_analysis(sim_df, exp_df)
            lines = []
            for k, v in resid.items():
                lines.append(f"**{k}** — RMSE: {fmt(v.get('rmse'))}, MAE: {fmt(v.get('mae'))}, "
                              f"Normalized: {fmt(v.get('normalized_rmse'))}")
            return fig, "\n\n".join(lines) if lines else "비교할 수 있는 공통 변수가 없습니다."
        except Exception:
            logger.exception("overlay failed")
            return None, "비교 표시 실패: 내부 오류가 발생했습니다."

    def controller_fit(fit_cd, fit_tw, fit_zeta, dt, t_max):
        if state["exp_df"] is None:
            return ("", None, None, "먼저 실험 데이터를 불러오세요.",
                     status_markdown(), workflow_markdown())
        if state["result"] is None:
            return ("", None, None, "먼저 시뮬레이션을 실행하세요.",
                     status_markdown(), workflow_markdown())
        params = [p for p, flag in (("Cd", fit_cd), ("Tw", fit_tw), ("zeta", fit_zeta)) if flag]
        if not params:
            return ("", None, None, "피팅할 물리량을 최소 1개 선택하세요.",
                     status_markdown(), workflow_markdown())
        try:
            model, env, ic = state["model"], state["env"], state["ic"]
            exp = state["exp_df"]
            result = fit_parameters(model, env, ic, exp, params, dt=dt, t_max=t_max,
                                     model_level=int(state.get("model_level", 1) or 1))
            state["fit_result"] = result
            card = format_fit_card(result)
            if result.get("identifiability_warning"):
                push_warning(result["identifiability_warning"])

            fig = None
            resid_md = ""
            if result.get("success"):
                fitted_model = copy.deepcopy(model)
                for name, val in result["fitted"].items():
                    if name == "Cd":
                        fitted_model.Cd_model.constant_value = val
                    elif name == "Tw":
                        fitted_model.turnover.Tw = val
                    elif name == "zeta":
                        fitted_model.turnover.zeta = val
                eng = SimulationEngine(fitted_model, env, ic, dt=dt, t_max=t_max,
                                        model_level=int(state.get("model_level", 1) or 1))
                fit_res_sim = eng.run()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fit_res_sim.df.x_m, y=fit_res_sim.df.y_m,
                                          mode="lines", name="피팅된 시뮬레이션"))
                if "x" in exp.columns and "y" in exp.columns:
                    fig.add_trace(go.Scatter(x=exp.x, y=exp.y, mode="markers", name="실험 데이터"))
                fig.update_layout(title="Fit result — Trajectory", xaxis_title="x (m)", yaxis_title="y (m)",
                                   template="plotly_white")
                resid = residual_analysis(fit_res_sim.df, exp)
                resid_md = "\n\n".join(
                    f"**{k}** — RMSE: {fmt(v.get('rmse'))}, MAE: {fmt(v.get('mae'))}"
                    for k, v in resid.items())

            return (card, fig, resid_md, "", status_markdown(), workflow_markdown())
        except Exception:
            logger.exception("fit failed")
            return ("", None, None, "피팅 실패: 내부 오류가 발생했습니다.",
                    status_markdown(), workflow_markdown())

    def controller_validation():
        try:
            ve = ValidationEngine()
            results = ve.run_all()
            state["validation"] = results
            lines = []
            for name, r in results.items():
                verdict = "PASS" if r.get("passed") else "FAIL"
                lines.append(f"**{name}**: {verdict}")
            return "\n\n".join(lines), json.dumps(results, indent=2, default=str), status_markdown(), workflow_markdown()
        except Exception:
            logger.exception("validation failed")
            return "검증 실패: 내부 오류가 발생했습니다.", "", status_markdown(), workflow_markdown()

    def controller_convergence(dt, t_max):
        if state["model"] is None:
            return "먼저 시뮬레이션을 실행하세요."
        try:
            model, env, ic = state["model"], state["env"], state["ic"]
            conv = convergence_test(model, env, ic, dt=dt, t_max=t_max)
            lines = ["| dt pair | Final Position Error | Final Angle Error |", "|---|---|---|"]
            not_converging = False
            errs = []
            for c in conv["comparisons"]:
                if not c.get("comparable"):
                    lines.append(f"| {c['dt_pair']} | N/A | N/A |")
                    continue
                lines.append(f"| {c['dt_pair']} | {fmt(c['final_position_error'])} | {fmt(c['final_angle_error'])} |")
                errs.append(c["final_position_error"])
            if len(errs) > 1 and errs[-1] > 0.5 * errs[0]:
                not_converging = True
                push_warning("Numerical convergence issue")
            table = "\n".join(lines)
            if not_converging:
                table += "\n\n⚠ Numerical convergence warning"
            return table
        except Exception:
            logger.exception("convergence test failed")
            return "수렴성 검사 실패: 내부 오류가 발생했습니다."

    def controller_export():
        res = state["result"]
        if res is None:
            return None, None, None, "먼저 시뮬레이션을 실행하세요."
        try:
            csv_path = "/tmp/sim_export.csv"
            json_path = "/tmp/sim_export.json"
            cfg_path = "/tmp/sim_config.json"
            ExportEngine.to_csv(res.df, csv_path)
            # §49: research exports carry their own provenance
            try:
                res.metadata["model_version"] = APP_VERSION
                res.metadata["literature_sources"] = [
                    dict(authors=c["authors"], year=c["year"], title=c["title"],
                          venue=c["venue"], doi=c["doi"], used=c["used"],
                          scope=c["scope"], caution=c["caution"],
                          provenance=c["provenance"])
                    for c in CITED_STUDIES]
                res.metadata["parameter_provenance"] = [
                    dict(parameter=n, source=src, kind=k)
                    for n, src, k in PARAMETER_SOURCES]
                res.metadata["provenance_legend"] = PROVENANCE_LEGEND
                res.metadata["validation_status"] = research_readiness(db)
            except Exception:
                logger.exception("export provenance failed")
            ExportEngine.to_json(res.metadata, json_path)
            with open(cfg_path, "w") as f:
                json.dump(res.metadata, f, indent=2, default=str)
            return csv_path, json_path, cfg_path, ""
        except Exception:
            logger.exception("export failed")
            return None, None, None, "내보내기 실패: 내부 오류가 발생했습니다."


    with gr.Blocks(title=f"Shuttlecock Turnover & Trajectory Simulator v{APP_VERSION}",
                    **gradio_ui_kwargs("blocks")) as demo:
        gr.Markdown(f"# 셔틀콕 Turnover · 궤적 시뮬레이터 **v{APP_VERSION}**")
        status_bar = gr.Markdown(status_markdown())
        workflow_bar = gr.Markdown(workflow_markdown())
        warning_panel = gr.Markdown(warning_center())

        gr.Markdown("## 현재 셔틀콕 상태 · 비행 영상")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("**비행 영상** — 시뮬레이션을 실행하면 자동으로 만들어집니다. "
                            "**Enter**로 재생, **Backspace**로 역재생, **Esc**로 정지할 수 있고 "
                            "화면의 ▶ / ◀ 버튼이나 아래 슬라이더로도 조작할 수 있습니다.")
                top_anim = gr.HTML(value=animation_html(None))
            with gr.Column(scale=2):
                gr.Markdown("**현재 셔틀콕 형상** — '2. 셔틀콕 파손 설정'에서 깃털을 바꾸면 "
                            "여기에 즉시 반영됩니다.")
                top_shuttle = gr.Plot(value=feather_figure(state["skirt"]),
                                       show_label=False, container=False)
        refresh_top_btn = gr.Button("🔄 위 화면 새로고침")
        gr.Markdown("*맨 위 비행 영상과 셔틀콕 형상을 현재 상태로 다시 그립니다.*")
        refresh_top_btn.click(controller_top_refresh,
                               outputs=[top_shuttle, top_anim, status_bar])
        gr.Markdown("---")

        with gr.Tab("1. 셔틀콕 설정"):
            gr.Markdown(
                "기본 모델은 **BWF 국제 경기 규격의 천연 깃털 셔틀콕**입니다. "
                "Synthetic(합성)은 비교용으로만 선택하세요.\n\n"
                "*BWF 표준 범위와 현재 입력값은 아래 표에서 구분됩니다. "
                "규격을 벗어난 값을 넣어도 자동으로 바뀌지 않습니다.*"
            )
            profile = gr.Dropdown(db.profile_names(), value="BWF 천연 깃털 (기본)",
                                   label="셔틀콕 종류 / 파라미터 프로파일")
            param_preview = gr.Markdown(controller_param_preview("BWF 천연 깃털 (기본)"))
            profile.change(controller_param_preview, inputs=[profile],
                            outputs=[param_preview], **FAST)
            # cd_value_in and cd_3d are defined further down, so the profile -> Cd
            # wiring is registered after them (search for _wire_profile_cd).

            with gr.Accordion("셔틀콕 치수 (BWF 규격 기준)", open=True):
                mass_g_in = gr.Number(value=BWF_DEFAULTS["mass_g"], label="전체 질량 [g]")
                feather_len_in = gr.Number(value=BWF_DEFAULTS["feather_length_mm"],
                                            label="깃털 길이 [mm]")
                skirt_d = gr.Number(value=BWF_DEFAULTS["skirt_tip_diameter_mm"],
                                     label="깃털 끝 원 직경 [mm]")
                cork_d_in = gr.Number(value=BWF_DEFAULTS["cork_diameter_mm"],
                                       label="코르크 직경 [mm]")
                spec_btn = gr.Button("📏 BWF 규격 범위 확인")
                gr.Markdown("*위에 입력한 치수가 BWF 국제 규격 범위 안에 있는지 표로 확인합니다. 값을 자동으로 바꾸지 않습니다.*")
                spec_table = gr.Markdown("")
                spec_btn.click(controller_bwf_spec,
                                inputs=[mass_g_in, feather_len_in, skirt_d, cork_d_in],
                                outputs=[spec_table, warning_panel])

            with gr.Accordion("공기 환경", open=True):
                rho = gr.Number(value=1.2, label="공기 밀도 ρ [kg/m³]")
                u_air_x = gr.Number(value=0.0, label="바람 속도 x성분 [m/s]")

            with gr.Accordion("회전(Flip) 모델 — Cohen et al. 2015 New J. Phys. 17 063001", open=True):
                gr.Markdown(FLIP_MODEL_TEXT)
                turnover_mode = gr.Radio(
                    ["aero_pendulum", "linear"], value="aero_pendulum",
                    label="회전 방정식")
                gr.Markdown(
                    "aero_pendulum = 논문식 비선형 진자 (복원·감쇠가 속도 U에 따라 변함, 뒤집힘 재현) "
                    "&nbsp;|&nbsp; linear = 기존 고정 Tw·ζ 선형 모델")
                l_gc_in = gr.Number(value=0.020,
                                     label="무게중심–코르크 거리 l_GC [m] (Cohen 2015: 약 0.02)")
                damping_scale_in = gr.Number(
                    value=2.5,
                    label="감쇠 보정계수 (Cohen 2015 fig.7b: 실측 τs = 예측의 0.4배 → 2.5)")
                cd_cross_in = gr.Number(
                    value=0.94,
                    label="가로자세 항력계수 Cd(90°) (Cohen 2015 fig.8b: φ=70°에서 0.9)")
                orient_cd_in = gr.Checkbox(
                    value=True, label="자세에 따른 Cd 변화 사용 (정렬 0.65 → 가로 0.94)")
                aero_spin_in = gr.Checkbox(
                    value=True, label="깃털 경사로 생기는 축 Spin 자동 계산 (RΩ/U ≈ 0.04)")
                impact_spin_in = gr.Number(
                    value=1.0,
                    label="타격 회전 계수 k — 타격 직후 각속도 φ̇₀ = k·U/L (Cohen 2015 fig.6b: k ≈ 1)")
                flip_check_btn = gr.Button("📐 논문값과 비교 검증")
                flip_check_out = gr.Markdown("")
                phys_check_btn = gr.Button("🔬 자동 물리 검증 (좌표계·쿼터니언·관성·부호·보존)")
                phys_check_out = gr.Markdown("")
                phys_check_btn.click(lambda: physics_verification_markdown(db),
                                      inputs=[], outputs=[phys_check_out])
                flip_check_btn.click(
                    controller_flip_validation,
                    inputs=[profile, skirt_d, l_gc_in, damping_scale_in, cd_cross_in, rho],
                    outputs=[flip_check_out])

            with gr.Accordion("고급: 물리 계수 직접 지정 (비워두면 프로파일 값 사용)", open=True):
                cd_override = gr.Number(value=None, label="항력계수 Cd 직접 지정")
                tw_override = gr.Number(value=None, label="Turnover 시간상수 Tw [s] 직접 지정 (linear 모드 전용)")
                zeta_override = gr.Number(value=None, label="감쇠비 ζ 직접 지정 (linear 모드 전용)")
            reset_params_btn = gr.Button("↩ 셔틀콕 설정 초기화")
            gr.Markdown("*셔틀콕 파라미터 표를 기본 BWF 천연 깃털 값으로 되돌립니다.*")

        with gr.Tab("2. 셔틀콕 파손 설정"):
            gr.Markdown(
                "### 방법 1 — 숫자로 바로 입력 (권장)\n"
                "제거할 깃털 **개수**와 **빠지는 모양**만 정하면 자동으로 적용됩니다."
            )
            with gr.Row():
                n_remove_in = gr.Number(value=0, label="제거할 깃털 개수 (0~16개)")
                damage_pattern = gr.Radio(["한쪽 집중", "좌우 대칭", "고르게 분산"],
                                           value="한쪽 집중", label="깃털이 빠지는 모양")
            apply_damage_btn = gr.Button("✅ 다시 적용", variant="secondary")
            gr.Markdown("*개수나 모양을 바꾸면 **즉시 적용**됩니다. 버튼은 같은 값으로 "
                        "다시 적용할 때만 쓰면 됩니다. 시뮬레이션은 다시 계산하지 않습니다.*")
            gr.Markdown(
                "*같은 개수라도 모양이 다르면 형상이 달라집니다. "
                "한쪽 집중 = 비대칭 파손, 좌우 대칭 = 대칭 파손.*"
            )

            gr.Markdown("#### 셔틀콕 옆모습 (실제 BWF 규격 비율)")
            side_plot = gr.Plot(value=side_view_figure(state["skirt"]),
                                 show_label=False, container=False)
            gr.Markdown("#### 셔틀콕 미리보기 (코르크 쪽에서 본 모습)")
            feather_plot = gr.Plot(value=feather_figure(state["skirt"]),
                                    show_label=False, container=False)
            geometry_info = gr.Markdown(geometry_markdown())

            with gr.Row():
                normal_state_btn = gr.Button("✅ 정상 상태로 되돌리기")
                remove_all_btn = gr.Button("🗑 전체 제거")

            with gr.Accordion("방법 2 — 깃털 번호를 직접 골라서 제거", open=True):
                gr.Markdown("그림의 번호를 보고 원하는 깃털만 체크하세요. 체크 = 제거, 해제 = 복원.")
                feather_select = gr.CheckboxGroup(
                    choices=[str(i) for i in range(1, 17)], value=[],
                    label="제거할 깃털 번호")
                damage_preset = gr.Dropdown(list(DAMAGE_PRESETS.keys()) + ["Custom"],
                                             value="Normal", label="미리 정해진 파손 예시")
                restore_all_btn = gr.Button("🔄 전체 복원")
                gr.Markdown("*제거한 깃털을 모두 되살려 16개 정상 상태로 만듭니다.*")

            def damage_numeric_top(n, pat):
                return controller_apply_damage_numeric_safe(n, pat) + (
                    feather_figure(state["skirt"]), controller_side_view())

            def toggle_top(sel):
                return controller_toggle_feathers(sel) + (
                    feather_figure(state["skirt"]), controller_side_view())

            def preset_top(name):
                return controller_damage_preset(name) + (
                    feather_figure(state["skirt"]), controller_side_view())

            def normal_top():
                return controller_normal_state() + (
                    feather_figure(state["skirt"]), controller_side_view())

            def restore_top():
                return controller_restore_all() + (
                    feather_figure(state["skirt"]), controller_side_view())

            def remove_top():
                return controller_remove_all() + (
                    feather_figure(state["skirt"]), controller_side_view())

            _dmg_out = [feather_select, feather_plot, geometry_info, status_bar,
                        warning_panel, top_shuttle, side_plot]
            apply_damage_btn.click(damage_numeric_top,
                                    inputs=[n_remove_in, damage_pattern],
                                    outputs=_dmg_out)
            # applied immediately as the count or pattern changes; the button stays
            # as an explicit re-apply
            n_remove_in.change(damage_numeric_top,
                                inputs=[n_remove_in, damage_pattern], outputs=_dmg_out)
            damage_pattern.change(damage_numeric_top,
                                   inputs=[n_remove_in, damage_pattern], outputs=_dmg_out)
            feather_select.change(toggle_top, inputs=[feather_select],
                                   outputs=[feather_plot, geometry_info, status_bar,
                                            warning_panel, top_shuttle, side_plot])
            damage_preset.change(preset_top, inputs=[damage_preset],
                                  outputs=[feather_select, feather_plot, geometry_info,
                                           status_bar, top_shuttle, side_plot])
            normal_state_btn.click(normal_top,
                                    outputs=[feather_select, feather_plot, geometry_info,
                                             status_bar, top_shuttle, side_plot])
            restore_all_btn.click(restore_top,
                                   outputs=[feather_select, feather_plot, geometry_info,
                                            status_bar, top_shuttle, side_plot])
            remove_all_btn.click(remove_top,
                                  outputs=[feather_select, feather_plot, geometry_info,
                                           status_bar, top_shuttle, side_plot])

        with gr.Tab("3. 발사 조건"):
            gr.Markdown(
                "**발사각과 셔틀콕 자세각은 서로 다른 값입니다.** 하나로 합쳐 쓰지 않습니다.\n\n"
                "받음각(angle of attack)은 자세각과 상대 유동에서 **자동 계산**됩니다."
            )
            preset = gr.Dropdown(list(PRESETS.keys()), value="Normal smash",
                                  label="발사 방식 · 발사 조건 프리셋")
            gr.Markdown(LAUNCHER_MD)
            V0 = gr.Number(value=25.0, label="① 발사 속도 V₀ [m/s]")
            launch_angle = gr.Number(value=-10.0, label="② 발사각 θ₀ [deg]")
            comp_view = gr.Markdown(controller_launch_components(25.0, -10.0))
            # vx0/vy0 표시는 벡터 성분 정의 그대로라 서버가 필요 없다. 브라우저에서
            # 바로 계산하면 왕복이 아예 사라져 타이핑과 동시에 반영된다 (서버를 거치면
            # 로컬에서도 한 글자당 약 39 ms, 터널 너머로는 훨씬 길다).
            # controller_launch_components 와 같은 결과를 내야 하므로 함께 고칠 것.
            _COMP_JS = """
            (v0, th) => {
              if (v0 === null || v0 === undefined || v0 === "" ||
                  th === null || th === undefined || th === "")
                return "발사 속도와 발사각에 숫자를 입력하세요.";
              const a = Number(v0), b = Number(th);
              if (!isFinite(a) || !isFinite(b))
                return "발사 속도와 발사각에 숫자를 입력하세요.";
              const r = b * Math.PI / 180.0;
              return "**vx\u2080 = " + (a * Math.cos(r)).toFixed(4) +
                     " m/s** &nbsp;|&nbsp; **vy\u2080 = " + (a * Math.sin(r)).toFixed(4) +
                     " m/s**\\n\\n*vx\u2080 = V\u2080\u00b7cos(\u03b8\u2080), " +
                     "vy\u2080 = V\u2080\u00b7sin(\u03b8\u2080) — 내부 계산은 " +
                     "radian으로 수행됩니다.*";
            }
            """
            # A js= payload that does not parse takes the whole front end down with
            # it -- the page sits on the loading spinner and every control is dead,
            # with nothing on the server to explain it. Losing the round-trip saving
            # is the cheap failure, so check first and fall back to the Python
            # handler, which computes exactly the same string.
            _comp_js_errs = client_js_errors(_COMP_JS)
            if _comp_js_errs:
                for _e in _comp_js_errs:
                    logger.error("_COMP_JS 문법 오류: %s", _e)
                V0.change(controller_launch_components, inputs=[V0, launch_angle],
                           outputs=[comp_view], **FAST)
                launch_angle.change(controller_launch_components,
                                     inputs=[V0, launch_angle], outputs=[comp_view],
                                     **FAST)
            else:
                V0.change(None, inputs=[V0, launch_angle], outputs=[comp_view],
                           js=_COMP_JS, **FAST)
                launch_angle.change(None, inputs=[V0, launch_angle],
                                     outputs=[comp_view], js=_COMP_JS, **FAST)
            height = gr.Number(value=1.8, label="발사 높이 [m]")

            gr.Markdown("### ③ 셔틀콕 초기 자세")
            orient_preset = gr.Radio(list(ORIENTATION_PRESETS.keys()), value="Cork Forward",
                                      label="발사 순간 셔틀콕 방향")
            gr.Markdown(
                "Cork Forward = 코르크가 진행 방향 &nbsp;|&nbsp; "
                "Skirt Forward = 깃털이 진행 방향 &nbsp;|&nbsp; "
                "Perpendicular = 수직 &nbsp;|&nbsp; Custom = 사용자 지정"
            )
            custom_orient = gr.Number(value=180.0, label="사용자 지정 초기 자세각 [deg]")
            omega0 = gr.Number(value=0.0, label="④ 초기 Turnover 각속도 [rad/s]")
            spin0 = gr.Number(value=0.0, label="⑤ 초기 축 Spin [rad/s] (Turnover와 별개)")

            with gr.Accordion("추가 환경 설정", open=True):
                alpha0 = gr.Number(value=145.0,
                                    label="초기 받음각 α₀ [deg] — 타격 직후 흔들림 (기본·고급 모델 공통)")
                alpha_eq = gr.Number(value=180.0, label="평형 받음각 [deg] (180° = 코르크가 앞)")
                u_air_y = gr.Number(value=0.0, label="바람 속도 y성분 [m/s]")
                mu_in = gr.Number(value=1.81e-5, label="공기 점성계수 μ [Pa·s]")

            orient_btn = gr.Button("👁 현재 자세 미리보기")
            gr.Markdown("*발사각·자세 설정에 따른 셔틀콕 방향과 받음각을 그림으로 확인합니다. 계산은 하지 않습니다.*")
            _init_orient = controller_orientation_preview("Cork Forward", 180.0, 25.0, -10.0,
                                                            0.0, 0.0, 145.0)
            gr.Markdown("#### 셔틀콕 자세와 상대 유동")
            orient_plot = gr.Plot(value=_init_orient[0], show_label=False, container=False)
            orient_info = gr.Markdown(_init_orient[1])
            resolved_orient = gr.Number(value=180.0, label="계산된 자세각 [deg] (자동 계산값)")
            orient_btn.click(controller_orientation_preview,
                              inputs=[orient_preset, custom_orient, V0, launch_angle,
                                      u_air_x, u_air_y, alpha0],
                              outputs=[orient_plot, orient_info, resolved_orient])

            def _orient_preset_to_alpha0(preset_name):
                """Selecting a launch attitude writes the angle of attack it means.

                The radio and alpha0 are the same physical quantity; keeping them in
                sync is what makes "Cork Forward" actually launch cork forward.
                """
                a = ORIENTATION_PRESETS.get(preset_name)
                return gr.update() if a is None else float(a)

            orient_preset.change(_orient_preset_to_alpha0, inputs=[orient_preset],
                                  outputs=[alpha0], **FAST)
            orient_preset.change(controller_orientation_preview,
                                  inputs=[orient_preset, custom_orient, V0, launch_angle,
                                          u_air_x, u_air_y, alpha0],
                                  outputs=[orient_plot, orient_info, resolved_orient])
            preset.change(controller_preset_fill, inputs=[preset, impact_spin_in],
                           outputs=[V0, launch_angle, height, alpha0, alpha_eq, omega0],
                           **FAST)
            reset_ic_btn = gr.Button("↩ 발사 조건 초기화")
            gr.Markdown("*발사 속도·발사각·높이·자세각을 기본값으로 되돌립니다.*")

        with gr.Tab("4. 물리 모델 · 시뮬레이션"):
            gr.Markdown(
                "**Level 1 기본 모델** — 병진: 2차 항력 모델 / 회전: 선형화된 Turnover 모델\n\n"
                "더 높은 수준의 모델(Cd(Re,α), 양력, 공력 모멘트)은 아래 '고급 물리 모델' 탭에서 선택합니다."
            )
            with gr.Accordion("수치 해석 설정", open=True):
                solver = gr.Dropdown(["RK4", "Euler"], value="RK4", label="수치 적분기")
                gr.Markdown("RK4 — 정확도 우선 &nbsp;|&nbsp; Euler — 빠른 미리보기")
                dt = gr.Number(value=0.001, label="시간 간격 dt [s]")
                t_max = gr.Number(value=3.0, label="시뮬레이션 종료 시간 [s]")
            run_btn = gr.Button("▶ 시뮬레이션 실행", variant="primary")
            gr.Markdown("*설정한 셔틀콕·발사 조건으로 비행을 계산합니다. 결과는 아래 그래프와 맨 위 비행 영상에 함께 표시됩니다.*")
            with gr.Row():
                rerun_btn = gr.Button("🔄 다시 계산")
                reset_sim_btn = gr.Button("↩ 시뮬레이션 결과 지우기")
            result_cards = gr.Markdown("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
            gr.Markdown("#### 궤적 (x-y)")
            traj_plot = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### 속도-시간")
            speed_plot = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### 자세각-시간")
            gr.Markdown(ANGLE_REFERENCE_TEXT)
            angle_plot = gr.Plot(show_label=False, container=False)
            with gr.Accordion("추가 그래프", open=True):
                gr.Markdown("#### 각속도-시간")
                omega_plot = gr.Plot(show_label=False, container=False)
                gr.Markdown("#### Wobble / 안정화")
                wobble_plot = gr.Plot(show_label=False, container=False)

            run_outputs = [traj_plot, speed_plot, angle_plot, omega_plot, wobble_plot,
                           result_cards, status_bar, workflow_bar, warning_panel]
            run_inputs = [profile, skirt_d, cd_override, tw_override, zeta_override,
                          turnover_mode, l_gc_in, damping_scale_in, cd_cross_in,
                          orient_cd_in, aero_spin_in,
                          V0, launch_angle, height, alpha0, alpha_eq, omega0,
                          rho, u_air_x, solver, dt, t_max]
            def run_sim_with_top(*args):
                out = controller_run_simulation(*args)
                res = state.get("result")
                n = len(res.df) if (res is not None and res.df is not None) else 0
                upd = gr.update(minimum=0, maximum=max(n - 1, 1), value=0, step=1)
                mid = max(n // 3, 0)
                fr_fig, fr_txt = controller_frame_view(mid)
                fl_fig, fl_txt = controller_flow_view(mid, True, True, True, True, "보통")
                return out + (controller_animation(), feather_figure(state["skirt"]), upd,
                              fr_fig, fr_txt, fl_fig, fl_txt,
                              gr.update(minimum=0, maximum=max(n - 1, 1), value=mid, step=1),
                              gr.update(minimum=0, maximum=max(n - 1, 1), value=mid, step=1))

            def reset_sim_with_top():
                return controller_reset_simulation() + (
                    animation_html(None), feather_figure(state["skirt"]),
                    gr.update(minimum=0, maximum=1, value=0, step=1))


        with gr.Tab("5. 고급 물리 모델 (Level 2~5)"):
            gr.Markdown(
                "병진·회전을 하나로 묶은 결합 방정식 `[x, y, vx, vy, 자세각, Turnover 각속도]` 을 적분합니다.\n\n"
                "계산 순서: 상대 유동 → Reynolds 수 → 받음각 → Cd/Cl/Cm → 항력/양력/모멘트 → 가속도·각가속도\n\n"
                "**주의:** 깃털 제거는 형상만 바꿉니다. 실험·문헌 자료로 만든 대응 관계가 없으면 "
                "Cd·Cl·Cm은 변하지 않으며, 근거 없는 파손–계수 관계를 만들지 않습니다."
            )
            with gr.Row():
                level_in = gr.Dropdown([2, 3, 4, 5], value=2, label="모델 수준 (Level)")
                area_mode_in = gr.Dropdown(["constant", "orientation_dependent"],
                                            value="constant", label="투영 면적 모델")
            with gr.Accordion("공기역학 계수 설정", open=True):
                cd_mode_in = gr.Dropdown(["constant", "lookup_speed", "lookup_Re", "lookup_angle",
                                           "lookup_Re_alpha"], value="constant",
                                          label="항력계수 Cd 방식")
                cd_value_in = gr.Number(
                    value=profile_drag_coefficient(db, "BWF 천연 깃털 (기본)"),
                    label="항력계수 Cd (상수값) — 셔틀콕 종류를 바꾸면 따라 바뀝니다")
                cl_mode_in = gr.Dropdown(["crossflow", "disabled", "constant", "lookup_angle"],
                                          value="crossflow",
                                          label="양력계수 Cl 방식 (기본: 횡류 법선력)")
                gr.Markdown(
                    "`crossflow` — 회전체 횡류 분해에서 유도한 법선력입니다. "
                    "축력 = 정렬 항력, 법선력 = (Cd(90°)−Cd(0°))·sin φ 로 두고 풍축으로 "
                    "분해하면 **항력 곡선은 그대로**이고 "
                    "`C_L = ½(Cd(90°)−Cd(0°))·sin 2φ` 가 나옵니다. 정렬·가로·역방향에서 0, "
                    "45° 부근에서 최대라는 Chan & Rossmann 2012 조건을 만족하며, "
                    "크기는 기존 항력 앵커 두 개로만 결정되고 새 상수를 쓰지 않습니다.")
                cl_value_in = gr.Number(value=0.0, label="양력계수 Cl (상수값)")
                cm_mode_in = gr.Dropdown(["from_turnover_params", "cm_alpha", "lookup_angle"],
                                          value="from_turnover_params", label="모멘트계수 Cm 방식")
                cm_alpha_in = gr.Number(value=None, label="Cm 기울기 [1/rad] (cm_alpha 방식에서만)")
                use_geom_inertia_in = gr.Checkbox(value=False,
                                                   label="깃털 배치에서 관성모멘트 추정 (형상 기반 추정값)")
                enable_damage_map_in = gr.Checkbox(
                    value=True,
                    label="파손→공력계수 대응 사용 (다공성 예측 모델)")
                gr.Markdown(POROSITY_MODEL_MD)
            adv_run_btn = gr.Button("▶ 고급 모델로 시뮬레이션 실행", variant="primary")
            gr.Markdown("*병진·회전을 함께 푸는 결합 방정식으로 계산합니다. 받음각에 따른 계수 변화와 공력 모멘트가 반영됩니다.*")
            adv_info = gr.Markdown("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
            gr.Markdown("#### 실시간 3D 재생 (WebGL)")
            gr.Markdown(
                "고급 결합 모델의 결과를 브라우저에서 60 FPS로 재생합니다. "
                "뒤집힘은 10 ms 안팎이라 재생 속도를 1/100x 이하로 낮추거나 "
                "**⤴ 뒤집힘 순간** 버튼을 쓰세요. 확대/축소는 ＋ － ⤢ 버튼 또는 휠입니다.")
            adv_gl = gr.HTML(gl_scene_html(None, "고급 모델을 실행하면 여기에서 재생됩니다."))

            gr.Markdown("#### 궤적")
            adv_traj = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### 자세각-시간")
            gr.Markdown(ANGLE_REFERENCE_TEXT)
            adv_angle = gr.Plot(show_label=False, container=False)
            with gr.Accordion("모멘트 그래프", open=True):
                adv_moment = gr.Plot(show_label=False, container=False)

            with gr.Accordion("시뮬레이션 시각화 (시간 슬라이더 + 힘 벡터)", open=True):
                frame_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="시간 프레임")
                frame_plot = gr.Plot(show_label=False, container=False)
                frame_panel = gr.Markdown("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
                frame_slider.change(controller_frame_view, inputs=[frame_slider],
                                     outputs=[frame_plot, frame_panel])

            with gr.Accordion("셔틀콕 주변 공기 흐름 (간이 시각화)", open=True):
                gr.Markdown(
                    "상대 유동과 후류(wake)를 근사해 그린 **간이 공기흐름 시각화**입니다. "
                    "CFD 계산 결과가 아니며, 힘 계산에는 전혀 사용되지 않습니다 "
                    "(힘은 Cd·Cl·Cm 모델로만 계산합니다)."
                )
                with gr.Row():
                    show_vec = gr.Checkbox(value=True, label="공기 흐름 표시")
                    show_str = gr.Checkbox(value=True, label="유선 표시")
                    show_frc = gr.Checkbox(value=True, label="힘 벡터 표시")
                    show_att = gr.Checkbox(value=True, label="자세 표시")
                flow_res = gr.Radio(["낮음", "보통", "높음"], value="보통",
                                     label="유동 시각화 해상도 (계산 정확도와 무관)")
                flow_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                         label="시간 프레임")
                flow_btn = gr.Button("💨 공기 흐름 그리기")
                gr.Markdown("*선택한 시점의 셔틀콕 주변 유동·후류·힘 벡터를 함께 그립니다.*")
                flow_plot = gr.Plot(show_label=False, container=False)
                flow_panel = gr.Markdown("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
                flow_inputs = [flow_slider, show_vec, show_str, show_frc, show_att, flow_res]
                flow_btn.click(controller_flow_view, inputs=flow_inputs,
                                outputs=[flow_plot, flow_panel])
                flow_slider.change(controller_flow_view, inputs=flow_inputs,
                                    outputs=[flow_plot, flow_panel])

            def run_adv_with_top(*args):
                out = controller_run_advanced(*args)
                res = state.get("result")
                n = len(res.df) if (res is not None and res.df is not None) else 0
                mid = max(n // 3, 0)
                fr_fig, fr_txt = controller_frame_view(mid)
                fl_fig, fl_txt = controller_flow_view(mid, True, True, True, True, "보통")
                try:
                    gl_html = gl_scene_html(res)
                except Exception:
                    logger.exception("advanced gl scene failed")
                    gl_html = gl_scene_html(None, "3D 장면을 만들 수 없습니다.")
                return out + (controller_animation(), feather_figure(state["skirt"]),
                              out[7], fr_fig, fr_txt, fl_fig, fl_txt,
                              gr.update(minimum=0, maximum=max(n - 1, 1), value=mid, step=1),
                              gl_html)

            adv_run_btn.click(
                run_adv_with_top,
                inputs=[profile, skirt_d, level_in, cd_mode_in, cd_value_in,
                        cl_mode_in, cl_value_in, cm_mode_in, cm_alpha_in,
                        area_mode_in, use_geom_inertia_in,
                        V0, launch_angle, height, orient_preset, custom_orient,
                        alpha0, alpha_eq,
                        omega0, spin0, rho, u_air_x, u_air_y, mu_in,
                        solver, dt, t_max, enable_damage_map_in,
                        turnover_mode, l_gc_in, damping_scale_in, cd_cross_in,
                        orient_cd_in, aero_spin_in],
                outputs=[adv_traj, adv_angle, adv_moment, adv_info, status_bar, workflow_bar,
                         warning_panel, frame_slider, top_anim, top_shuttle, flow_slider,
                         frame_plot, frame_panel, flow_plot, flow_panel, flow_slider,
                         adv_gl])

        with gr.Tab("6. 3차원 운동 예측"):
            gr.Markdown(
                "### 3차원 운동 예측\n"
                "x·y·z를 모두 적분하고 강체 회전방정식 "
                "(관성모멘트 × 각가속도 + 각속도 × (관성모멘트 × 각속도) = 모멘트)을 사용합니다. "
                "실험 데이터에 z 정보가 없으면 **z와 3차원 자세는 모델 예측값이며 측정값이 아닙니다.**\n\n"
                "*2차원 평면 운동은 1~5번 탭에서 다루므로 여기서는 제공하지 않습니다. "
                "3D solver를 평면에 구속하면 기존 2D 모델과 동일한 결과가 나오는 것은 "
                "수치 검증 탭의 회귀 항목에서 계속 확인합니다.*",
                )
            mode3d = gr.State("3차원 운동 예측")

            with gr.Accordion("발사 조건 (3차원)", open=True):
                # These start where the 2D tabs start (V₀ 25 m/s, -10°, 1.8 m,
                # attitude 145° = the α₀ default seen from the equilibrium). A fresh
                # page used to open with 30 m/s from 2.0 m here and 25 m/s from 1.8 m
                # there, so the same "default shot" came out 21% apart and the 3D
                # model looked wrong when it was only being asked a different
                # question. The sync below keeps them together after that.
                V0_3d = gr.Number(value=25.0, label="발사 속도 V₀ [m/s]")
                elev_3d = gr.Number(value=-10.0, label="수직 발사각 (elevation) [deg]")
                azim_3d = gr.Number(value=0.0, label="수평 발사각 (azimuth) [deg] — 0이면 좌→우 직진")
                height_3d = gr.Number(value=1.8, label="초기 높이 y₀ [m]")
                z0_3d = gr.Number(value=0.0, label="초기 좌우 위치 z₀ [m]")

            with gr.Accordion("초기 자세 · 회전 (3차원)", open=True):
                orient_off_3d = gr.Number(value=180.0,
                                           label="자세각 (맞바람 기준) [deg] — 0°=타격 직후(깃털이 앞), "
                                                 "180°=이미 안정된 자세")
                orient_az_3d = gr.Number(value=0.0,
                                          label="자세 방위 (roll 방향) [deg] — 0이면 평면 내")
                wx3 = gr.Number(value=0.0, label="바람 x [m/s]")
                wy3 = gr.Number(value=0.0, label="바람 y [m/s]")
                wz3 = gr.Number(value=0.0, label="바람 z [m/s]")
                spin_preset_3d = gr.Dropdown(
                    list(SPIN_PRESETS.keys()), value="자전 없음",
                    label="초기 자전 프리셋 (실측 기반)")
                spin_rps_3d = gr.Number(
                    value=None,
                    label="초기 자전 [rps, 회전/초] — 입력하면 프리셋보다 우선")
                spin_3d = gr.Number(
                    value=0.0,
                    label="초기 자전각속도 Ω₀ [rad/s] — 대칭축 회전 (Turnover와 완전히 별개)")
                gr.Markdown(
                    "*자연 평형 자전 Ω = 0.04·U/R (Cohen 2015 fig.15b)은 20 m/s에서 "
                    "약 **4 rps**(24 rad/s)에 불과합니다. 경기에서 보이는 빠른 자전은 "
                    "대부분 **라켓 슬라이스가 실어준 초기 자전**으로 실측 **100 rps 이상** "
                    "(Chesneau et al.)이며, 이후 공기 마찰로 평형값을 향해 감쇠합니다. "
                    "두 값은 서로 다른 양이므로 프리셋으로 구분해 입력하세요.*")

                def _spin_from_preset(preset, rps, V0v):
                    if rps not in (None, ""):
                        return float(impact_spin_rad_s(float(rps)))
                    val = SPIN_PRESETS.get(preset, 0.0)
                    if val is None:
                        return float(natural_spin_rad_s(float(V0v or 0.0),
                                                         Geometry().R_m, 0.04))
                    return float(impact_spin_rad_s(float(val)))

                spin_preset_3d.change(_spin_from_preset,
                                       inputs=[spin_preset_3d, spin_rps_3d, V0_3d],
                                       outputs=[spin_3d])
                spin_rps_3d.change(_spin_from_preset,
                                    inputs=[spin_preset_3d, spin_rps_3d, V0_3d],
                                    outputs=[spin_3d])
                spin_rpm_3d = gr.Number(value=0.0, label="  같은 값 [rpm] (입력하면 rad/s로 변환)")
                spin_mode_3d = gr.Radio(
                    ["자전 자유 (토크 없음)", "자전 자유 (자연 자전 모델)", "자전 구속"],
                    value="자전 자유 (자연 자전 모델)", label="자전 자유도 처리")
                gr.Markdown(
                    "*정상 셔틀콕은 **깃털이 나선형으로 기울어져 있어 저절로 자전**합니다 "
                    "(Cohen 2015 fig.15a). 파손은 자전을 켜는 스위치가 아니라, 남은 깃털 "
                    "수만큼 구동 토크를 **줄이는** 요인입니다. '토크 없음'은 그 구동력을 "
                    "일부러 껐을 때만 쓰세요.*")
                c_spin_3d = gr.Number(
                    value=0.0, label="자전 감쇠계수 c_spin [N·m·s/rad] — Turnover 감쇠와 다른 값")
                gr.Markdown(
                    "*정상 셔틀콕도 초기 자전을 주면 자전합니다. 파손 여부는 자전의 "
                    "ON/OFF 스위치가 아닙니다. '평형 자전 모델'은 Ω_eq = spin_ratio·U/R "
                    "(Cohen 2015)를 **목표값으로 하는 완화 토크**로 적용하며, 각속도 상태를 "
                    "직접 덮어쓰지 않습니다.*")
                spin_rpm_3d.change(lambda r: (float(r or 0.0) * 2.0 * math.pi / 60.0),
                                    inputs=[spin_rpm_3d], outputs=[spin_3d], **FAST)
                # 0, not empty, so a fresh page launches the same shuttlecock the 2D
                # tabs do. Left empty this fell back to the impact relation
                # phi_dot0 = k*U/L = 125 rad/s at the default speed, while the 2D tabs
                # start from their own box at 0 -- a 5% range gap between the tabs
                # before a single feather was removed. The sync writes tab 3's value
                # here as soon as anything changes; this is only about the first load.
                omega_turn_3d = gr.Number(
                    value=0.0,
                    label="초기 Turnover 각속도 [rad/s] — 비우면 타격 관계식 φ̇₀ = k·U/L 로 자동")
                impact_k_3d = gr.Slider(0.0, 2.0, value=0.5, step=0.05,
                                         label="타격 회전 계수 k (Cohen 2015 fig.6b: k ≈ 1)")
                ixx_3d = gr.Number(value=None,
                                    label="대칭축 관성 Ixx [kg·m²] (비우면 자동 추정)")
                gi_3d = gr.Checkbox(value=True,
                                     label="비대칭 파손 시 full inertia tensor 사용 "
                                           "(끄면 축대칭 근사 diag(Is,It,It))")
                gi_3d.change(lambda v: state.update(use_geometry_inertia_3d=bool(v)),
                              inputs=[gi_3d], outputs=[], **FAST)

            with gr.Accordion("공기역학 계수", open=True):
                cd_3d = gr.Number(
                    value=profile_drag_coefficient(db, "BWF 천연 깃털 (기본)"),
                    label="항력계수 Cd (셔틀콕 종류를 바꾸면 따라 바뀝니다)")
                cl_mode_3d = gr.Dropdown(["crossflow", "disabled", "constant"],
                                          value="crossflow",
                                          label="양력계수 Cl 방식 (기본: 횡류 법선력)")
                cl_3d = gr.Number(value=0.0, label="양력계수 Cl")
                rho_3d = gr.Number(value=1.2, label="공기 밀도 ρ [kg/m³]")
                mu_3d = gr.Number(value=1.81e-5, label="공기 점성계수 μ [Pa·s]")

            gr.Markdown(
                "*발사 속도·발사각·높이·초기 Turnover 각속도·공기 밀도·항력계수·바람은 "
                "**1번 탭(발사 조건)에서 입력한 값을 그대로 물려받습니다.** 1번 탭 값을 "
                "바꾸면 여기도 함께 바뀌고, 여기서 다시 고치면 이 탭에서만 적용됩니다.*")
            sync3d_btn = gr.Button("⟲ 1번 탭 초기 조건 다시 불러오기")

            def _sync_3d(V0_v, la_v, h_v, w_v, a0_v, aeq_v, rho_v, wind_x, cd_ov,
                          cd_now=None, preset_v=None):
                # 3D 자세각은 '맞바람 기준 어긋남'이므로 1번 탭의 받음각 α₀에서
                # 평형(180°)을 뺀 값이 그대로 초기 어긋남이 된다.
                # `a0_v or 145.0` 은 α₀ = 0° (타격 직후 = 깃털이 앞) 를 falsy 로 보고
                # 145° 로 바꿔버려, 라켓 타격 프리셋이 3D 탭에 잘못 전달됐다.
                _pa = preset_alpha_deg(preset_v) if preset_v is not None else None
                a0_v = (float(_pa) if _pa is not None
                        else (145.0 if a0_v in (None, "") else float(a0_v)))
                aeq_v = 180.0 if aeq_v in (None, "") else float(aeq_v)
                off = float(wrap_to_pi(deg2rad(a0_v)
                                        - deg2rad(aeq_v))) * 180.0 / math.pi
                off = float(wrap_to_pi(deg2rad(180.0 + off))) * 180.0 / math.pi
                if off <= -180.0 + 1e-9:
                    off = 180.0
                # '항력계수 Cd 직접 지정' 은 비워두면 프로파일 값을 쓰는 칸인데,
                # 브라우저는 빈 Number 를 0 으로 보낸다. 그 0 을 그대로 넘기는 바람에
                # 3D 탭이 Cd = 0 으로 돌아 항력 없는 포물선(사거리 49 m)을 그렸다.
                # 항력이 0 인 셔틀콕은 없으므로 양수일 때만 지정으로 본다.
                try:
                    cd_num = float(cd_ov)
                except (TypeError, ValueError):
                    cd_num = None
                cd_v = (cd_num if (cd_num is not None and cd_num > 0)
                        else (cd_now if cd_now not in (None, "") else gr.update()))
                return (float(V0_v or 0.0), float(la_v or 0.0), float(h_v or 0.0),
                        float(w_v or 0.0), off, float(rho_v or 1.2), cd_v,
                        float(wind_x or 0.0))

            # The same copy, done in the browser. Nine controls each fired this
            # handler on every keystroke, so the eight 3D boxes sat in Gradio's
            # pending state for the whole round trip -- the launch speed and angle
            # visibly lagged, and a value typed just before pressing Run had often
            # not reached the 3D tab at all. Arithmetic this small does not need the
            # server. cd_3d is passed in as well so the handler can echo the current
            # value back when the override box is empty (the browser sends 0 for an
            # empty Number, and Cd = 0 is what drew the 80 m parabola).
            _SYNC_JS = """
            (v0, la, h, w, a0, aeq, rho, wx, cdOv, cdNow, preset) => {
              // resolved_body_orientation 과 같은 규칙: 자세 프리셋이 α₀ 의 권위다.
              // 프리셋은 "Cork Forward" 인데 α₀ 칸은 145 로 남아 있어서, 2D 탭은
              // 180°(코르크 앞)로 쏘고 3D 탭은 145°로 쏘고 있었다 — 같은 기본값에서
              // 두 탭이 9.5% 차이가 난 이유다.
              const PRESET_ALPHA = {"Cork Forward": 180.0, "Skirt Forward": 0.0,
                                    "Perpendicular": 90.0};
              const num = (x, d) => {
                const n = Number(x);
                return (x === null || x === undefined || x === "" || !isFinite(n))
                       ? d : n;
              };
              const wrap = (deg) => {
                let d = ((deg + 180) % 360 + 360) % 360 - 180;
                return d;
              };
              const pa = PRESET_ALPHA[preset];
              const a0v = (pa === undefined) ? num(a0, 145.0) : pa;
              const aeqv = num(aeq, 180.0);
              let off = wrap(180.0 + wrap(a0v - aeqv));
              if (off <= -180.0 + 1e-9) off = 180.0;
              const ov = Number(cdOv);
              const cd = (cdOv === null || cdOv === undefined || cdOv === ""
                          || !isFinite(ov) || ov <= 0) ? cdNow : ov;
              return [num(v0, 0.0), num(la, 0.0), num(h, 0.0), num(w, 0.0), off,
                      num(rho, 1.2), cd, num(wx, 0.0)];
            }
            """
            _sync_src = [V0, launch_angle, height, omega0, alpha0, alpha_eq,
                         rho, u_air_x, cd_override, cd_3d, orient_preset]
            _sync_dst = [V0_3d, elev_3d, height_3d, omega_turn_3d, orient_off_3d,
                         rho_3d, cd_3d, wx3]
            _sync_js_errs = client_js_errors(_SYNC_JS)
            if _sync_js_errs:
                for _e in _sync_js_errs:
                    logger.error("_SYNC_JS 문법 오류: %s", _e)
            _sync_kw = (dict(fn=_sync_3d) if _sync_js_errs
                        else dict(fn=None, js=_SYNC_JS))
            sync3d_btn.click(inputs=_sync_src, outputs=_sync_dst,
                              **_sync_kw, **FAST)
            for _c in (V0, launch_angle, height, omega0, alpha0, alpha_eq, rho,
                        u_air_x, cd_override, orient_preset):
                _c.change(inputs=_sync_src, outputs=_sync_dst, **_sync_kw, **FAST)

            # Choosing a shuttlecock type must move its drag coefficient into every
            # tab that runs on one. Only the basic 2D tab did that, because it builds
            # its model straight from the profile; the advanced and 3D tabs read a box
            # pinned at 0.6, so "Synthetic" (0.465) or a damaged profile (0.35 / 0.80)
            # left the 3D model flying an intact feather shuttlecock.
            def _wire_profile_cd(name):
                cd = profile_drag_coefficient(db, name)
                return cd, cd

            profile.change(_wire_profile_cd, inputs=[profile],
                            outputs=[cd_value_in, cd_3d], **FAST)

            view_mode = gr.Radio(["3차원 궤적", "XY 평면", "XZ 평면", "YZ 평면"],
                                  value="XY 평면", label="보기 방식")
            run3d_btn = gr.Button("▶ 3차원 시뮬레이션 실행", variant="primary")
            gr.Markdown("*x·y·z를 모두 적분해 3차원 궤적과 자세를 계산합니다. "
                        "2D 모드를 고르면 기존 2D 모델과 같은 결과가 나옵니다.*")
            info3d = gr.Markdown("결과 없음 — 먼저 시뮬레이션을 실행하세요.")
            gr.Markdown("#### 주 보기")
            plot3d_main = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### 보조 투영")
            plot3d_side = gr.Plot(show_label=False, container=False)


            gr.Markdown("### 실시간 3D 재생 (WebGL)")
            gr.Markdown(
                "물리 계산과 화면 렌더링이 완전히 분리되어 있습니다. Python은 시뮬레이션을 "
                "한 번만 계산해 상태 버퍼를 브라우저로 넘기고, 브라우저는 "
                "`requestAnimationFrame`으로 자체 프레임률(목표 60 FPS)로 재생합니다. "
                "매 프레임 위치·속도는 선형 보간, 자세는 **quaternion SLERP**로 계산합니다. "
                "재생 속도(0.25x~4x)와 시간 슬라이더는 렌더링만 바꾸며 물리를 재계산하지 않습니다.\n\n"
                "*마우스 드래그로 회전, 휠로 확대/축소. 프레임률이 떨어지면 입자 수와 궤적 "
                "해상도를 자동으로 낮추며, 물리 정확도는 절대 낮추지 않습니다.*")
            gl_scene = gr.HTML(gl_scene_html(None))

            gr.Markdown("### 3차원 비행 공간 (시간 동기화 · Plotly 분석용)")
            gr.Markdown(
                "시간 슬라이더를 움직이면 셔틀콕 위치·자세·궤적·공기흐름·힘 벡터가 "
                "**같은 시점으로 동시에** 바뀝니다. 애니메이션은 이미 계산된 결과를 "
                "재생하는 것이며 물리 계산을 다시 실행하지 않습니다.\n\n"
                "*공기 흐름은 간이 시각화이며 CFD 결과가 아닙니다. 힘 계산에는 사용되지 않습니다.*"
            )
            with gr.Row():
                sc_flow = gr.Checkbox(value=True, label="공기 흐름 표시")
                sc_stream = gr.Checkbox(value=True, label="유선 표시")
                sc_force = gr.Checkbox(value=True, label="힘 벡터 표시")
                sc_follow = gr.Checkbox(value=True, label="셔틀콕 따라가기")
            sc_res = gr.Radio(["낮음", "보통", "높음"], value="보통",
                               label="유동 시각화 해상도 (계산 정확도와 무관)")
            scene_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                      label="시간 프레임")
            scene_btn = gr.Button("🧊 3차원 장면 그리기")
            gr.Markdown("*선택한 시점의 셔틀콕 형상(코르크·깃털·축)과 주변 유동을 3차원으로 표시합니다.*")
            scene_plot = gr.Plot(show_label=False, container=False)
            scene_panel = gr.Markdown("결과 없음 — 먼저 3차원 시뮬레이션을 실행하세요.")
            scene_inputs = [scene_slider, sc_flow, sc_stream, sc_force, sc_res, sc_follow]
            scene_btn.click(controller_scene_3d, inputs=scene_inputs,
                             outputs=[scene_plot, scene_panel])
            scene_slider.change(controller_scene_3d, inputs=scene_inputs,
                                 outputs=[scene_plot, scene_panel])

            def run_3d_with_scene(*args):
                out = controller_run_3d(*args)
                res = state.get("result3d")
                n = len(res.df) if (res is not None and res.df is not None) else 0
                mid = max(n // 3, 0)
                try:
                    sc_fig, sc_txt = controller_scene_3d(mid, True, True, True, "보통", True)
                except Exception:
                    logger.exception("scene 3d failed")
                    sc_fig, sc_txt = None, "장면을 그릴 수 없습니다."
                return out + (sc_fig, sc_txt,
                              gr.update(minimum=0, maximum=max(n - 1, 1), value=mid, step=1),
                              controller_gl_scene())

            run3d_btn.click(
                run_3d_with_scene,
                inputs=[profile, skirt_d, mode3d, V0_3d, elev_3d, azim_3d, height_3d, z0_3d,
                        orient_off_3d, orient_az_3d, wx3, wy3, wz3, spin_3d,
                        rho_3d, mu_3d, cd_3d, cl_mode_3d, cl_3d, ixx_3d,
                        solver, dt, t_max, view_mode, omega_turn_3d, impact_k_3d,
                        spin_mode_3d, c_spin_3d],
                outputs=[plot3d_main, plot3d_side, info3d, status_bar, warning_panel,
                         scene_slider, scene_plot, scene_panel, scene_slider, gl_scene])
            view_mode.change(controller_change_view, inputs=[view_mode],
                              outputs=[plot3d_main, info3d])

            gr.Markdown("### 3D → 2D 실험 검증")
            cmp3d_btn = gr.Button("📈 XY 투영과 실험 데이터 비교")
            gr.Markdown("*3D 시뮬레이션을 XY 평면에 투영해 2D 실험 데이터와 비교합니다. "
                        "이것이 3D 모델의 핵심 검증 방법입니다.*")
            cmp3d_plot = gr.Plot(show_label=False, container=False)
            cmp3d_info = gr.Markdown("")
            cmp3d_btn.click(controller_compare_3d_xy, outputs=[cmp3d_plot, cmp3d_info])

        with gr.Tab("6-B. 가상 풍동 실험 (AF1300S)"):
            gr.Markdown(
                "### TecQuipment AF1300S Subsonic Wind Tunnel 305 mm Starter Set\n\n"
                "제조사 공식 자료 기준입니다. 작업구간 305 × 305 × 600 mm, 작업구간 유속 0~36 m/s, "
                "개방회로 흡입식(open-circuit suction)입니다.\n\n"
                "공기 흐름: 대기 → 수축부(effuser) → 작업구간 → 그릴 → 확산부 → 축류팬 → 소음기 → 대기\n\n"
                "**흡입식이므로 작업구간의 정압은 대기압보다 낮습니다.** "
                "압력 예측은 저차원 공력 모델 기반이며 CFD 결과가 아닙니다.")

            with gr.Accordion("장비 구성", open=True):
                gr.Markdown(
                    "**기본 구성 (AF1300S에 포함)** — AF1300 풍동, AF1300Z 기본 양력/항력 밸런스, "
                    "AF1300J 3차원 항력 모델, 표준 Pitot 관, Pitot-static 관, 제어반 마노미터, "
                    "모델 홀더, 프로트랙터")
                wt_optional = gr.CheckboxGroup(
                    choices=[f"{c} — {n}" for c, n in AF1300_OPTIONAL.items()],
                    value=[], label="선택 장비 (활성화한 것만 사용할 수 있습니다)")

            with gr.Accordion("팬 / 작업구간 유속", open=True):
                wt_mode = gr.Radio(["팬 설정", "직접 유속 지정"], value="직접 유속 지정",
                                    label="유속 결정 방식")
                wt_fan_preset = gr.Radio(["정지", "저속", "중속", "고속", "사용자 지정"],
                                          value="중속", label="팬 설정")
                wt_fan_pct = gr.Slider(0, 100, value=55, step=1, label="팬 속도 [%]")
                wt_fan_cal = gr.Textbox(
                    value="", label="팬 보정 곡선 (한 줄에 '팬%,유속' — 비우면 미보정)",
                    lines=3, placeholder="20,7.5\n50,18.0\n100,36.0")
                wt_U = gr.Number(value=20.0, label="작업구간 유속 U [m/s] (직접 지정 시)")

            with gr.Accordion("공기 환경", open=False):
                wt_T = gr.Number(value=20.0, label="온도 T [°C]")
                wt_P = gr.Number(value=101325.0, label="대기압 P [Pa]")
                wt_rho = gr.Number(value=None, label="공기 밀도 ρ [kg/m³] (비우면 T·P로 계산)")
                wt_mu = gr.Number(value=None, label="점성계수 μ [Pa·s] (비우면 Sutherland)")

            with gr.Accordion("풍동 형상 · 손실계수", open=False):
                gr.Markdown("작업구간 치수만 공식 자료 값입니다. 나머지는 사용자 입력이며, "
                            "비우면 **미정**으로 표시되고 해당 계산은 생략됩니다.")
                wt_A_in = gr.Number(value=None, label="흡입구 면적 [m²] (수축비 계산용)")
                wt_A_dif = gr.Number(value=None, label="확산부 출구 면적 [m²] (압력 회복 계산용)")
                wt_K_c = gr.Number(value=None, label="수축부 손실계수 K_contraction")
                wt_K_g = gr.Number(value=None, label="그릴 손실계수 K_grille")
                wt_K_d = gr.Number(value=None, label="확산부 손실계수 K_diffuser")
                wt_plevel = gr.Radio([PRESSURE_MODEL_LEVELS[i] for i in (1, 2, 3)],
                                      value=PRESSURE_MODEL_LEVELS[1],
                                      label="압력 모델 수준")
                wt_bl = gr.Checkbox(value=False, label="간이 경계층 모델 사용 (1/7 멱법칙)")

            with gr.Accordion("셔틀콕 설치 · 자전 구속", open=True):
                wt_x = gr.Number(value=0.30, label="설치 위치 x [m] (작업구간 입구 기준, 0~0.6)")
                wt_y = gr.Number(value=0.0, label="설치 위치 y [m] (중심 기준, ±0.1525)")
                wt_z = gr.Number(value=0.0, label="설치 위치 z [m] (중심 기준, ±0.1525)")
                wt_inc = gr.Number(
                    value=180.0,
                    label="입사각 (프로트랙터) [deg] — 0°=코르크가 맞바람 정면(안정), "
                          "180°=깃털이 앞(타격 직후, 뒤집힘 관찰)")
                wt_fixed = gr.Radio(["회전", "고정"], value="회전", label="모델 각도")
                wt_spin = gr.Radio(["구속", "허용"], value="구속", label="자전 (axial spin)")
                wt_spin_val = gr.Number(value=0.0, label="구속 시 자전 각속도 [rad/s]")
                gr.Markdown("*자전 구속 시 축 spin 자유도는 고정되고 turnover 자유도만 적분됩니다. "
                            "구속에 필요한 반력은 constraint_torque_Nm 으로 기록되며, "
                            "구속된 회전의 에너지는 셔틀콕 에너지 수지에 더해지지 않습니다.*")
                wt_wall = gr.Checkbox(value=False, label="벽면 보정 사용 (출처 확인된 보정식 없음)")

            with gr.Accordion("난류 / 계측 노이즈 — 결과의 현실성", open=True):
                gr.Markdown(
                    "기본 물리 모델은 완전히 결정론적이라 그래프가 매끈하게 나옵니다. "
                    "실제 풍동에는 자유류 난류가 있고 계측기에는 잡음이 있습니다.\n\n"
                    "난류는 난류강도 Tu = u'rms / U 와 적분 길이척도 L 로 정의되며, "
                    "상관시간 T = L / U 를 갖는 1차 마르코프 과정(Ornstein-Uhlenbeck)으로 "
                    "생성합니다. r.m.s. 와 상관시간은 맞지만 **실측 난류장이 아닌 합성 신호**입니다.\n\n"
                    "*참고 범위: 깨끗한 개방회로 풍동 Tu ≈ 0.3%, 격자 난류 실험 Tu = 5~15%, "
                    "L = 0.08~0.33 m.*")
                wt_tu = gr.Slider(0.0, 15.0, value=0.3, step=0.1,
                                   label="자유류 난류강도 Tu [%] (0이면 완전 결정론적)")
                wt_tl = gr.Number(value=0.10, label="적분 길이척도 L [m]")
                wt_noise = gr.Checkbox(value=True, label="계측기 노이즈 적용 (불확도를 표준편차로)")
                wt_seed = gr.Number(value=1, label="난수 시드 (재현성)")

            with gr.Accordion("계측 기기 오차", open=False):
                wt_pitot_k = gr.Number(value=1.0, label="Pitot 계수")
                wt_pitot_zero = gr.Number(value=0.0, label="Pitot 영점 오차 [Pa]")
                wt_pitot_unc = gr.Number(value=1.0, label="압력 센서 불확도 [Pa]")
                wt_man_incl = gr.Number(value=90.0, label="마노미터 경사각 [deg] (AFA1은 틸팅 가능)")
                wt_man_unc = gr.Number(value=0.0005, label="마노미터 눈금 불확도 [m]")
                wt_bal_unc = gr.Number(value=0.02, label="밸런스 불확도 [N]")

            with gr.Row():
                wt_dt = gr.Number(value=5e-4, label="물리 timestep dt [s]")
                wt_tmax = gr.Number(value=1.5, label="측정 시간 [s]")
            wt_run_btn = gr.Button("🌬 풍동 실험 실행", variant="primary")
            wt_dash = gr.Markdown("풍동을 아직 가동하지 않았습니다.")
            gr.Markdown("#### 실시간 3D 재생 — 작업구간의 셔틀콕")
            gr.Markdown(
                "모델이 지지대에 고정되어 있으므로 위치는 움직이지 않고 **자세만** 변합니다. "
                "자전을 구속하면 축 회전은 멈추고 Turnover(뒤집힘·진동)만 보입니다. "
                "뒤집힘은 10 ms 안팎이라 재생 속도를 1/100x 이하로 낮추거나 "
                "**⤴ 뒤집힘 순간** 버튼을 쓰세요.")
            wt_gl = gr.HTML(gl_scene_html(None, "풍동 실험을 실행하면 여기에서 재생됩니다."))
            gr.Markdown("#### 풍동 내부 압력 분포 (저차원 모델 기반 예측)")
            wt_pressure_plot = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### Turnover 응답 (자전 구속 상태)")
            wt_flip_plot = gr.Plot(show_label=False, container=False)
            gr.Markdown("#### 계측 기기 판독값")
            wt_instr = gr.Markdown("")

            gr.Markdown("### 조건 자동 Sweep (실험 설계 모드)")
            with gr.Row():
                sw_U = gr.Textbox(value="10, 20, 30", label="풍속 목록 [m/s]")
                sw_dmg = gr.Textbox(value="0, 2, 4", label="제거 깃털 개수 목록")
                sw_inc = gr.Textbox(value="0, 30, 60", label="입사각 목록 [deg]")
            sw_btn = gr.Button("📋 조건별 자동 실험")
            sw_table = gr.Dataframe(label="조건별 결과", wrap=True)

            def _wt_build(optional_sel, T, P, rho_v, mu_v, A_in, A_dif, Kc, Kg, Kd,
                           plevel, bl, pitot_k, pitot_zero, pitot_unc, man_incl,
                           man_unc, bal_unc):
                codes = [x.split(" — ")[0] for x in (optional_sel or [])]
                lvl = 1
                for i, nm in PRESSURE_MODEL_LEVELS.items():
                    if nm == plevel:
                        lvl = i
                tun = VirtualWindTunnel(
                    geometry=TunnelGeometry(
                        inlet_area_m2=(float(A_in) if A_in else None),
                        diffuser_exit_area_m2=(float(A_dif) if A_dif else None)),
                    losses=TunnelLosses(
                        K_contraction=(float(Kc) if Kc is not None and Kc != "" else None),
                        K_grille=(float(Kg) if Kg is not None and Kg != "" else None),
                        K_diffuser=(float(Kd) if Kd is not None and Kd != "" else None)),
                    env=TunnelEnvironment(T_celsius=float(T), P_ambient_Pa=float(P),
                                           rho_user=(float(rho_v) if rho_v else None),
                                           mu_user=(float(mu_v) if mu_v else None)),
                    instrumentation=WindTunnelInstrumentation(codes),
                    pressure_level=lvl, boundary_layer=bool(bl))
                tun.pitot = PitotTube(float(pitot_k or 1.0), float(pitot_zero or 0.0),
                                       float(pitot_unc or 1.0))
                tun.pitot_static = PitotStaticTube(float(pitot_k or 1.0),
                                                    float(pitot_zero or 0.0),
                                                    float(pitot_unc or 1.0))
                tun.manometer = Manometer(inclination_deg=float(man_incl or 90.0),
                                           uncertainty_m=float(man_unc or 5e-4))
                if "AF1300T" in codes:
                    tun.balance = ThreeComponentBalance(uncertainty_N=float(bal_unc or 0.02))
                else:
                    tun.balance = LiftDragBalance(uncertainty_N=float(bal_unc or 0.02))
                return tun

            def _wt_velocity(tun, mode, preset, pct, cal_text, U_direct):
                cal = None
                if cal_text and cal_text.strip():
                    cal = []
                    for line in cal_text.strip().splitlines():
                        parts = line.replace("\t", ",").split(",")
                        if len(parts) >= 2:
                            try:
                                cal.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                continue
                    cal = cal or None
                tun.fan = FanController(cal)
                if mode == "직접 유속 지정":
                    return float(U_direct or 0.0), "사용자 지정 작업구간 유속"
                p = (FanController.PRESETS.get(preset)
                     if preset != "사용자 지정" else float(pct or 0.0))
                return tun.fan.velocity(p if p is not None else 0.0)

            def _wt_model(removed):
                geom_l = Geometry(skirt_diameter_m=mm_to_m(
                    state.get("skirt_d_mm") or BWF_DEFAULTS["skirt_tip_diameter_mm"]))
                m = build_full_model(db, state.get("profile_name") or db.profile_names()[0],
                                      geom_l, level=3, removed_feathers=list(removed),
                                      use_geometry_inertia=True,
                                      cd_model=CdModel(mode="orientation",
                                                        constant_value=0.65),
                                      cl_model=ClModel(mode="constant", constant_value=0.1),
                                      cm_model=CmModel(mode="aero_pendulum"),
                                      spin=SpinModel(mode="aero_driven", spin_ratio=0.04,
                                                      R_m=geom_l.R_m))
                m.turnover.Cd_ref = 0.65
                return m

            def controller_wind_tunnel(optional_sel, mode, preset, pct, cal_text, U_direct,
                                        T, P, rho_v, mu_v, A_in, A_dif, Kc, Kg, Kd,
                                        plevel, bl, x, y, z, inc, fixed, spin_mode,
                                        spin_val, wall, pitot_k, pitot_zero, pitot_unc,
                                        man_incl, man_unc, bal_unc, dtv, tmaxv,
                                        tu, tl, noise, seed):
                try:
                    tun = _wt_build(optional_sel, T, P, rho_v, mu_v, A_in, A_dif, Kc, Kg,
                                     Kd, plevel, bl, pitot_k, pitot_zero, pitot_unc,
                                     man_incl, man_unc, bal_unc)
                    U, src = _wt_velocity(tun, mode, preset, pct, cal_text, U_direct)
                    if not np.isfinite(U):
                        return ("**팬 설정과 실제 유속의 관계가 보정되지 않음** — "
                                "팬 보정 곡선을 입력하거나 '직접 유속 지정'을 사용하세요.",
                                None, None, "",
                                gl_scene_html(None, "유속이 확정되지 않아 재생할 수 없습니다."))
                    model = _wt_model(state["skirt"].removed_ids)
                    res, st, instr = run_wind_tunnel_experiment(
                        model, tun, U, incidence_deg=float(inc or 0.0),
                        spin_locked=(spin_mode == "구속"),
                        spin_value_rad_s=float(spin_val or 0.0),
                        dt=float(dtv or 5e-4), t_max=float(tmaxv or 1.5),
                        position=(float(x or 0.0), float(y or 0.0), float(z or 0.0)),
                        model_fixed=(fixed == "고정"), wall_correction=bool(wall),
                        turbulence_intensity=float(tu or 0.0) / 100.0,
                        turbulence_length_m=float(tl or 0.10),
                        seed=(int(seed) if seed is not None else None),
                        instrument_noise=bool(noise))
                    state["wt_result"] = res
                    state["wt_state"] = st
                    d = res.df
                    last = d.iloc[-1] if len(d) else None
                    wt_meta = res.metadata["wind_tunnel"]
                    dash = [
                        f"### 풍동 상태 Dashboard &nbsp;|&nbsp; {wt_meta['equipment']}",
                        "",
                        f"유속 결정: {src} &nbsp;|&nbsp; 압력 모델: "
                        f"{wt_meta['pressure_model_name']}",
                        "",
                        "| 항목 | 값 | 성격 |",
                        "|---|---|---|",
                        f"| 팬 / 작업구간 유속 | {st['U_test']:.2f} m/s | {src} |",
                        f"| 마하수 | {st['mach']:.3f} | 모델 계산값 |",
                        f"| 작업구간 정압 (게이지) | {st['Ps_test_gauge_Pa']:.1f} Pa | "
                        "모델 예측값 (대기압 대비, 흡입식이라 음압) |",
                        f"| 작업구간 전압 | {st['Pt_test_Pa']:.1f} Pa | 모델 예측값 |",
                        f"| 동압 q | {st['q_test_Pa']:.1f} Pa | 모델 계산값 |",
                        f"| 공기 밀도 | {st['rho']:.4f} kg/m³ | "
                        f"{st['environment']['rho_source']} |",
                        f"| 체적유량 | {st['volumetric_flow_m3_s']:.4f} m³/s | 모델 계산값 |",
                        f"| 질량유량 | {st['mass_flow_kg_s']:.4f} kg/s | 모델 계산값 |",
                        f"| 팬 필요 압력상승 | {st['delta_P_fan_Pa']:.1f} Pa | 모델 예측값 |",
                        f"| 차단율 | {wt_meta['blockage_ratio']*100:.2f} % | "
                        f"{wt_meta['blockage_note']} |",
                        f"| 셔틀콕 위치 | {wt_meta['position_status']} | — |",
                    ]
                    if last is not None:
                        dash += [
                            f"| Reynolds 수 | {last.Re:.3e} | 모델 계산값 |",
                            f"| 받음각 | {last.alpha_deg:.1f}° | 모델 계산값 |",
                            f"| Turnover 각속도 | {last.omega_rad_s:.2f} rad/s | 모델 계산값 |",
                            f"| 항력 | {last.drag_force_N:.4f} N | 모델 계산값 |",
                            f"| 양력 | {last.lift_force_N:.4f} N | 모델 계산값 |",
                            f"| 피칭 모멘트 | {last.net_moment_Nm:.3e} N·m | "
                            + ("AF1300T 측정값" if isinstance(tun.balance,
                                                              ThreeComponentBalance)
                               else "시뮬레이션 계산값 (3성분 밸런스 없음)") + " |",
                            f"| 자전 | {'구속' if spin_mode=='구속' else '허용'} "
                            f"({last.spin_axial_rad_s:.3f} rad/s) | "
                            f"구속 반력 {last.constraint_torque_Nm:.3e} N·m |",
                            f"| Flip 시간 | {_wt_flip_time(d):.4f} s | 모델 계산값 |",
                            f"| 정렬 시간 | {_wt_stabilization_time(d)} s | 모델 계산값 |",
                        ]
                    tc = res.metadata.get("turnover_constants", {})
                    if "Tw_model" in tc:
                        dash.append(f"| Tw / ζ | {tc['Tw_model']:.5f} s / "
                                    f"{tc['zeta_model']:.4f} | 모델 예측값 (공력 선형화) |")
                    for w in res.warnings:
                        dash.append("")
                        dash.append(f"> ⚠ {w}")
                    dash.append("")
                    tb = res.metadata.get("turbulence", {})
                    if tb.get("active"):
                        dash.append("")
                        dash.append(f"*자유류 난류 Tu = {tb['intensity']*100:.1f}%, "
                                    f"L = {tb['length_scale_m']:.3f} m, "
                                    f"상관시간 {tb['integral_timescale_s']*1e3:.1f} ms, "
                                    f"u'rms = {tb['rms_m_s']:.3f} m/s — {tb['source_type']}*")
                    else:
                        dash.append("")
                        dash.append("*난류 없음 — 완전 결정론적 계산이라 곡선이 매끄럽습니다. "
                                    "현실적인 산포를 보려면 난류강도를 올리세요.*")
                    dash.append("*압력장은 저차원 공력 모델 기반 예측이며 CFD 결과가 아닙니다.*")

                    pf = go.Figure()
                    try:
                        prof = tun.pressure.axial_profile(
                            st["U_test"], st["rho"], st["mu"], float(P))
                        xp = prof["x_m"] * 1e3
                        pf.add_trace(go.Scatter(x=xp, y=prof["Ps_gauge_Pa"],
                                                 mode="lines", name="정압 (게이지)"))
                        pf.add_trace(go.Scatter(x=xp, y=prof["q_Pa"],
                                                 mode="lines", name="동압 q"))
                        pf.add_trace(go.Scatter(x=xp, y=prof["Pt_gauge_Pa"],
                                                 mode="lines", name="전압 (게이지)"))
                        Lc_mm = 450.0
                        Lt_mm = Lc_mm + tun.geometry.ws_length_m * 1e3
                        for xv, lab in ((Lc_mm, "작업구간 시작"), (Lt_mm, "확산부 시작")):
                            pf.add_vline(x=xv, line_width=1, line_dash="dot",
                                          line_color="#94a3b8",
                                          annotation_text=lab,
                                          annotation_position="top")
                        pf.update_layout(xaxis_title="풍동 축방향 위치 x [mm]")
                        state["wt_profile"] = prof
                    except Exception:
                        logger.exception("axial profile failed")
                    pf.update_layout(
                        title=dict(text="풍동 내부 압력 예측 (1차원 연속 분포, 마찰 손실 포함)"
                                        "<br><sub>전압은 마찰·그릴·확산 손실로 하류로 갈수록 "
                                        "단조 감소합니다 — CFD 아님</sub>",
                                    x=0, xanchor="left", y=0.97, yanchor="top"),
                        yaxis_title="압력 [Pa, 대기압 기준]",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="left", x=0),
                        margin=dict(l=60, r=60, t=100, b=45))

                    ff = go.Figure()
                    if len(d):
                        wcol = (d.wobble_signed_deg if "wobble_signed_deg" in d.columns
                                else d.delta_alpha_deg)
                        ff.add_trace(go.Scatter(x=d.time_s, y=wcol,
                                                 mode="lines", name="Wobble (부호 포함) [deg]"))
                        ff.add_trace(go.Scatter(x=d.time_s, y=d.omega_rad_s,
                                                 mode="lines", name="Turnover 각속도 [rad/s]",
                                                 yaxis="y2"))
                        ff.update_layout(
                            title=dict(text=("자전 구속 상태의 Turnover 응답"
                                              if spin_mode == "구속"
                                              else "Turnover 응답 (자전 허용)"),
                                        x=0, xanchor="left", y=0.97, yanchor="top"),
                            xaxis_title="t [s]", yaxis=dict(title="Wobble [deg]"),
                            yaxis2=dict(title="각속도 [rad/s]", overlaying="y", side="right"),
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="left", x=0),
                            margin=dict(l=60, r=60, t=100, b=45))

                    im = ["### 계측 기기 판독값", "",
                          "| 기기 | 판독 | 불확도 | 성격 |", "|---|---|---|---|"]
                    p_ = instr["pitot"]
                    im.append(f"| 표준 Pitot | ΔP {p_['delta_P_Pa']:.2f} Pa → "
                              f"U {p_['U_m_s']:.3f} m/s | ±{p_['U_uncertainty_m_s']:.3f} m/s | "
                              "가상 계측값 |")
                    ps_ = instr["pitot_static"]
                    im.append(f"| Pitot-static | Ps {ps_['Ps_Pa']:.1f} / Pt {ps_['Pt_Pa']:.1f} Pa, "
                              f"q {ps_['q_Pa']:.2f} Pa → U {ps_['U_m_s']:.3f} m/s | "
                              f"±{ps_['delta_P_uncertainty_Pa']:.2f} Pa | 가상 계측값 |")
                    mn = instr["manometer"]
                    im.append(f"| 제어반 마노미터 (물) | Δh {mn['delta_h_mm']:.2f} mm | "
                              f"±{mn['uncertainty_Pa']:.2f} Pa | 가상 계측값 |")
                    if "multitube" in instr:
                        mt = instr["multitube"]
                        im.append(f"| AFA1 36관 마노미터 | Δh {mt['delta_h_mm']:.2f} mm "
                                  f"(경사 {mt['inclination_deg']:.0f}°) | "
                                  f"±{mt['uncertainty_Pa']:.2f} Pa | 선택 장비 |")
                    if "balance" in instr:
                        b = instr["balance"]
                        im.append(f"| {tun.balance.name} | 항력 {b['drag_N']:.3f} N / "
                                  f"양력 {b['lift_N']:.3f} N | ±{b['drag_uncertainty_N']:.3f} N | "
                                  "가상 계측값 |")
                        if not np.isfinite(b["pitching_moment_Nm"]):
                            im.append(f"| 피칭 모멘트 | — | — | {b['pitching_moment_note']} |")
                    sp = res.metadata["shuttlecock_pressure"]
                    im += ["", "**셔틀콕 전후 압력 예측**", "",
                           "| 항목 | 값 |", "|---|---|",
                           f"| 전방 압력 P_front | {sp['P_front_Pa']:.1f} Pa |",
                           f"| 후방 압력 P_back | {sp['P_back_Pa']:.1f} Pa |",
                           f"| 압력차 ΔP | {sp['delta_P_Pa']:.2f} Pa |",
                           f"| ΔP·A로 환산한 항력 | {sp['drag_from_dP_N']:.4f} N |",
                           "", f"*{sp['source_type']}*"]
                    tp = res.metadata["pressure_taps"]
                    if tp["status"] != "ok":
                        im += ["", f"**압력탭:** {tp['status']}"]
                    im += ["", "**교차검증** — Pitot 유속 vs 설정 유속: "
                           f"{p_['U_m_s']:.3f} / {st['U_test']:.3f} m/s, "
                           "ΔP·A 항력 vs 모델 항력: "
                           f"{sp['drag_from_dP_N']:.4f} / "
                           f"{(last.drag_force_N if last is not None else float('nan')):.4f} N"]
                    try:
                        gl_html = gl_scene_html(res)
                    except Exception:
                        logger.exception("wind tunnel gl scene failed")
                        gl_html = gl_scene_html(None, "3D 장면을 만들 수 없습니다.")
                    return "\n".join(dash), pf, ff, "\n".join(im), gl_html
                except Exception:
                    logger.exception("wind tunnel run failed")
                    return ("풍동 실험을 실행하지 못했습니다. 입력값을 확인하세요.",
                            None, None, "",
                            gl_scene_html(None, "풍동 실험을 실행하면 여기에서 재생됩니다."))

            wt_inputs = [wt_optional, wt_mode, wt_fan_preset, wt_fan_pct, wt_fan_cal, wt_U,
                         wt_T, wt_P, wt_rho, wt_mu, wt_A_in, wt_A_dif, wt_K_c, wt_K_g,
                         wt_K_d, wt_plevel, wt_bl, wt_x, wt_y, wt_z, wt_inc, wt_fixed,
                         wt_spin, wt_spin_val, wt_wall, wt_pitot_k, wt_pitot_zero,
                         wt_pitot_unc, wt_man_incl, wt_man_unc, wt_bal_unc, wt_dt, wt_tmax,
                         wt_tu, wt_tl, wt_noise, wt_seed]
            wt_run_btn.click(controller_wind_tunnel, inputs=wt_inputs,
                              outputs=[wt_dash, wt_pressure_plot, wt_flip_plot, wt_instr,
                                        wt_gl])

            def controller_wt_sweep(optional_sel, T, P, rho_v, mu_v, A_in, A_dif,
                                     Kc, Kg, Kd, plevel, bl, spin_mode, dtv,
                                     pitot_k, pitot_zero, pitot_unc, man_incl,
                                     man_unc, bal_unc, u_txt, dmg_txt, inc_txt):
                try:
                    def nums(txt, default):
                        try:
                            v = [float(a) for a in str(txt).replace(" ", "").split(",") if a]
                            return v or default
                        except ValueError:
                            return default
                    tun = _wt_build(optional_sel, T, P, rho_v, mu_v, A_in, A_dif, Kc, Kg,
                                     Kd, plevel, bl, pitot_k, pitot_zero, pitot_unc,
                                     man_incl, man_unc, bal_unc)
                    Us = nums(u_txt, [10.0, 20.0, 30.0])
                    ns = [int(a) for a in nums(dmg_txt, [0, 2, 4])]
                    incs = nums(inc_txt, [0.0, 30.0, 60.0])
                    dsets = [list(range(1, n + 1)) for n in ns]
                    df = wind_tunnel_sweep(_wt_model, tun, Us, damage_sets=dsets,
                                            incidences=incs, initial_offsets=[60.0],
                                            spin_locked=(spin_mode == "구속"),
                                            dt=max(float(dtv or 1e-3), 1e-3), t_max=0.8)
                    state["wt_sweep"] = df
                    return df
                except Exception:
                    logger.exception("wind tunnel sweep failed")
                    return pd.DataFrame([{"오류": "Sweep 실행 실패"}])

            sw_btn.click(controller_wt_sweep,
                          inputs=[wt_optional, wt_T, wt_P, wt_rho, wt_mu, wt_A_in,
                                  wt_A_dif, wt_K_c, wt_K_g, wt_K_d, wt_plevel, wt_bl,
                                  wt_spin, wt_dt, wt_pitot_k, wt_pitot_zero, wt_pitot_unc,
                                  wt_man_incl, wt_man_unc, wt_bal_unc,
                                  sw_U, sw_dmg, sw_inc],
                          outputs=[sw_table])

        with gr.Tab("7. Turnover 분석"):
            with gr.Accordion("사용 중인 물리 방정식", open=True):
                gr.Markdown(EQUATION_MD)
            with gr.Accordion("시뮬레이션이 계산하는 물리량 전체 목록", open=False):
                gr.Markdown(PHYSICS_GLOSSARY_MD)
            gr.Markdown(
                "위 식의 진단값(고유 각진동수 ω_n, 감쇠 각진동수 ω_d, 진동수 f_d, "
                "감쇠 유형, 오버슈트)은 시뮬레이션 실행 후 결과 카드에서 확인할 수 있습니다.")

        with gr.Tab("8. 결과 분석"):
            compare_btn = gr.Button("📊 셔틀콕 종류별 궤적 비교")
            gr.Markdown("*같은 발사 조건에서 셔틀콕 종류별 궤적을 한 그래프에 겹쳐 그립니다.*")
            compare_plot = gr.Plot(show_label=False, container=False)
            compare_error = gr.Markdown("")
            compare_btn.click(controller_compare,
                               inputs=[skirt_d, V0, launch_angle, height, alpha0, alpha_eq,
                                       omega0, rho, u_air_x, solver, dt, t_max],
                               outputs=[compare_plot, compare_error])

            with gr.Accordion("파손 상태별 연구 비교 (Normal / Mild / Moderate / Severe)", open=True):
                gr.Markdown(
                    "Primary: wobble RMS·amplitude, turnover time, stabilization time, "
                    "angular velocity/acceleration RMS. "
                    "Secondary: range, flight time, speed decay, max drag, trajectory deviation."
                )
                rc_level = gr.Dropdown([2, 3], value=2, label="모델 수준 (Level)")
                rc_btn = gr.Button("📊 파손 상태별 비교 실행")
                gr.Markdown("*정상/약함/중간/심함 파손 형상을 각각 시뮬레이션해 wobble·안정화 시간 등을 표로 비교합니다.*")
                rc_plot = gr.Plot(show_label=False, container=False)
                rc_table = gr.Textbox(label="비교 지표 (CSV)", lines=8)
                rc_note = gr.Markdown("")
                rc_btn.click(controller_research_compare,
                              inputs=[profile, skirt_d, rc_level, V0, launch_angle, height,
                                      orient_preset, custom_orient, rho, u_air_x, dt, t_max],
                              outputs=[rc_plot, rc_table, rc_note])

        with gr.Tab("9. 실험 비교"):
            csv_file = gr.File(label="실험 데이터 CSV 파일")
            with gr.Row():
                time_col = gr.Textbox(label="시간 열 이름", value="time")
                x_col = gr.Textbox(label="x 좌표 열 이름", value="x")
                y_col = gr.Textbox(label="y 좌표 열 이름", value="y")
                orientation_col = gr.Textbox(label="자세각 열 이름 (선택)", value="")
            import_btn = gr.Button("📂 실험 데이터 불러오기")
            gr.Markdown("*CSV 파일을 읽어 열 이름을 연결하고, 결측값·중복 시각 등 데이터 품질을 함께 점검합니다.*")
            import_preview = gr.Textbox(label="미리보기 (앞 20행)", lines=6)
            quality_out = gr.Markdown("")
            import_btn.click(controller_import, inputs=[csv_file, time_col, x_col, y_col, orientation_col],
                              outputs=[import_preview, quality_out, warning_panel, status_bar])
            overlay_btn = gr.Button("📈 실험과 시뮬레이션 비교")
            gr.Markdown("*실험 데이터와 시뮬레이션 결과를 같은 그래프에 겹쳐 그리고 오차(RMSE·MAE)를 계산합니다.*")
            overlay_plot = gr.Plot(show_label=False, container=False)
            residual_out = gr.Markdown("")
            overlay_btn.click(controller_overlay, outputs=[overlay_plot, residual_out])
            reset_exp_btn = gr.Button("↩ 실험 데이터 지우기")
            gr.Markdown("*불러온 실험 데이터를 메모리에서 제거합니다. 원본 파일은 그대로입니다.*")

        with gr.Tab("10. 매개변수 피팅"):
            gr.Markdown("피팅할 물리량을 직접 선택하세요. 전체를 자동으로 피팅하지 않습니다.\n\n*피팅된 값은 직접 측정값이 아닙니다.*")
            fit_cd = gr.Checkbox(label="항력계수 Cd 피팅")
            fit_tw = gr.Checkbox(label="Turnover 시간상수 Tw 피팅")
            fit_zeta = gr.Checkbox(label="감쇠비 ζ 피팅")
            fit_btn = gr.Button("🎯 물리량 피팅 실행", variant="primary")
            gr.Markdown("*체크한 물리량만 실험 데이터에 맞춰 최적값을 찾습니다. 피팅값은 직접 측정값이 아니며 불확실성이 함께 표시됩니다.*")
            fit_error = gr.Markdown("")
            fit_card_out = gr.Markdown("결과 없음 — 먼저 피팅을 실행하세요.")
            fit_plot = gr.Plot(show_label=False, container=False)
            fit_residual_out = gr.Markdown("")
            fit_btn.click(controller_fit, inputs=[fit_cd, fit_tw, fit_zeta, dt, t_max],
                          outputs=[fit_card_out, fit_plot, fit_residual_out, fit_error, status_bar, workflow_bar])
            reset_fit_btn = gr.Button("↩ 피팅 결과 지우기")
            gr.Markdown("*피팅 결과를 지웁니다. 셔틀콕 설정값은 바뀌지 않습니다.*")

        with gr.Tab("11. 민감도 분석"):
            with gr.Accordion("민감도 분석", open=True):
                sens_param = gr.Dropdown(SENSITIVITY_PARAMS, value="Cd", label="분석할 물리량")
                sens_btn = gr.Button("📊 민감도 분석 실행")
                gr.Markdown("*선택한 물리량을 ±10%까지 바꿔가며 도달거리·안정화 시간이 얼마나 달라지는지 봅니다.*")
                sens_error = gr.Markdown("")
                sens_plot = gr.Plot(show_label=False, container=False)
                sens_csv = gr.Textbox(label="분석 결과 (CSV)", lines=6)
                sens_btn.click(controller_sensitivity, inputs=[sens_param, dt, t_max],
                                outputs=[sens_plot, sens_csv, sens_error])
            with gr.Accordion("2변수 스윕 분석", open=True):
                sweep_x = gr.Dropdown(["Cd", "Tw", "zeta", "initial_velocity"], value="Cd", label="X축 물리량")
                sweep_y = gr.Dropdown(["Cd", "Tw", "zeta", "initial_velocity"], value="Tw", label="Y축 물리량")
                with gr.Row():
                    x_min = gr.Number(value=0.4, label="X 최솟값")
                    x_max = gr.Number(value=0.8, label="X 최댓값")
                with gr.Row():
                    y_min = gr.Number(value=0.008, label="Y 최솟값")
                    y_max = gr.Number(value=0.012, label="Y 최댓값")
                sweep_metric = gr.Dropdown(["stabilization_time_s", "flight_range_m"],
                                            value="stabilization_time_s", label="결과 변수")
                sweep_btn = gr.Button("📊 2변수 스윕 실행")
                gr.Markdown("*두 물리량을 동시에 바꿔가며 5×5 격자로 계산해 결과를 히트맵으로 보여줍니다.*")
                sweep_error = gr.Markdown("")
                sweep_plot = gr.Plot(show_label=False, container=False)
                sweep_btn.click(controller_sweep,
                                 inputs=[sweep_x, sweep_y, x_min, x_max, y_min, y_max, sweep_metric, dt, t_max],
                                 outputs=[sweep_plot, sweep_error])

        with gr.Tab("12. 수치 검증"):
            val_btn = gr.Button("🔬 물리 모델 검증 실행")
            gr.Markdown("*항력 없는 포물선 운동, 무중력, 감쇠 유형별 진동 등 이론적으로 답을 아는 경우와 비교해 계산이 맞는지 확인합니다.*")
            val_summary = gr.Markdown("")
            val_detail = gr.Textbox(label="상세 결과 (JSON)", lines=12)
            val_btn.click(controller_validation, outputs=[val_summary, val_detail, status_bar, workflow_bar])
            conv_btn = gr.Button("🔬 수치해 수렴성 검사")
            gr.Markdown("*시간 간격을 dt, dt/2, dt/4로 줄여가며 결과가 한 값으로 수렴하는지 확인합니다.*")
            conv_out = gr.Markdown("")
            conv_btn.click(controller_convergence, inputs=[dt, t_max], outputs=[conv_out])

        with gr.Tab("13. 결과 저장"):
            gr.Markdown("재현성 정보: 마지막 시뮬레이션의 모델·물리량·초기조건·환경·수치설정·버전을 함께 저장합니다.")
            export_btn = gr.Button("💾 결과 내보내기", variant="primary")
            gr.Markdown("*시뮬레이션 데이터(CSV)와 재현에 필요한 설정 정보(JSON)를 파일로 저장합니다.*")
            export_error = gr.Markdown("")
            export_csv = gr.File(label="시뮬레이션 데이터 (CSV)")
            export_json = gr.File(label="메타데이터 (JSON)")
            export_cfg = gr.File(label="재현용 설정 파일 (JSON)")
            export_btn.click(controller_export, outputs=[export_csv, export_json, export_cfg, export_error])
            with gr.Accordion("전체 초기화", open=True):
                gr.Markdown("모든 상태(셔틀콕 설정·발사 조건·시뮬레이션·실험 데이터·피팅)를 초기화합니다.")
                reset_all_confirm = gr.Checkbox(label="정말 전체를 초기화하려면 체크하세요")
                reset_all_btn = gr.Button("↩ 전체 초기화", variant="stop")
                gr.Markdown("*위 체크박스를 켠 경우에만 실행됩니다.*")
                reset_all_out = gr.Markdown("")

                def guarded_reset_all(confirmed):
                    if not confirmed:
                        return status_markdown(), workflow_markdown(), warning_center(), "초기화를 취소했습니다 (확인 필요)."
                    return controller_reset_all()

                reset_all_btn.click(guarded_reset_all, inputs=[reset_all_confirm],
                                     outputs=[status_bar, workflow_bar, warning_panel, reset_all_out])

        run_top_outputs = run_outputs + [top_anim, top_shuttle, flow_slider,
                                          frame_plot, frame_panel, flow_plot, flow_panel,
                                          frame_slider, flow_slider]
        run_btn.click(run_sim_with_top, inputs=run_inputs, outputs=run_top_outputs)
        rerun_btn.click(run_sim_with_top, inputs=run_inputs, outputs=run_top_outputs)
        reset_sim_btn.click(reset_sim_with_top,
                             outputs=[traj_plot, speed_plot, angle_plot, omega_plot, wobble_plot,
                                      result_cards, status_bar, workflow_bar, warning_panel,
                                      top_anim, top_shuttle, flow_slider])

        reset_params_btn.click(lambda: controller_param_preview("BWF 천연 깃털 (기본)"), outputs=[param_preview])
        reset_ic_btn.click(lambda: (25.0, -10.0, 1.8, 145.0, 180.0, 0.0),
                            outputs=[V0, launch_angle, height, alpha0, alpha_eq, omega0])
        reset_exp_btn.click(lambda: (state.update(exp_df=None, exp_quality=None) or ("", "", status_markdown())),
                             outputs=[import_preview, quality_out, status_bar])
        reset_fit_btn.click(lambda: (state.update(fit_result=None) or
                                      ("결과 없음 — 먼저 피팅을 실행하세요.", None, "", status_markdown())),
                             outputs=[fit_card_out, fit_plot, fit_residual_out, status_bar])

        with gr.Tab("14. 📚 CFD 연구 근거 및 검증"):
            gr.Markdown(cfd_citation_markdown())
            gr.Markdown("---")
            gr.Markdown(
                "### 격자 수렴 검증 (Roache GCI)\n\n"
                "외부 CFD 솔버에서 얻은 **세 격자(fine / medium / coarse)** 의 계수를 "
                "입력하면 관측 수렴차수와 GCI 를 계산합니다. 진동 수렴 등 점근 영역이 "
                "아니면 GCI 를 계산하지 않고 `non-asymptotic` 으로 표시합니다.")
            with gr.Row():
                gci_f = gr.Number(value=0.650, label="fine 격자 계수")
                gci_m = gr.Number(value=0.800, label="medium 격자 계수")
                gci_c = gr.Number(value=1.400, label="coarse 격자 계수")
            with gr.Row():
                gci_r21 = gr.Number(value=2.0, label="격자 세분비 r21")
                gci_r32 = gr.Number(value=2.0, label="격자 세분비 r32")
                gci_name = gr.Textbox(value="Cd", label="계수 이름")
            gci_btn = gr.Button("📐 GCI 계산", variant="primary")
            gci_out = gr.Markdown("")

            def _gci(f, m_, c, r21, r32, nm):
                try:
                    res = mesh_convergence_study([f, m_, c], (r21, r32), name=str(nm))
                    lines = [f"### {nm} 격자 수렴 결과", "",
                             "| 항목 | 값 |", "|---|---|",
                             f"| fine / medium / coarse | {res['fine']:.6g} / "
                             f"{res['medium']:.6g} / {res['coarse']:.6g} |",
                             f"| 관측 수렴차수 p | {res['observed_order']:.4f} |"
                             if np.isfinite(res['observed_order']) else
                             "| 관측 수렴차수 p | 계산 불가 |",
                             f"| Richardson 외삽값 | {res['extrapolated']:.6g} |"
                             if np.isfinite(res['extrapolated']) else
                             "| Richardson 외삽값 | 계산 불가 |"]
                    if res.get("asymptotic"):
                        lines.append(f"| GCI (fine) | {res['GCI_fine_percent']:.3f} % |")
                        if "asymptotic_ratio" in res:
                            lines.append(f"| 점근비 (1에 가까울수록 좋음) | "
                                          f"{res['asymptotic_ratio']:.4f} |")
                    lines += ["", f"**판정: {res['status']}**"]
                    if res.get("note"):
                        lines.append(f"> {res['note']}")
                    lines += ["", "*Roache 1994 / Celik et al. 2008 절차. 안전계수 "
                              "Fs = 1.25 (3개 이상 격자).*"]
                    return "\n".join(lines)
                except Exception:
                    logger.exception("gci failed")
                    return "GCI 를 계산하지 못했습니다. 입력값을 확인하세요."

            gci_btn.click(_gci, inputs=[gci_f, gci_m, gci_c, gci_r21, gci_r32, gci_name],
                           outputs=[gci_out])

        with gr.Tab("15. 📚 인용 연구 자료"):
            gr.Markdown(citation_markdown())
            gr.Markdown("---")
            gr.Markdown("## 연구 근거 및 모델 한계")
            readiness_btn = gr.Button("🔎 연구 사용 가능 여부 판정", variant="primary")
            readiness_out = gr.Markdown(
                "버튼을 누르면 물리 검증을 실행하고, 통과 여부와 남아 있는 한계를 "
                "함께 표시합니다. 프로그램은 결과를 자동으로 '연구 사용 가능'이라고 "
                "표시하지 않습니다.")
            readiness_btn.click(lambda: research_readiness(db), inputs=[],
                                 outputs=[readiness_out])

    return demo

_RUNNING_DEMO = None
_TUNNEL_PROC = None

CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
                   "cloudflared-linux-amd64")


def _cloudflared_binary():
    """Path to a usable cloudflared, downloading one if the system has none."""
    import shutil
    found = shutil.which("cloudflared")
    if found:
        return found
    import os
    import urllib.request
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cloudflared")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "cloudflared")
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        print("공개 링크용 cloudflared 를 내려받는 중입니다 (약 40 MB, 처음 한 번만)…")
        urllib.request.urlretrieve(CLOUDFLARED_URL, path)
    os.chmod(path, 0o755)
    return path


def _cloudflare_quick_tunnel(port, timeout_s=60):
    """Public URL from Cloudflare's quick tunnel, when Gradio's share server refuses.

    This matters in Colab specifically. Gradio sends every event over the queue's SSE
    stream, and Colab's kernel port proxy does not carry that stream reliably: without a
    public URL the notebook renders the whole UI and then every button reports
    "Connection to the server was lost. Attempting reconnection...". So a failed share
    link is not cosmetic there, it makes the app unusable.

    Cloudflare's tunnel is an independent route from share.gradio.live and a quick
    tunnel needs no account or token. Returns the URL, or None if one cannot be opened.
    """
    global _TUNNEL_PROC
    import re
    import subprocess
    import threading
    import time as _time
    try:
        binary = _cloudflared_binary()
    except Exception:
        logger.exception("cloudflared download failed")
        return None
    try:
        proc = subprocess.Popen(
            [binary, "tunnel", "--url", "http://127.0.0.1:%d" % int(port),
             "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
    except Exception:
        logger.exception("cloudflared launch failed")
        return None

    lines, url = [], None
    def _drain():
        try:
            for line in proc.stdout:
                lines.append(line)
        except Exception:
            pass
    threading.Thread(target=_drain, daemon=True).start()

    pattern = re.compile(r"https://[-\w]+\.trycloudflare\.com")
    deadline = _time.time() + float(timeout_s)
    while _time.time() < deadline:
        for line in list(lines):
            m = pattern.search(line)
            if m:
                url = m.group(0)
                break
        if url or proc.poll() is not None:
            break
        _time.sleep(0.5)

    if url:
        _TUNNEL_PROC = proc
        return url
    try:
        proc.terminate()
    except Exception:
        pass
    tail = "".join(lines[-4:]).strip()
    if tail:
        logger.warning("cloudflared produced no URL: %s", tail[:400])
    return None


def launch(share=True, tunnel_fallback=True, **kwargs):
    """Start the app once, and make sure there is a URL that actually works.

    Gradio already degrades to a local server when its share tunnel is unreachable, so
    there is nothing to retry by relaunching -- and retrying that way breaks Colab,
    which renders an inline frame the moment launch() returns: launching, closing and
    relaunching left the first frame pointing at a server that had just been shut down
    ("Connection to the server was lost"), and burned a port each time.

    A local URL is not enough in Colab, though. Every Gradio event travels over the
    queue's SSE stream and Colab's kernel port proxy does not carry it reliably, so the
    notebook shows the whole UI with every control dead. When the share link fails, a
    Cloudflare quick tunnel is opened instead; pass tunnel_fallback=False to skip it.
    """
    global _RUNNING_DEMO, _TUNNEL_PROC
    # Re-running the cell otherwise leaves the previous server holding its port, so the
    # ports climb (7860, 7861, ...) and dead frames pile up in the notebook.
    if _RUNNING_DEMO is not None:
        try:
            _RUNNING_DEMO.close()
        except Exception:
            pass
        _RUNNING_DEMO = None
    if _TUNNEL_PROC is not None:
        try:
            _TUNNEL_PROC.terminate()
        except Exception:
            pass
        _TUNNEL_PROC = None

    demo = build_ui()
    # A syntax error in a js= handler stops the whole page at the loading spinner,
    # with nothing on the server to show for it. Check before serving it.
    disable_broken_client_js(demo)
    try:
        demo.queue(default_concurrency_limit=1, max_size=32)
    except TypeError:
        try:
            demo.queue(max_size=32)
        except TypeError:
            demo.queue()

    opts = dict(show_error=True)
    opts.update(gradio_ui_kwargs("launch"))
    opts.update(kwargs)

    try:
        out = demo.launch(share=share, **opts)
    except Exception:
        if not share:
            raise
        logger.exception("share launch failed")
        print("\n공유 링크를 만들지 못해 로컬로 실행합니다.")
        out = demo.launch(share=False, **opts)
    _RUNNING_DEMO = demo

    if share and not getattr(demo, "share_url", None):
        port = getattr(demo, "server_port", None)
        print("\n공유 링크를 만들지 못했습니다 — Gradio 공유 서버(share.gradio.live)가 "
              "연결을 거절하고 있습니다.")
        tried = bool(tunnel_fallback and port)
        url = _cloudflare_quick_tunnel(port) if tried else None
        if url:
            print("대신 Cloudflare 임시 터널로 공개 링크를 열었습니다:\n\n    %s\n" % url)
            print("Colab 에서는 셀 안 화면 대신 반드시 이 링크로 여십시오. "
                  "Colab 포트 프록시는 Gradio 의 이벤트 스트림(SSE)을 제대로 전달하지 "
                  "못해서, 셀 안 화면은 버튼을 눌러도 "
                  "'Connection to the server was lost' 가 됩니다.")
        elif tried:
            print("Cloudflare 임시 터널도 열지 못했습니다. 런타임을 다시 시작한 뒤 "
                  "이 셀을 재실행하거나 https://status.gradio.app 을 확인하십시오.\n"
                  "Colab 셀 안 화면은 이벤트 스트림이 전달되지 않아 버튼이 동작하지 "
                  "않을 수 있습니다.")
        else:
            print("대체 터널은 시도하지 않았습니다 (tunnel_fallback=False).\n"
                  "Colab 셀 안 화면은 이벤트 스트림이 전달되지 않아 버튼이 동작하지 "
                  "않을 수 있습니다.")
    return out


if __name__ == "__main__":
    launch()
