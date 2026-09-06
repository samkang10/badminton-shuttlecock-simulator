"""검증용 합성 셔틀콕 영상 생성기.

AI Physics Tracker 의 셔틀콕 추적을 시험하려면 '정답을 아는 영상'이 필요하다. 실제
실험 영상에는 사람이 찍은 기준 궤적이 있어야 오차를 말할 수 있는데, 그 기준 자체가
사람 손으로 만든 것이라 오차를 포함한다. 여기서 만드는 영상은 궤적을 수식으로 먼저
정하고 그 자리에 셔틀콕을 그리므로, 추적 결과와 비교할 기준 궤적이 오차 없이 존재한다.

그리는 물체는 실제 셔틀콕의 구조를 그대로 따른다.
  - 코르크 헤드 : 밝고 속이 찬 원
  - 깃털 스커트 : 코르크 반대쪽으로 벌어지는 원뿔(깃대 여러 개 + 반투명 막)
  - 손상 조건   : 깃대를 대칭/비대칭으로 빼서 2개·4개 손상 셔틀콕을 만든다

운동은 항력이 있는 포물선(2차원)이며, 셔틀콕의 장축은 항상 속도 방향을 향한다
(실제 셔틀콕이 코르크를 앞세우고 나는 성질). 여기에 촬영 조건을 얹는다.
  - motion blur : 프레임 노출 시간 동안의 위치를 여러 장 겹쳐 평균낸다(실제 블러와 같은 방식)
  - 크기 감소   : 카메라에서 멀어지는 효과
  - 배경        : 코트 비슷한 무늬 + 잡음
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _rotate(vx: float, vy: float, ang: float) -> Tuple[float, float]:
    c, s = np.cos(ang), np.sin(ang)
    return vx * c - vy * s, vx * s + vy * c


FEATHER_PATTERNS: Dict[str, List[int]] = {
                                                                          
    "NORMAL": list(range(16)),
    "2_FEATHERS_SYMMETRIC": [i for i in range(16) if i not in (0, 8)],
    "2_FEATHERS_ASYMMETRIC": [i for i in range(16) if i not in (0, 1)],
    "4_FEATHERS_SYMMETRIC": [i for i in range(16) if i not in (0, 4, 8, 12)],
    "4_FEATHERS_ASYMMETRIC": [i for i in range(16) if i not in (0, 1, 2, 3)],
}


def draw_shuttlecock(canvas: np.ndarray, cx: float, cy: float, axis_deg: float,
                     scale: float, damage: str = "NORMAL",
                     brightness: float = 1.0) -> None:
    """코르크 중심이 (cx, cy) 가 되도록 셔틀콕 한 개를 그린다.

    중심 정의를 명확히 하기 위해 '코르크의 중심'을 기준점으로 삼는다. 추적기가 어떤
    중심 정의를 쓰든, 손상 조건에 상관없이 이 점 하나가 기준 궤적이 된다.
    """
    ang = np.radians(axis_deg)
                                                                              
    ux, uy = np.cos(ang), np.sin(ang)
    cork_r = max(1.6, 4.2 * scale)
    skirt_len = 11.0 * scale
    skirt_r = 6.0 * scale

    tip_x, tip_y = cx - ux * skirt_len, cy - uy * skirt_len
    keep = FEATHER_PATTERNS.get(damage, FEATHER_PATTERNS["NORMAL"])
    n_slots = 16

                                                                                  
                                                                                
                                                                              
    overlay = canvas.copy()
    vane_half = 2.0 * np.pi / n_slots * 0.42
    root_x = cx - ux * cork_r
    root_y = cy - uy * cork_r
    for i in keep:
        theta = 2.0 * np.pi * i / n_slots
        depth = np.cos(theta)
        pts = []
        for edge in (-vane_half, vane_half):
            lateral = np.sin(theta + edge) * skirt_r
            dx, dy = _rotate(-0.06 * skirt_len * depth, lateral, ang)
            pts.append((tip_x + dx, tip_y + dy))
                                                                       
        shade = int(np.clip(210 * brightness * (0.62 + 0.38 * (0.5 + 0.5 * depth)), 0, 255))
        poly = np.array([[root_x, root_y], list(pts[0]), list(pts[1])], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (shade, shade, shade), cv2.LINE_AA)
        cv2.polylines(overlay, [poly], True, (int(shade * 0.7),) * 3, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)

                                                              
    cork_shade = int(np.clip(248 * brightness, 0, 255))
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(cork_r)),
               (cork_shade, cork_shade, cork_shade), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(cork_r)),
               (int(cork_shade * 0.72),) * 3, 1, cv2.LINE_AA)


def _background(w: int, h: int, rng: np.random.Generator, level: int = 58) -> np.ndarray:
    """코트 비슷한 배경. level 을 올리면 셔틀콕(밝은 흰색)과 배경의 대비가 줄어든다."""
    bg = np.full((h, w, 3), int(level), dtype=np.uint8)
    _line = int(min(250, level + 12))
    for gx in range(0, w, 90):
        cv2.line(bg, (gx, 0), (gx, h), (_line, _line + 4, _line), 1)
    for gy in range(0, h, 90):
        cv2.line(bg, (0, gy), (w, gy), (_line, _line + 4, _line), 1)
    _court = int(min(255, level + 38))
    cv2.line(bg, (0, int(h * 0.82)), (w, int(h * 0.82)), (_court, _court + 4, _court), 2)
    noise = rng.normal(0.0, 4.0, bg.shape)
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def simulate(n_frames: int, fps: float, w: int, h: int,
             v0: float, angle_deg: float, drag: float,
             x0: float, y0: float) -> np.ndarray:
    """항력이 있는 2차원 포물선. 반환은 프레임당 (x, y) 픽셀 좌표.

    dv/dt = -k|v|v + g 를 작은 시간 간격으로 적분한다. 노출 시간 안의 중간 위치까지
    필요하므로 프레임 간격을 다시 20 등분해 계산하고, 그 전부를 돌려준다.
    """
    sub = 20
    dt = 1.0 / (fps * sub)
    g = 900.0
    vx = v0 * np.cos(np.radians(angle_deg))
    vy = -v0 * np.sin(np.radians(angle_deg))
    x, y = float(x0), float(y0)
    out = np.zeros((n_frames * sub, 2), dtype=float)
    for i in range(n_frames * sub):
        out[i] = (x, y)
        sp = float(np.hypot(vx, vy))
        ax = -drag * sp * vx
        ay = -drag * sp * vy + g
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
    return out


def render(path: str, n_frames: int = 90, fps: float = 120.0,
           w: int = 960, h: int = 540, damage: str = "NORMAL",
           blur_frames: int = 1, base_scale: float = 1.0,
           shrink: float = 0.0, v0: float = 1500.0, angle_deg: float = 34.0,
           drag: float = 0.00035, seed: int = 7,
           bg_level: int = 58, occlude: Tuple[int, int] = (0, 0)) -> Tuple[str, np.ndarray]:
    """영상을 쓰고 (경로, 기준 궤적) 을 돌려준다. 기준 궤적은 코르크 중심의 화소 좌표.

    bg_level 을 올리면 배경이 밝아져 흰 셔틀콕과 구분하기 어려워진다.
    occlude=(a, b) 는 프레임 a 부터 b 까지 셔틀콕을 가리는 기둥을 세운다(일시적 가림).
    기준 궤적은 가려진 구간에도 그대로 남는다 — 가려졌다고 물체가 사라진 것은 아니므로,
    추적기가 그 구간을 어떻게 처리하는지(예측/LOST) 확인할 수 있다.
    """
    rng = np.random.default_rng(seed)
    sub = 20
    path_sub = simulate(n_frames, fps, w, h, v0, angle_deg, drag, 60.0, h * 0.72)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    truth = np.full((n_frames, 2), np.nan, dtype=float)
    occ_a, occ_b = int(occlude[0]), int(occlude[1])
    try:
        for f in range(n_frames):
            bg = _background(w, h, rng, bg_level)
            k = f * sub
            cx, cy = path_sub[k]
            truth[f] = (cx, cy)
            scale = base_scale * (1.0 - shrink * f / max(1, n_frames - 1))
                                                                                 
                                                                                
                                                                                     
                                                                                     
                                                                                
            acc = bg.astype(np.float32)
            n_sub = max(1, int(blur_frames))
            layers = 0
            span = sub // 2
            for j in range(n_sub):
                offset = 0 if n_sub == 1 else int(round(-span / 2 + span * j / (n_sub - 1)))
                kk = int(np.clip(k + offset, 0, len(path_sub) - 2))
                sx, sy = path_sub[kk]
                nx, ny = path_sub[kk + 1]
                axis = float(np.degrees(np.arctan2(ny - sy, nx - sx)))
                layer = bg.copy()
                draw_shuttlecock(layer, sx, sy, axis, scale, damage)
                acc += layer.astype(np.float32)
                layers += 1
            frame = np.clip(acc / float(layers + 1), 0, 255).astype(np.uint8)
                                                                                  
            if occ_a <= f <= occ_b and occ_b > occ_a:
                bar_w = max(18, int(26 * base_scale))
                bx = int(round(cx))
                cv2.rectangle(frame, (bx - bar_w, 0), (bx + bar_w, h),
                              (int(bg_level * 0.7),) * 3, -1)
            writer.write(frame)
    finally:
        writer.release()
    return path, truth


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_shuttlecock.mp4"
    p, t = render(out)
    print(p, t.shape)
