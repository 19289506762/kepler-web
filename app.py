import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import io
import base64
import csv
from datetime import datetime

st.set_page_config(page_title="开普勒第二定律", layout="wide")

# ==================== 常量 ====================
AU = 1.0
EARTH_ECCENTRICITY = 0.0167
T = 1.0
DT = 0.001
STEPS_PER_FRAME = 10
AU_KM = 1.496e8
YEAR_SEC = 365.25 * 24 * 3600
AREA_SAMPLES_MIN = 50
AREA_SAMPLES_PER_UNIT = 200
BISECTION_MAX_ITER = 60
BISECTION_TOL = 1e-6

SOLAR_TERMS = {'春分': np.pi/2, '秋分': 3*np.pi/2, '夏至': np.pi, '冬至': 0}

# ==================== 核心函数 ====================
def solve_eccentric_anomaly(M, e, tol=1e-10):
    E = M
    while True:
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if abs(delta) < tol:
            break
    return E

def orbit_position(t, a, e):
    t_mod = t % T
    M = 2 * np.pi * t_mod / T
    E = solve_eccentric_anomaly(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)
    return x, y

def orbit_velocity(t, a, e):
    dt_small = 1e-6
    x1, y1 = orbit_position(t, a, e)
    x2, y2 = orbit_position(t + dt_small, a, e)
    return np.hypot(x2 - x1, y2 - y1) / dt_small

def compute_avg_speed(t_start, t_end, a, e, num_samples=50):
    if t_start == t_end:
        return 0.0
    if t_end < t_start:
        t_end += T
    times = np.linspace(t_start, t_end, num_samples)
    speeds = [orbit_velocity(ti, a, e) for ti in times]
    return np.mean(speeds)

def compute_sector_area(t_start, t_end, a, e):
    if t_start == t_end:
        return 0.0
    t_start_adj = t_start
    t_end_adj = t_end
    if t_end_adj < t_start_adj:
        t_end_adj += T
    interval = t_end_adj - t_start_adj
    num_samples = max(AREA_SAMPLES_MIN, int(interval * AREA_SAMPLES_PER_UNIT))
    num_samples = min(num_samples, 500)
    times = np.linspace(t_start_adj, t_end_adj, num_samples)
    area = 0.0
    theta_prev = None
    for ti in times:
        x, y = orbit_position(ti, a, e)
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        if theta_prev is not None:
            dtheta = theta - theta_prev
            if dtheta > np.pi:
                dtheta -= 2 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2 * np.pi
            area += 0.5 * r**2 * abs(dtheta)
        theta_prev = theta
    return area

# ==================== 初始化 Session State ====================
if "t" not in st.session_state:
    st.session_state.t = 0.0
    st.session_state.paused = False
    st.session_state.mark_times = []
    st.session_state.segment_visible = []
    st.session_state.segment_avg_speed = []
    st.session_state.ref_area = None
    st.session_state.speed_factor = 1.0
    st.session_state.ecc = EARTH_ECCENTRICITY

# ==================== 侧边栏控件 ====================
with st.sidebar:
    st.header("控制面板")
    ecc = st.slider("偏心率 e", 0.0, 0.5, st.session_state.ecc, 0.001)
    st.session_state.ecc = ecc
    speed_factor = st.slider("动画速度倍率", 0.1, 10.0, st.session_state.speed_factor, 0.1)
    st.session_state.speed_factor = speed_factor
    auto_pause = st.checkbox("标记后自动暂停", value=False)
    show_axes = st.checkbox("显示长短轴", value=True)
    show_terms = st.checkbox("显示节气", value=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏸️ 暂停"):
            st.session_state.paused = True
        if st.button("↺ 重置时间"):
            st.session_state.t = 0.0
            st.session_state.mark_times = []
            st.session_state.segment_visible = []
            st.session_state.segment_avg_speed = []
            st.session_state.ref_area = None
        if st.button("◀ 后退"):
            st.session_state.paused = True
            st.session_state.t = max(0.0, st.session_state.t - DT * STEPS_PER_FRAME)
    with col2:
        if st.button("▶️ 继续"):
            st.session_state.paused = False
        if st.button("🗑️ 重置标记"):
            st.session_state.mark_times = []
            st.session_state.segment_visible = []
            st.session_state.segment_avg_speed = []
            st.session_state.ref_area = None
        if st.button("前进 ▶"):
            st.session_state.paused = True
            st.session_state.t += DT * STEPS_PER_FRAME
            if st.session_state.t > T:
                st.session_state.t -= T

    if st.button("📍 添加标记", use_container_width=True):
        st.session_state.mark_times.append(st.session_state.t)
        n = len(st.session_state.mark_times)
        if n >= 2:
            if n == 2:
                st.session_state.segment_visible.append(True)
                st.session_state.ref_area = compute_sector_area(
                    st.session_state.mark_times[0], st.session_state.mark_times[1], AU, ecc
                )
            else:
                st.session_state.segment_visible.append(False)
            st.session_state.segment_avg_speed.append(None)
        if auto_pause and not st.session_state.paused:
            st.session_state.paused = True

    if st.button("⚖️ 等面积点", use_container_width=True):
        if len(st.session_state.mark_times) < 2:
            st.warning("至少需要两个标记点")
        else:
            if st.session_state.ref_area is None:
                st.session_state.ref_area = compute_sector_area(
                    st.session_state.mark_times[0], st.session_state.mark_times[1], AU, ecc
                )
            t_start = st.session_state.mark_times[-1]
            left, right = t_start, t_start + T
            if compute_sector_area(t_start, right, AU, ecc) < st.session_state.ref_area:
                st.error("参考面积大于一圈面积")
            else:
                for _ in range(BISECTION_MAX_ITER):
                    mid = (left + right) / 2
                    area_mid = compute_sector_area(t_start, mid, AU, ecc)
                    if abs(area_mid - st.session_state.ref_area) < BISECTION_TOL:
                        break
                    if area_mid < st.session_state.ref_area:
                        left = mid
                    else:
                        right = mid
                t_end = (left + right) / 2
                st.session_state.mark_times.append(t_end)
                st.session_state.segment_visible.append(True)
                st.session_state.segment_avg_speed.append(None)

    if st.button("📁 导出标记数据", use_container_width=True):
        if len(st.session_state.mark_times) < 2:
            st.warning("没有足够的标记点")
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['序号', '时刻(年)', 'x(AU)', 'y(AU)', '距离(AU)', '等面积段起点', '平均速度(km/s)'])
            for i, t in enumerate(st.session_state.mark_times):
                x, y = orbit_position(t, AU, ecc)
                r = np.hypot(x, y)
                is_equal = (i < len(st.session_state.segment_visible) and st.session_state.segment_visible[i]) if i > 0 else False
                avg_speed = ''
                if i > 0 and i - 1 < len(st.session_state.segment_avg_speed):
                    v = st.session_state.segment_avg_speed[i - 1]
                    if v:
                        avg_speed = f"{v:.2f}"
                writer.writerow([i + 1, f"{t:.6f}", f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", is_equal, avg_speed])
            csv_data = output.getvalue()
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="kepler_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">点击下载CSV</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**实时信息**")
    if st.session_state.ref_area:
        st.info(f"参考面积: {st.session_state.ref_area:.4f} AU²")
    speed_ratio = np.sqrt((1 + ecc) / (1 - ecc))
    st.info(f"近日/远日速度比: {speed_ratio:.3f}")
    if len(st.session_state.mark_times) > 1:
        areas = []
        for i in range(len(st.session_state.mark_times) - 1):
            if st.session_state.segment_visible[i]:
                area = compute_sector_area(st.session_state.mark_times[i], st.session_state.mark_times[i + 1], AU, ecc)
                areas.append(f"段{i + 1}: {area:.4f} AU²")
        if areas:
            st.write(" | ".join(areas))
        if st.session_state.ref_area and st.session_state.ref_area > 0:
            errors = []
            for i in range(len(st.session_state.mark_times) - 1):
                if st.session_state.segment_visible[i]:
                    area = compute_sector_area(st.session_state.mark_times[i], st.session_state.mark_times[i + 1], AU, ecc)
                    errors.append(abs(area - st.session_state.ref_area) / st.session_state.ref_area * 100)
            if errors:
                avg_err = np.mean(errors)
                st.write(f"平均误差: {avg_err:.2f}%")
    else:
        st.write("暂无面积段")

    st.markdown("**标记点列表**")
    marks_text = ""
    for i, t in enumerate(st.session_state.mark_times):
        x, y = orbit_position(t, AU, ecc)
        r = np.hypot(x, y)
        line = f"{i + 1}. t={t:.3f}, r={r:.3f}AU"
        if i > 0 and i - 1 < len(st.session_state.segment_avg_speed):
            v = st.session_state.segment_avg_speed[i - 1]
            if v:
                line += f", v_avg={v:.1f}km/s"
        marks_text += line + "\n"
    st.text_area("", marks_text, height=200)

# ==================== 更新时间 ====================
if not st.session_state.paused:
    st.session_state.t += DT * STEPS_PER_FRAME * st.session_state.speed_factor
    if st.session_state.t > T:
        st.session_state.t -= T

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')

# 星星背景
np.random.seed(42)
stars_x = np.random.uniform(-1.5, 1.5, 150)
stars_y = np.random.uniform(-1.5, 1.5, 150)
ax.scatter(stars_x, stars_y, c='white', s=1.5, alpha=0.6)

ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.15, color='gray')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x (AU)', color='white')
ax.set_ylabel('y (AU)', color='white')
ax.set_title('开普勒第二定律：等面积验证', color='cyan', fontsize=16)
ax.tick_params(colors='white')

# 太阳
try:
    sun_img = np.array(Image.open("sun.png"))
    ax.add_artist(AnnotationBbox(OffsetImage(sun_img, zoom=0.06), (0, 0), frameon=False))
except:
    ax.plot(0, 0, 'yo', markersize=20, markeredgecolor='orange')

# 轨道
theta = np.linspace(0, 2 * np.pi, 500)
xo = AU * (np.cos(theta) - ecc)
yo = AU * np.sqrt(1 - ecc**2) * np.sin(theta)
ax.plot(xo, yo, color='cyan', lw=5, alpha=0.2)
ax.plot(xo, yo, color='cyan', lw=2.5)

# 另一个焦点
other_x = -2 * AU * ecc
ax.plot(other_x, 0, 'ro', markersize=8, markeredgecolor='white')
ax.text(other_x, 0.1, '另一个焦点', color='white', ha='center', fontsize=9,
        bbox=dict(facecolor='black', alpha=0.5))

# 椭圆中心
center_x = -AU * ecc
ax.plot(center_x, 0, 'o', color='gray', markersize=5, alpha=0.7)

# 长短轴
peri = AU * (1 - ecc)
aph = -AU * (1 + ecc)
b = AU * np.sqrt(1 - ecc**2)
if show_axes:
    ax.plot([aph, peri], [0, 0], 'w--', lw=1.5, alpha=0.7)
    ax.plot([center_x, center_x], [-b, b], 'w--', lw=1.5, alpha=0.7)
    ax.text(center_x + AU / 2, 0.1, f'a = {AU:.2f}', color='white', ha='center',
            bbox=dict(facecolor='black', alpha=0.5))
    ax.text(center_x + 0.1, b / 2, f'b = {b:.2f}', color='white', ha='left',
            bbox=dict(facecolor='black', alpha=0.5))

# 行星
x_planet, y_planet = orbit_position(st.session_state.t, AU, ecc)
try:
    earth_img = np.array(Image.open("earth.png"))
    earth_ab = AnnotationBbox(OffsetImage(earth_img, zoom=0.045), (x_planet, y_planet), frameon=False)
    ax.add_artist(earth_ab)
    glow = Circle((x_planet, y_planet), radius=0.12, color='deepskyblue', alpha=0.3, zorder=1)
    ax.add_patch(glow)
except:
    ax.plot(x_planet, y_planet, 'o', color='deepskyblue', markersize=14, markeredgecolor='white')

# 日-行连线
ax.plot([0, x_planet], [0, y_planet], 'w--', lw=1.2, alpha=0.8)

# 节气
for name, f in SOLAR_TERMS.items():
    r = AU * (1 - ecc**2) / (1 + ecc * np.cos(f))
    xt = r * np.cos(f)
    yt = r * np.sin(f)
    if show_terms:
        ax.plot(xt, yt, '^', color='orange', markersize=10, markeredgecolor='white')
        ax.text(xt + 0.1, yt + 0.1, name, color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5))

# 标记点和面积填充
colors_fill = ['lime', 'orange', 'violet', 'cyan', 'magenta', 'gold']
for i, t in enumerate(st.session_state.mark_times):
    xm, ym = orbit_position(t, AU, ecc)
    ax.plot(xm, ym, 'bo', markersize=6, markeredgecolor='white')
for i in range(len(st.session_state.mark_times) - 1):
    if not st.session_state.segment_visible[i]:
        continue
    t1, t2 = st.session_state.mark_times[i], st.session_state.mark_times[i + 1]
    area_val = compute_sector_area(t1, t2, AU, ecc)
    t1_adj = t1
    t2_adj = t2
    if t2_adj < t1_adj:
        t2_adj += T
    times = np.linspace(t1_adj, t2_adj, 100)
    xs = [0] + [orbit_position(ti, AU, ecc)[0] for ti in times] + [0]
    ys = [0] + [orbit_position(ti, AU, ecc)[1] for ti in times] + [0]
    ax.fill(xs, ys, alpha=0.4, color=colors_fill[i % len(colors_fill)], edgecolor='white', linewidth=0.5)
    t_mid = (t1_adj + t2_adj) / 2
    xm, ym = orbit_position(t_mid, AU, ecc)
    rm = np.hypot(xm, ym)
    if rm < 0.1:
        xm, ym = (0.2, 0) if rm == 0 else (xm / rm * 0.2, ym / rm * 0.2)
    ax.text(xm, ym, f"{area_val:.3f}", color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# 计算并存储平均速度
for i in range(len(st.session_state.mark_times) - 1):
    if st.session_state.segment_visible[i] and (len(st.session_state.segment_avg_speed) <= i or st.session_state.segment_avg_speed[i] is None):
        t1, t2 = st.session_state.mark_times[i], st.session_state.mark_times[i + 1]
        avg_au = compute_avg_speed(t1, t2, AU, ecc)
        st.session_state.segment_avg_speed[i] = avg_au * (AU_KM / YEAR_SEC)

# 信息文本
r = np.hypot(x_planet, y_planet)
if abs(r - AU * (1 - ecc)) < 0.01:
    pos = '近日点'
elif abs(r - AU * (1 + ecc)) < 0.01:
    pos = '远日点'
else:
    pos = ''
ax.text(0.02, 0.95, f"时间: {st.session_state.t:.3f}年\n距离: {r:.3f}AU\n{pos}",
        transform=ax.transAxes, color='white', fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

st.pyplot(fig)
plt.close(fig)
