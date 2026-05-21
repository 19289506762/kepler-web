# 开普勒第二定律：等面积验证程序（增强版 - 显示段平均速度）
# 功能：
#   - 标记点列表显示每段平均速度（km/s），单位转换为直观数值
#   - 保留所有原有功能：等面积点、长短轴、节气显示、导出数据等

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image

# ==================== 常量设置 ====================
AU = 1.0                     # 半长轴 (AU)
EARTH_ECCENTRICITY = 0.0167  # 地球偏心率
T = 1.0                      # 周期 (年)
DT = 0.001                   # 时间步长 (年)
STEPS_PER_FRAME = 10         # 每帧步数

# 天文常数（用于速度单位换算）
AU_KM = 1.496e8              # 1 AU = 1.496e8 km
YEAR_SEC = 365.25 * 24 * 3600  # 1年 ≈ 3.15576e7秒

# 计算参数
AREA_SAMPLES_BASE = 150       # 面积计算基础采样点数
AREA_SAMPLES_MIN = 50         # 最小采样点数
AREA_SAMPLES_PER_UNIT = 200   # 每单位时间采样点数（用于动态调整）

# 二分查找参数
BISECTION_MAX_ITER = 60
BISECTION_TOL = 1e-6

# 动画更新间隔（毫秒）
ANIMATION_INTERVAL = 60

# 节气数据（真近点角近似）
SOLAR_TERMS = {
    '春分': np.pi/2,
    '秋分': 3*np.pi/2,
    '夏至': np.pi,
    '冬至': 0
}

# 尝试加载地球图片
earth_img = None
if os.path.exists("earth.png"):
    try:
        img = Image.open("earth.png")
        earth_img = np.array(img)
        print("已加载地球图片")
    except Exception as e:
        print(f"earth.png 加载失败: {e}，将使用蓝色圆点")
else:
    print("未找到 earth.png，将使用蓝色圆点")

# 尝试加载太阳图片
sun_img = None
if os.path.exists("sun.png"):
    try:
        img = Image.open("sun.png")
        sun_img = np.array(img)
        print("已加载太阳图片")
    except Exception as e:
        print(f"sun.png 加载失败: {e}，将使用黄色圆点")
else:
    print("未找到 sun.png，将使用黄色圆点")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 核心计算函数 ====================
def solve_eccentric_anomaly(M, e, tol=1e-10):
    """牛顿迭代求解偏近点角 E"""
    E = M
    while True:
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if abs(delta) < tol:
            break
    return E

def orbit_position(t, a, e):
    """根据时间 t（年）计算行星位置 (x, y)"""
    t_mod = t % T
    M = 2 * np.pi * t_mod / T
    E = solve_eccentric_anomaly(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)
    return x, y

def orbit_velocity(t, a, e):
    """计算行星瞬时速度 (AU/年) 的标量大小"""
    dt_small = 1e-6  # 微小时间间隔（年）
    x1, y1 = orbit_position(t, a, e)
    x2, y2 = orbit_position(t + dt_small, a, e)
    dx = x2 - x1
    dy = y2 - y1
    v = np.hypot(dx, dy) / dt_small  # AU/年
    return v

def compute_avg_speed(t_start, t_end, a, e, num_samples=50):
    """
    计算时间段内行星的平均速度（标量速率），单位 AU/年
    通过采样多个时间点的瞬时速度取平均获得
    """
    if t_start == t_end:
        return 0.0
    # 处理跨周期
    if t_end < t_start:
        t_end += T
    times = np.linspace(t_start, t_end, num_samples)
    speeds = []
    for ti in times:
        v = orbit_velocity(ti, a, e)
        speeds.append(v)
    return np.mean(speeds)

def compute_sector_area(t_start, t_end, a, e):
    """
    计算从 t_start 到 t_end 时间段内，太阳-行星连线扫过的面积。
    采样点数根据时间间隔动态调整。
    """
    if t_start == t_end:
        return 0.0

    # 调整时间区间，处理跨周期
    t_start_adj = t_start
    t_end_adj = t_end
    if t_end_adj < t_start_adj:
        t_end_adj += T
    while t_end_adj < t_start_adj:
        t_end_adj += T
    interval = t_end_adj - t_start_adj

    # 动态确定采样点数
    num_samples = max(AREA_SAMPLES_MIN, int(interval * AREA_SAMPLES_PER_UNIT))
    num_samples = min(num_samples, 500)  # 上限

    times = np.linspace(t_start_adj, t_end_adj, num_samples)
    area = 0.0
    theta_prev = None
    for ti in times:
        x, y = orbit_position(ti, a, e)
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        if theta_prev is not None:
            dtheta = theta - theta_prev
            # 处理角度跳变
            if dtheta > np.pi:
                dtheta -= 2*np.pi
            elif dtheta < -np.pi:
                dtheta += 2*np.pi
            area += 0.5 * r**2 * abs(dtheta)
        theta_prev = theta
    return area

# ==================== Tkinter 应用程序 ====================
class KeplerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("开普勒第二定律：等面积验证（含平均速度）")
        self.root.configure(bg='black')
        self.root.geometry("1450x850")

        # 动画状态
        self.t = 0.0
        self.paused = False
        self.speed_factor = 1.0
        self.auto_pause = False

        # 标记数据
        self.mark_times = []          # 时刻列表
        self.segment_visible = []     # 每段是否可见
        self.segment_avg_speed = []   # 每段平均速度 (AU/年)
        self.mark_points = []          # 蓝色标记点对象
        self.mark_fills = []           # 填充区域对象
        self.area_texts = []           # 面积文本对象

        # 参考面积
        self.ref_area = None

        # 控制长短轴显示
        self.show_axes = True

        # 布局
        self.create_layout()
        self.setup_plot()
        self.update_animation()
        self.bind_shortcuts()

    # ---------- 布局 ----------
    def create_layout(self):
        main_panel = tk.Frame(self.root, bg='black')
        main_panel.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板（宽280）
        self.left_frame = tk.Frame(main_panel, width=280, bg='black', relief=tk.SUNKEN, borderwidth=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_frame.pack_propagate(False)

        # 中间图形区域
        self.center_frame = tk.Frame(main_panel, bg='black')
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧任务面板（宽280）
        self.right_frame = tk.Frame(main_panel, width=280, bg='black', relief=tk.SUNKEN, borderwidth=2)
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.right_frame.pack_propagate(False)

        self.create_left_controls()
        self.create_right_panel()

    def create_left_controls(self):
        # 标题
        title_frame = tk.Frame(self.left_frame, bg='black')
        title_frame.pack(pady=(15,10))
        tk.Label(title_frame, text="🪐", font=('Segoe UI', 30), bg='black', fg='gold').pack(side=tk.LEFT, padx=5)
        tk.Label(title_frame, text="开普勒\n第二定律", font=('微软雅黑', 20, 'bold'),
                 bg='black', fg='cyan', justify=tk.LEFT).pack(side=tk.LEFT)

        # 偏心率滑块（暖橙色系）
        tk.Label(self.left_frame, text="偏心率 e:", font=('微软雅黑', 12),
                 bg='black', fg='white').pack(anchor=tk.W, padx=10, pady=(10,0))
        self.e_var = tk.DoubleVar(value=EARTH_ECCENTRICITY)
        e_slider = tk.Scale(self.left_frame, from_=0.0, to=0.5, orient=tk.VERTICAL,
                            variable=self.e_var, command=self.on_e_change,
                            length=150, resolution=0.001, bg='#FF8C00', fg='white',
                            troughcolor='#FFB347', highlightbackground='black',
                            activebackground='#FFA500', font=('微软雅黑', 10))
        e_slider.pack(pady=5)
        self.e_label = tk.Label(self.left_frame, text=f"{EARTH_ECCENTRICITY:.3f}",
                                font=('微软雅黑', 11), bg='black', fg='white')
        self.e_label.pack()

        # 重置地球偏心率按钮
        reset_earth_btn = tk.Button(self.left_frame, text="🌍 重置为地球偏心率", command=self.reset_to_earth,
                                     bg='#FFD700', fg='black', font=('微软雅黑', 9), width=18)
        reset_earth_btn.pack(pady=5)

        # 速度倍率滑块（冷绿色系）
        tk.Label(self.left_frame, text="速度倍率:", font=('微软雅黑', 12),
                 bg='black', fg='white').pack(anchor=tk.W, padx=10, pady=(15,0))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = tk.Scale(self.left_frame, from_=0.1, to=10.0, orient=tk.VERTICAL,
                                variable=self.speed_var, command=self.on_speed_change,
                                length=150, resolution=0.1, bg='#32CD32', fg='black',
                                troughcolor='#98FB98', highlightbackground='black',
                                activebackground='#3CB371', font=('微软雅黑', 10))
        speed_slider.pack(pady=5)
        self.speed_label = tk.Label(self.left_frame, text="1.0",
                                    font=('微软雅黑', 11), bg='black', fg='white')
        self.speed_label.pack()

        # 按钮区域
        btn_frame = tk.Frame(self.left_frame, bg='black')
        btn_frame.pack(pady=10)

        # 暂停/继续
        self.pause_btn = tk.Button(btn_frame, text="⏸️ 暂停", command=self.toggle_pause,
                                   width=14, bg='#FFA500', fg='black', relief=tk.RAISED,
                                   font=('微软雅黑', 11))
        self.pause_btn.grid(row=0, column=0, pady=2, padx=2)

        # 重置时间
        self.reset_time_btn = tk.Button(btn_frame, text="↺ 重置时间", command=self.reset_time,
                                        width=14, bg='#808080', fg='white', relief=tk.RAISED,
                                        font=('微软雅黑', 11))
        self.reset_time_btn.grid(row=0, column=1, pady=2, padx=2)

        # 步进按钮（前进/后退）
        self.step_back_btn = tk.Button(btn_frame, text="◀ 后退", command=self.step_back,
                                        width=7, bg='#A9A9A9', fg='white', relief=tk.RAISED,
                                        font=('微软雅黑', 9))
        self.step_back_btn.grid(row=1, column=0, pady=2, padx=2, sticky='ew')

        self.step_forward_btn = tk.Button(btn_frame, text="前进 ▶", command=self.step_forward,
                                          width=7, bg='#A9A9A9', fg='white', relief=tk.RAISED,
                                          font=('微软雅黑', 9))
        self.step_forward_btn.grid(row=1, column=1, pady=2, padx=2, sticky='ew')

        # 标记相关按钮
        self.mark_btn = tk.Button(btn_frame, text="📍 添加标记", command=self.add_mark,
                                  width=14, bg='#1E90FF', fg='white', relief=tk.RAISED,
                                  font=('微软雅黑', 11))
        self.mark_btn.grid(row=2, column=0, columnspan=2, pady=2, sticky='ew')

        self.find_equal_btn = tk.Button(btn_frame, text="⚖️ 等面积点", command=self.find_equal_area,
                                        width=14, bg='#32CD32', fg='black', relief=tk.RAISED,
                                        font=('微软雅黑', 11))
        self.find_equal_btn.grid(row=3, column=0, columnspan=2, pady=2, sticky='ew')

        self.reset_marks_btn = tk.Button(btn_frame, text="🗑️ 重置标记", command=self.reset_marks,
                                          width=14, bg='#FF4500', fg='white', relief=tk.RAISED,
                                          font=('微软雅黑', 11))
        self.reset_marks_btn.grid(row=4, column=0, columnspan=2, pady=2, sticky='ew')

        # 导出数据按钮
        self.export_btn = tk.Button(btn_frame, text="📁 导出标记数据", command=self.export_data,
                                     width=14, bg='#6A5ACD', fg='white', relief=tk.RAISED,
                                     font=('微软雅黑', 11))
        self.export_btn.grid(row=5, column=0, columnspan=2, pady=2, sticky='ew')

        # 复选框区域
        cb_frame = tk.Frame(self.left_frame, bg='black')
        cb_frame.pack(pady=5)

        self.auto_pause_var = tk.BooleanVar(value=False)
        self.auto_pause_cb = tk.Checkbutton(cb_frame, text="标记后自动暂停",
                                             variable=self.auto_pause_var,
                                             bg='black', fg='white', selectcolor='black',
                                             activebackground='black', activeforeground='white',
                                             font=('微软雅黑', 10))
        self.auto_pause_cb.pack(anchor=tk.W)

        self.show_terms_var = tk.BooleanVar(value=True)
        self.terms_cb = tk.Checkbutton(cb_frame, text="显示节气",
                                        variable=self.show_terms_var,
                                        command=self.toggle_terms,
                                        bg='black', fg='white', selectcolor='black',
                                        activebackground='black', activeforeground='white',
                                        font=('微软雅黑', 10))
        self.terms_cb.pack(anchor=tk.W)

        self.show_axes_var = tk.BooleanVar(value=True)
        self.axes_cb = tk.Checkbutton(cb_frame, text="显示长短轴",
                                       variable=self.show_axes_var,
                                       command=self.toggle_axes,
                                       bg='black', fg='white', selectcolor='black',
                                       activebackground='black', activeforeground='white',
                                       font=('微软雅黑', 10))
        self.axes_cb.pack(anchor=tk.W)

        # 信息显示区域
        info_frame = tk.Frame(self.left_frame, bg='black')
        info_frame.pack(fill=tk.X, pady=5)

        # 参考面积显示
        tk.Label(info_frame, text="参考面积 (第一段):", font=('微软雅黑', 10),
                 bg='black', fg='white').pack(anchor=tk.W)
        self.ref_area_label = tk.Label(info_frame, text="--", font=('微软雅黑', 10, 'bold'),
                                        bg='black', fg='cyan')
        self.ref_area_label.pack(anchor=tk.W)

        # 速度比显示
        self.speed_ratio_label = tk.Label(info_frame,
                                          text="近日/远日速度比: --",
                                          font=('微软雅黑', 10), bg='black', fg='white')
        self.speed_ratio_label.pack(anchor=tk.W)

        # 面积结果显示（带颜色）
        self.area_label = tk.Label(info_frame, text="", font=('微软雅黑', 9),
                                    bg='black', fg='white', wraplength=260, justify=tk.LEFT)
        self.area_label.pack(anchor=tk.W, pady=2)

        # 误差百分比显示（带颜色）
        self.error_label = tk.Label(info_frame, text="", font=('微软雅黑', 9, 'bold'),
                                     bg='black', fg='white', wraplength=260)
        self.error_label.pack(anchor=tk.W, pady=2)

        # 最后标记点信息
        self.last_mark_label = tk.Label(info_frame, text="最后标记: --", font=('微软雅黑', 9),
                                         bg='black', fg='white')
        self.last_mark_label.pack(anchor=tk.W)

        # 退出按钮
        tk.Button(self.left_frame, text="退出", command=self.root.quit,
                  width=14, bg='#DC143C', fg='white', relief=tk.RAISED,
                  font=('微软雅黑', 11)).pack(pady=10, side=tk.BOTTOM)

    def create_right_panel(self):
        # 标题
        title_frame = tk.Frame(self.right_frame, bg='black')
        title_frame.pack(pady=(15,10))
        tk.Label(title_frame, text="📋", font=('Segoe UI', 22), bg='black', fg='gold').pack(side=tk.LEFT, padx=5)
        tk.Label(title_frame, text="探究任务", font=('微软雅黑', 20, 'bold'),
                 bg='black', fg='yellow').pack(side=tk.LEFT)

        # 任务文本框 - 增大高度、字体、行间距，使其内容完全可见
        task_text = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD,
                                              width=32, height=15,
                                              bg='#222222', fg='lightgreen',
                                              font=('微软雅黑', 11), spacing2=5)
        task_text.pack(padx=10, pady=5, fill=tk.X)
        task_content = """【探究任务】
1. 暂停动画（或开启自动暂停），点击“📍 添加标记”记录点A。
2. 继续动画一段时间，再次点击“📍 添加标记”记录点B（此时AB段面积自动显示，并作为参考面积）。
3. 点击“⚖️ 等面积点”，程序自动计算从B开始、面积与AB相等的点C，并显示BC段。
4. 可多次点击“⚖️ 等面积点”，生成连续的等面积段。
5. 调节偏心率，观察面积段变化，理解定律的普适性。
6. 观察近日点/远日点速度比，验证“近快远慢”。

【操作提示】
- 自动暂停：勾选后，添加标记时动画自动暂停。
- 参考面积：以第一段面积为基准，后续等面积段误差会显示颜色。
- 重置时间或改变偏心率会清空所有标记。
- 键盘快捷键：空格（暂停/继续），←→（调节速度倍率）。"""
        task_text.insert(tk.END, task_content)
        task_text.config(state=tk.DISABLED)

        # 标记点列表 - 显示平均速度
        tk.Label(self.right_frame, text="标记点列表（含段平均速度）", font=('微软雅黑', 12, 'bold'),
                 bg='black', fg='cyan').pack(pady=(10,0))
        self.marks_listbox = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD,
                                                        width=32, height=8,
                                                        bg='#111111', fg='lightblue',
                                                        font=('Consolas', 10), spacing2=2)
        self.marks_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.marks_listbox.config(state=tk.DISABLED)

        tk.Label(self.right_frame, text="✨ 开普勒定律 ✨", font=('微软雅黑', 12),
                 bg='black', fg='cyan').pack(pady=5)

    # ---------- 事件处理 ----------
    def on_e_change(self, value):
        e = float(value)
        self.e_label.config(text=f"{e:.3f}")
        self.update_orbit_line(e)
        self.update_terms()
        self.update_axes(e)
        self.reset_marks()
        self.update_speed_ratio()
        self.canvas.draw_idle()

    def on_speed_change(self, value):
        self.speed_factor = float(value)
        self.speed_label.config(text=f"{self.speed_factor:.1f}")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="⏸️ 继续" if self.paused else "⏸️ 暂停")

    def reset_time(self):
        self.t = 0.0
        self.reset_marks()
        self.canvas.draw_idle()

    def step_back(self):
        """后退一帧（需暂停）"""
        if not self.paused:
            self.toggle_pause()
        self.t = max(0.0, self.t - DT * STEPS_PER_FRAME)
        self.canvas.draw_idle()

    def step_forward(self):
        """前进一帧（需暂停）"""
        if not self.paused:
            self.toggle_pause()
        self.t += DT * STEPS_PER_FRAME
        if self.t > T:
            self.t -= T
        self.canvas.draw_idle()

    def reset_to_earth(self):
        """重置偏心率到地球值"""
        self.e_var.set(EARTH_ECCENTRICITY)
        self.on_e_change(EARTH_ECCENTRICITY)

    def add_mark(self):
        self.mark_times.append(self.t)
        n = len(self.mark_times)

        if n >= 2:
            if n == 2:
                self.segment_visible.append(True)
                e = self.e_var.get()
                self.ref_area = compute_sector_area(self.mark_times[0], self.mark_times[1], AU, e)
                self.ref_area_label.config(text=f"{self.ref_area:.4f} AU²")
            else:
                self.segment_visible.append(False)
            # 新段平均速度暂时为None，等刷新时计算
            self.segment_avg_speed.append(None)
        else:
            # 只有一个点时，无段
            pass

        # 画标记点
        e = self.e_var.get()
        x, y = orbit_position(self.t, AU, e)
        point, = self.ax.plot(x, y, 'bo', markersize=6, markeredgecolor='white', markeredgewidth=0.5,
                               picker=True)
        self.mark_points.append(point)

        # 更新最后标记信息
        r = np.hypot(x, y)
        self.last_mark_label.config(text=f"最后标记: t={self.t:.3f}年, r={r:.3f}AU")

        if self.auto_pause_var.get() and not self.paused:
            self.toggle_pause()

        self.refresh_mark_areas()  # 重新计算所有可见段的面积和平均速度
        self.update_marks_list()   # 更新列表显示
        self.canvas.draw_idle()

    def reset_marks(self):
        self.mark_times.clear()
        self.segment_visible.clear()
        self.segment_avg_speed.clear()
        self.ref_area = None
        self.ref_area_label.config(text="--")

        for fill in self.mark_fills:
            fill.remove()
        self.mark_fills.clear()
        for txt in self.area_texts:
            txt.remove()
        self.area_texts.clear()
        for p in self.mark_points:
            p.remove()
        self.mark_points.clear()

        self.area_label.config(text="")
        self.error_label.config(text="")
        self.last_mark_label.config(text="最后标记: --")
        self.update_marks_list()
        self.canvas.draw_idle()

    def find_equal_area(self):
        if len(self.mark_times) < 2:
            self.area_label.config(text="至少需要两个标记点")
            return

        e = self.e_var.get()
        if self.ref_area is None:
            self.ref_area = compute_sector_area(self.mark_times[0], self.mark_times[1], AU, e)
            self.ref_area_label.config(text=f"{self.ref_area:.4f} AU²")

        t_start = self.mark_times[-1]
        left, right = t_start, t_start + T
        if compute_sector_area(t_start, right, AU, e) < self.ref_area:
            self.area_label.config(text="错误：参考面积大于一圈面积")
            return

        # 二分查找
        for _ in range(BISECTION_MAX_ITER):
            mid = (left + right) / 2
            area_mid = compute_sector_area(t_start, mid, AU, e)
            if abs(area_mid - self.ref_area) < BISECTION_TOL:
                break
            if area_mid < self.ref_area:
                left = mid
            else:
                right = mid
        t_end = (left + right) / 2

        self.mark_times.append(t_end)
        self.segment_visible.append(True)
        self.segment_avg_speed.append(None)  # 占位，刷新时会计算

        x, y = orbit_position(t_end, AU, e)
        point, = self.ax.plot(x, y, 'bo', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        self.mark_points.append(point)

        # 更新最后标记信息
        r = np.hypot(x, y)
        self.last_mark_label.config(text=f"最后标记: t={t_end:.3f}年, r={r:.3f}AU")

        self.refresh_mark_areas()
        self.update_marks_list()
        self.canvas.draw_idle()

    def export_data(self):
        """将标记点数据导出为CSV文件"""
        if len(self.mark_times) < 2:
            messagebox.showwarning("警告", "没有足够的标记点可导出")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"kepler_marks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not filename:
            return

        try:
            e = self.e_var.get()
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['序号', '时刻 (年)', 'x (AU)', 'y (AU)', '日心距离 (AU)',
                                 '是否为等面积段起点', '段平均速度 (km/s)'])
                for i, t in enumerate(self.mark_times):
                    x, y = orbit_position(t, AU, e)
                    r = np.hypot(x, y)
                    is_equal_start = (i < len(self.segment_visible) and self.segment_visible[i]) if i > 0 else False
                    # 计算该段平均速度（如果有前一标记）
                    avg_speed = ''
                    if i > 0 and self.segment_visible[i-1]:
                        # 使用存储的平均速度值
                        if i-1 < len(self.segment_avg_speed):
                            avg_speed_val = self.segment_avg_speed[i-1]
                            if avg_speed_val is not None:
                                avg_speed = f"{avg_speed_val:.2f}"
                    writer.writerow([i+1, f"{t:.6f}", f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", is_equal_start, avg_speed])
            messagebox.showinfo("导出成功", f"数据已保存至 {filename}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def update_marks_list(self):
        """更新右侧标记点列表，显示平均速度"""
        self.marks_listbox.config(state=tk.NORMAL)
        self.marks_listbox.delete(1.0, tk.END)
        if not self.mark_times:
            self.marks_listbox.insert(tk.END, "暂无标记点")
        else:
            e = self.e_var.get()
            for i, t in enumerate(self.mark_times):
                x, y = orbit_position(t, AU, e)
                r = np.hypot(x, y)
                line = f"{i+1:2d}. t={t:.3f}年, r={r:.3f}AU"
                # 如果存在前一标记，显示平均速度
                if i > 0 and i-1 < len(self.segment_avg_speed):
                    avg_speed = self.segment_avg_speed[i-1]
                    if avg_speed is not None:
                        line += f", v_avg={avg_speed:.1f} km/s"
                self.marks_listbox.insert(tk.END, line + "\n")
        self.marks_listbox.config(state=tk.DISABLED)

    def refresh_mark_areas(self):
        """重新绘制面积填充，并计算每段平均速度"""
        for fill in self.mark_fills:
            fill.remove()
        self.mark_fills.clear()
        for txt in self.area_texts:
            txt.remove()
        self.area_texts.clear()

        if len(self.mark_times) < 2:
            self.area_label.config(text="至少需要两个标记点")
            return

        e = self.e_var.get()
        areas_info = []
        error_info = ""
        colors = ['lime', 'orange', 'violet', 'cyan', 'magenta', 'gold']

        # 清空平均速度列表
        self.segment_avg_speed = [None] * (len(self.mark_times) - 1)

        for i in range(len(self.mark_times) - 1):
            if not self.segment_visible[i]:
                continue

            t_start = self.mark_times[i]
            t_end = self.mark_times[i+1]
            area = compute_sector_area(t_start, t_end, AU, e)
            areas_info.append(f"段{i+1}: {area:.4f} AU²")

            # 计算平均速度 (km/s)
            avg_speed_au_per_year = compute_avg_speed(t_start, t_end, AU, e, num_samples=50)
            avg_speed_km_s = avg_speed_au_per_year * (AU_KM / YEAR_SEC)
            self.segment_avg_speed[i] = avg_speed_km_s

            # 填充多边形
            t_start_adj = t_start
            t_end_adj = t_end
            if t_end_adj < t_start_adj:
                t_end_adj += T
            times = np.linspace(t_start_adj, t_end_adj, 100)
            xs = [0] + [orbit_position(ti, AU, e)[0] for ti in times] + [0]
            ys = [0] + [orbit_position(ti, AU, e)[1] for ti in times] + [0]
            fill = self.ax.fill(xs, ys, alpha=0.4, color=colors[i % len(colors)],
                                edgecolor='white', linewidth=0.5)[0]
            self.mark_fills.append(fill)

            # 标注面积数值
            t_mid = (t_start_adj + t_end_adj) / 2
            xm, ym = orbit_position(t_mid, AU, e)
            rm = np.hypot(xm, ym)
            if rm < 0.1:
                xm, ym = (0.2, 0) if rm == 0 else (xm/rm*0.2, ym/rm*0.2)
            txt = self.ax.text(xm, ym, f"{area:.3f}", fontsize=9, color='white',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            self.area_texts.append(txt)

        if areas_info:
            self.area_label.config(text="  ".join(areas_info))
            if self.ref_area is not None and self.ref_area > 0:
                errors = []
                for i in range(len(self.mark_times)-1):
                    if self.segment_visible[i]:
                        area = compute_sector_area(self.mark_times[i], self.mark_times[i+1], AU, e)
                        errors.append(abs(area - self.ref_area) / self.ref_area * 100)
                if errors:
                    avg_err = np.mean(errors)
                    max_err = max(errors)
                    if avg_err < 1.0:
                        color = 'lightgreen'
                    elif avg_err < 5.0:
                        color = 'yellow'
                    else:
                        color = 'orange'
                    error_info = f"平均误差: {avg_err:.2f}% | 最大: {max_err:.2f}%"
                    self.error_label.config(text=error_info, fg=color)
                else:
                    self.error_label.config(text="")
        else:
            self.area_label.config(text="无可显示的面积段")
            self.error_label.config(text="")

        # 更新右侧列表显示
        self.update_marks_list()

    def update_speed_ratio(self):
        e = self.e_var.get()
        ratio = np.sqrt((1+e)/(1-e))
        self.speed_ratio_label.config(text=f"近日/远日速度比: {ratio:.3f}")

    def toggle_terms(self):
        visible = self.show_terms_var.get()
        for point, text in self.term_artists.values():
            point.set_visible(visible)
            text.set_visible(visible)
        self.canvas.draw_idle()

    def toggle_axes(self):
        self.show_axes = self.show_axes_var.get()
        self.major_axis_line.set_visible(self.show_axes)
        self.minor_axis_line.set_visible(self.show_axes)
        self.a_label.set_visible(self.show_axes)
        self.b_label.set_visible(self.show_axes)
        self.other_focus_text.set_visible(self.show_axes)
        self.center_point.set_visible(self.show_axes)
        self.canvas.draw_idle()

    def bind_shortcuts(self):
        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<Left>', lambda e: self.adjust_speed(-0.5))
        self.root.bind('<Right>', lambda e: self.adjust_speed(0.5))

    def adjust_speed(self, delta):
        new_val = self.speed_factor + delta
        if 0.1 <= new_val <= 10.0:
            self.speed_var.set(new_val)
            self.on_speed_change(new_val)

    # ---------- 绘图 ----------
    def setup_plot(self):
        self.fig = plt.figure(figsize=(7, 7), facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')

        # 星星（150颗）
        np.random.seed(42)
        stars_x = np.random.uniform(-AU*1.5, AU*1.5, 150)
        stars_y = np.random.uniform(-AU*1.5, AU*1.5, 150)
        self.ax.scatter(stars_x, stars_y, c='white', s=1.5, alpha=0.6)

        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.15, color='gray')
        self.ax.set_xlim(-AU*1.5, AU*1.5)
        self.ax.set_ylim(-AU*1.5, AU*1.5)
        self.ax.set_xlabel('x (AU)', color='white', fontsize=12)
        self.ax.set_ylabel('y (AU)', color='white', fontsize=12)
        self.ax.set_title('开普勒第二定律：等面积验证', color='cyan', fontsize=22, fontweight='bold', pad=20)
        self.ax.tick_params(colors='white', labelsize=11)

        # 太阳
        if sun_img is not None:
            sun_imagebox = OffsetImage(sun_img, zoom=0.06)
            self.sun_ab = AnnotationBbox(sun_imagebox, (0, 0), frameon=False)
            self.ax.add_artist(self.sun_ab)
            self.sun = self.sun_ab
        else:
            self.sun = self.ax.plot(0, 0, 'yo', markersize=20, markeredgecolor='orange', markeredgewidth=1)[0]

        e = self.e_var.get()

        # 另一个焦点（红色圆点）
        other_focus_x = -2 * AU * e
        self.other_focus, = self.ax.plot(other_focus_x, 0, 'o', color='red', markersize=8,
                                          markeredgecolor='white', markeredgewidth=1)
        self.other_focus_text = self.ax.text(other_focus_x, 0.1, '另一个焦点', fontsize=9, color='white',
                                              ha='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='red'))

        # 椭圆中心（灰色小点）
        center_x = -AU * e
        self.center_point, = self.ax.plot(center_x, 0, 'o', color='gray', markersize=5, alpha=0.7)

        # 长轴（白色虚线）
        perihelion = (AU*(1-e), 0)
        aphelion = (-AU*(1+e), 0)
        self.major_axis_line, = self.ax.plot([aphelion[0], perihelion[0]], [0, 0],
                                              linestyle='--', color='white', lw=1.5, alpha=0.7)

        # 短轴（白色虚线）
        b = AU * np.sqrt(1 - e**2)
        self.minor_axis_line, = self.ax.plot([center_x, center_x], [-b, b],
                                              linestyle='--', color='white', lw=1.5, alpha=0.7)

        # 半长轴标注
        self.a_label = self.ax.text(center_x + AU/2, 0.1, f'a = {AU:.2f}', fontsize=10, color='white',
                                     ha='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan'))
        # 半短轴标注
        self.b_label = self.ax.text(center_x + 0.1, b/2, f'b = {b:.2f}', fontsize=10, color='white',
                                     ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan'))

        # 轨道（高亮发光）
        self.theta = np.linspace(0, 2*np.pi, 500)
        x_orbit = AU * (np.cos(self.theta) - e)
        y_orbit = AU * np.sqrt(1 - e**2) * np.sin(self.theta)
        self.orbit_glow, = self.ax.plot(x_orbit, y_orbit, color='cyan', lw=5, alpha=0.2)
        self.orbit_line, = self.ax.plot(x_orbit, y_orbit, color='cyan', lw=2.5, label='轨道')

        # 行星
        if earth_img is not None:
            earth_imagebox = OffsetImage(earth_img, zoom=0.045)
            self.earth_ab = AnnotationBbox(earth_imagebox, (0, 0), frameon=False)
            self.ax.add_artist(self.earth_ab)
            self.planet = self.earth_ab
            self.earth_glow = Circle((0, 0), radius=0.12, color='deepskyblue', alpha=0.3, zorder=1)
            self.ax.add_patch(self.earth_glow)
        else:
            self.planet, = self.ax.plot([], [], 'o', color='deepskyblue', markersize=14,
                                         markeredgecolor='white', markeredgewidth=0.5)
            self.earth_glow = None

        # 日-行连线
        self.line, = self.ax.plot([], [], 'w--', lw=1.2, alpha=0.8)

        # 节气标注
        self.term_artists = {}
        for name, f in SOLAR_TERMS.items():
            r = AU*(1-e**2)/(1+e*np.cos(f))
            xt = r*np.cos(f)
            yt = r*np.sin(f)
            point, = self.ax.plot(xt, yt, '^', color='orange', markersize=11, markeredgecolor='white', markeredgewidth=0.5)
            text = self.ax.text(xt+0.1, yt+0.1, name, fontsize=11, color='white', weight='bold',
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='orange'))
            self.term_artists[name] = (point, text)

        # 信息框
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12,
                                       verticalalignment='top', color='white',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_speed_ratio()

    def update_axes(self, e):
        other_x = -2 * AU * e
        self.other_focus.set_data([other_x], [0])
        self.other_focus_text.set_position((other_x, 0.1))

        center_x = -AU * e
        self.center_point.set_data([center_x], [0])

        perihelion = AU*(1-e)
        aphelion = -AU*(1+e)
        self.major_axis_line.set_data([aphelion, perihelion], [0, 0])

        b = AU * np.sqrt(1 - e**2)
        self.minor_axis_line.set_data([center_x, center_x], [-b, b])

        self.a_label.set_position((center_x + AU/2, 0.1))
        self.b_label.set_position((center_x + 0.1, b/2))
        self.b_label.set_text(f'b = {b:.2f}')

        visible = self.show_axes_var.get()
        self.major_axis_line.set_visible(visible)
        self.minor_axis_line.set_visible(visible)
        self.a_label.set_visible(visible)
        self.b_label.set_visible(visible)
        self.other_focus_text.set_visible(visible)
        self.center_point.set_visible(visible)

    def update_orbit_line(self, e):
        xo = AU * (np.cos(self.theta) - e)
        yo = AU * np.sqrt(1 - e**2) * np.sin(self.theta)
        self.orbit_line.set_data(xo, yo)
        self.orbit_glow.set_data(xo, yo)

    def update_terms(self):
        e = self.e_var.get()
        for name, f in SOLAR_TERMS.items():
            r = AU*(1-e**2)/(1+e*np.cos(f))
            xt = r*np.cos(f)
            yt = r*np.sin(f)
            point, text = self.term_artists[name]
            point.set_data([xt], [yt])
            text.set_position((xt+0.1, yt+0.1))

    def update_animation(self):
        if not self.paused:
            self.t += DT * STEPS_PER_FRAME * self.speed_factor
            if self.t > T:
                self.t -= T

        e = self.e_var.get()
        x, y = orbit_position(self.t, AU, e)

        # 更新行星位置
        if earth_img is not None:
            self.earth_ab.xybox = (x, y)
            self.earth_ab.xy = (x, y)
            if self.earth_glow is not None:
                self.earth_glow.center = (x, y)
        else:
            self.planet.set_data([x], [y])

        self.line.set_data([0, x], [0, y])

        r = np.hypot(x, y)
        if abs(r - AU*(1-e)) < 0.01:
            pos = '近日点'
        elif abs(r - AU*(1+e)) < 0.01:
            pos = '远日点'
        else:
            pos = ''
        info = f"时间: {self.t:.3f} 年\n距离: {r:.3f} AU\n{pos}"
        self.info_text.set_text(info)

        self.canvas.draw_idle()
        self.root.after(ANIMATION_INTERVAL, self.update_animation)

# ==================== 启动 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = KeplerApp(root)
    root.mainloop()
