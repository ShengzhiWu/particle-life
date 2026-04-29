import numpy as np
import taichi as ti
import tkinter as tk
import os
import ctypes
import time
import colorsys
from ctypes import wintypes

ti.init(arch=ti.gpu)

# -----------------------------
# Simulation configuration
# -----------------------------
DIMENSION = 2
N_PARTICLES = 6000
space_filling_factor = 10  # 10 (for 2d) 1 (for 3d) 0.1 (for 4d)
BOX_SIZE = 1.0  # Simulation box is [0, BOX_SIZE] x [0, BOX_SIZE]
R_FACTOR = (BOX_SIZE ** DIMENSION / (N_PARTICLES / space_filling_factor)) ** (1.0 / DIMENSION)  # Interaction radius factor relative to box size
R1 = R_FACTOR * 1
R2 = R_FACTOR * 5  # 5
DT = 0.002
REPULSION_STRENGTH = 2
SUBSTEPS_PER_FRAME = 10  # 5
print(f'R1 = {R1:.5f}, R2 = {R2:.5f}')

# Non-symmetric interaction matrix in [-1, 1]
# interaction[a, b] means force coefficient on type-a particle from type-b particle
a = 0.3
b = 0.05
INTERACTION_INIT = np.array(  # 相互作用矩阵，负值相斥正值相吸
    # [
    #     [0.03, -1],
    #     [0.01, -0.1],
    # ],

    # 彩虹虫子
    [
        [a, 0, 0, 0, 0],
        [b, a, 0, 0, 0],
        [0, b, a, 0, 0],
        [0, 0, b, a, 0],
        [0, 0, 0, b, a],
    ],
    dtype=np.float32,
)

# Grid
MAX_GRID_N = max(1, int(BOX_SIZE / R2))
GRID_N = MAX_GRID_N
CELL_SIZE = BOX_SIZE / GRID_N
MAX_PARTICLES_PER_CELL = int(N_PARTICLES / GRID_N ** DIMENSION * 2)  # 每个格子中的最大粒子数

def get_windows_work_area():  # 获取Windows工作区（任务栏以上的空间）尺寸
    if os.name != "nt":  # 非 Windows 系统不支持
        return None

    SPI_GETWORKAREA = 0x0030
    rect = wintypes.RECT()
    ok = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    if ok:
        return rect.left, rect.top, rect.right, rect.bottom
    return None

def center_window_on_screen_windows(window_title, retries=50, delay_s=0.01):  # 让窗口居中
    if os.name != "nt":
        return False

    user32 = ctypes.windll.user32

    for _ in range(retries):
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return False

            window_w = rect.right - rect.left
            window_h = rect.bottom - rect.top

            work_area = get_windows_work_area()
            if work_area is not None:
                left, top, right, bottom = work_area
                usable_w = right - left
                usable_h = bottom - top
                x = left + max(0, (usable_w - window_w) // 2)
                y = top + max(0, (usable_h - window_h) // 2)
            else:
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
                x = max(0, (screen_w - window_w) // 2)
                y = max(0, (screen_h - window_h) // 2)

            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(hwnd, 0, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)
            return True
        time.sleep(delay_s)

    return False

try:
    work_area = get_windows_work_area()
    if work_area is not None:
        left, top, right, bottom = work_area
        usable_w = right - left
        usable_h = bottom - top
        WINDOW_RES = max(200, int(min(usable_w, usable_h) * 0.95))  # 根据可用工作区自动设置窗口尺寸
    else:
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        WINDOW_RES = max(200, int(min(screen_w, screen_h) * 0.9))  # 回退为屏幕尺寸
        root.destroy()
except Exception:
    WINDOW_RES = 900
PARTICLE_RADIUS = 1.5  # 粒子显示尺寸
_n_types = len(INTERACTION_INIT)
PARTICLE_COLORS = np.array(
    [int('{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in colorsys.hsv_to_rgb(i / _n_types, 1.0, 1.0)]), 16)
     for i in range(_n_types)],
    dtype=np.uint32,
)  # 粒子颜色，在色相环上等间距分布

BACKGROUND_COLOR = 0x000000

print(f"DIMENSION = {DIMENSION}, GRID_N = {GRID_N}, CELL_SIZE = {CELL_SIZE:.5f}")

if np.any(INTERACTION_INIT < -1.0) or np.any(INTERACTION_INIT > 1.0):
    raise ValueError("Interaction matrix values must be within [-1, 1].")

pos = ti.Vector.field(DIMENSION, dtype=ti.f32, shape=N_PARTICLES)
force = ti.Vector.field(DIMENSION, dtype=ti.f32, shape=N_PARTICLES)
ptype = ti.field(dtype=ti.i32, shape=N_PARTICLES)  # 粒子类型

interaction = ti.field(dtype=ti.f32, shape=INTERACTION_INIT.shape)

cell_shape = (GRID_N,) * DIMENSION
cell_count = ti.field(dtype=ti.i32, shape=cell_shape)
cell_particles = ti.field(dtype=ti.i32, shape=cell_shape + (MAX_PARTICLES_PER_CELL,))
overflow_counter = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def init_particles_and_types():
    for i in range(N_PARTICLES):
        p = ti.Vector.zero(ti.f32, DIMENSION)
        for d in ti.static(range(DIMENSION)):
            p[d] = ti.random(dtype=ti.f32) * BOX_SIZE
        pos[i] = p
        ptype[i] = ti.cast(ti.random(dtype=ti.f32) * interaction.shape[0], ti.i32)
        force[i] = ti.Vector.zero(ti.f32, DIMENSION)


@ti.func
def periodic_delta_vec(dx):
    # Map each component into [-BOX_SIZE/2, BOX_SIZE/2) for minimum-image displacement.
    return (dx + 0.5 * BOX_SIZE) % BOX_SIZE - 0.5 * BOX_SIZE


@ti.func
def middle_band_profile(r):
    # Triangular profile: 0 at r1, 1 at midpoint, 0 at r2
    t = (r - R1) / (R2 - R1)
    return 1.0 - ti.abs(2.0 * t - 1.0)


@ti.kernel
def clear_grid():
    overflow_counter[None] = 0
    for idx in ti.grouped(ti.ndrange(*cell_count.shape)):
        cell_count[idx] = 0


@ti.kernel
def build_grid(cell_particles: ti.template()):  # 将粒子分配到网格中 # type: ignore
    for i in range(N_PARTICLES):
        g = ti.cast(pos[i] / CELL_SIZE, ti.i32)

        slot = ti.atomic_add(cell_count[g], 1)
        if slot < MAX_PARTICLES_PER_CELL:
            cell_particles[g, slot] = i
        else:
            ti.atomic_add(overflow_counter[None], 1)


@ti.kernel
def compute_forces(cell_particles: ti.template()):  # 计算受力 # type: ignore
    for i in range(N_PARTICLES):
        xi = pos[i]
        ti_i = ptype[i]
        fi = ti.Vector.zero(ti.f32, DIMENSION)

        c = ti.cast(xi / CELL_SIZE, ti.i32)

        for offs in ti.grouped(ti.ndrange(*((-1, 2),) * DIMENSION)):  # 遍历临近的格子
            n = (c + offs + GRID_N) % GRID_N
            count = ti.min(cell_count[n], MAX_PARTICLES_PER_CELL)

            for k in range(count):
                j = cell_particles[n, k]
                if j != i:
                    rij = periodic_delta_vec(pos[j] - xi)
                    r = rij.norm()

                    if r < R2:
                        unit = rij / r

                        if r < R1:
                            mag = -REPULSION_STRENGTH * (1.0 - r / R1)
                            fi += mag * unit
                        else:
                            tri = middle_band_profile(r)
                            ti_j = ptype[j]
                            mag = interaction[ti_i, ti_j] * tri
                            fi += mag * unit

        force[i] = fi


@ti.kernel
def integrate_overdamped():  # 更新位置
    for i in range(N_PARTICLES):
        pos[i] = (pos[i] + force[i] * (DT * R_FACTOR)) % BOX_SIZE


# ---------- Main ----------
init_particles_and_types()
interaction.from_numpy(INTERACTION_INIT)

ptype_np = ptype.to_numpy()
colors_np = PARTICLE_COLORS[ptype_np]

window = ti.GUI("Particle Life", res=(WINDOW_RES, WINDOW_RES), background_color=BACKGROUND_COLOR)  # 创建窗口
center_window_on_screen_windows("Particle Life")  # 居中窗口

frame = 0
last_time = time.perf_counter()
fps = None
while window.running:
    for _ in range(SUBSTEPS_PER_FRAME):
        clear_grid()
        build_grid(cell_particles)
        overflow = overflow_counter[None]  # 格子比较多时这里会消耗一些时间
        while overflow > 0:  # 有的格子溢出了
            MAX_PARTICLES_PER_CELL *= 2  # 加倍格子容量
            cell_particles = ti.field(dtype=ti.i32, shape=cell_shape + (MAX_PARTICLES_PER_CELL,))
            print(f"Grid overflow. The capacity has been automatically doubled.")
            clear_grid()
            build_grid(cell_particles)
            overflow = overflow_counter[None]  # 格子比较多时这里会消耗一些时间
        compute_forces(cell_particles)
        integrate_overdamped()

    # 可视化（粒子比较多时这里非常耗时）
    positions_np = pos.to_numpy()[:, :2] / BOX_SIZE
    window.circles(positions_np, radius=PARTICLE_RADIUS, color=colors_np)

    now = time.perf_counter()
    dt_frame = now - last_time
    last_time = now
    if dt_frame > 0:
        fps = fps * 0.9 + (1.0 / dt_frame) * 0.1 if fps is not None else 1.0 / dt_frame

    window.text(f"FPS: {fps:.1f}", pos=(0.01, 0.03), font_size=18, color=0xFFFFFF)  # 显示帧率

    window.show()
    frame += 1
