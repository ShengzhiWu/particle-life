import numpy as np
import taichi as ti
import tkinter as tk
import os
import ctypes
import time
from ctypes import wintypes

ti.init(arch=ti.gpu)

# -----------------------------
# Simulation configuration
# -----------------------------
N_PARTICLES = 40000
BOX_SIZE = 1.0  # Simulation box is [0, BOX_SIZE] x [0, BOX_SIZE]
R_FACTOR = (BOX_SIZE * BOX_SIZE / N_PARTICLES) ** 0.5 * 1.0  # Interaction radius factor relative to box size
R1 = R_FACTOR * 1
R2 = R_FACTOR * 5
DT = 0.0002
REPULSION_STRENGTH = 2
SUBSTEPS_PER_FRAME = 20

# Grid requirement: cell_size >= R2
REQUESTED_GRID_N = 24
MAX_GRID_N = max(1, int(BOX_SIZE / R2))
GRID_N = min(REQUESTED_GRID_N, MAX_GRID_N)
CELL_SIZE = BOX_SIZE / GRID_N
# assert CELL_SIZE >= R2, "Grid cell size must be >= R2."

# Fixed-capacity particle list per cell for high performance
MAX_PARTICLES_PER_CELL = 256

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
PARTICLE_RADIUS = 1.3  # 粒子显示尺寸
PARTICLE_COLORS = np.array([0x3EA6FF, 0xFF7043], dtype=np.uint32)  # 粒子颜色

BACKGROUND_COLOR = 0x000000

print(f"GRID_N={GRID_N}, CELL_SIZE={CELL_SIZE:.5f}, MAX_GRID_N={MAX_GRID_N}")

# Non-symmetric interaction matrix in [-1, 1]
# interaction[a, b] means force coefficient on type-a particle from type-b particle
INTERACTION_INIT = np.array(  # 相互作用矩阵
    [
        [0.1, 0],
        [0,   0],
    ],
    dtype=np.float32,
)

if np.any(INTERACTION_INIT < -1.0) or np.any(INTERACTION_INIT > 1.0):
    raise ValueError("Interaction matrix values must be within [-1, 1].")

pos = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
force = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
ptype = ti.field(dtype=ti.i32, shape=N_PARTICLES)

interaction = ti.field(dtype=ti.f32, shape=(2, 2))

cell_count = ti.field(dtype=ti.i32, shape=(GRID_N, GRID_N))
cell_particles = ti.field(dtype=ti.i32, shape=(GRID_N, GRID_N, MAX_PARTICLES_PER_CELL))
overflow_counter = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def init_particles_and_types():
    for i in range(N_PARTICLES):
        pos[i] = ti.Vector([ti.random(dtype=ti.f32) * BOX_SIZE, ti.random(dtype=ti.f32) * BOX_SIZE])
        ptype[i] = ti.cast(ti.random(dtype=ti.f32) * 2.0, ti.i32)
        force[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def clear_grid():
    overflow_counter[None] = 0
    for i, j in ti.ndrange(GRID_N, GRID_N):
        cell_count[i, j] = 0


@ti.kernel
def build_grid():
    for i in range(N_PARTICLES):
        gx = ti.cast(pos[i][0] / CELL_SIZE, ti.i32)
        gy = ti.cast(pos[i][1] / CELL_SIZE, ti.i32)

        gx = ti.max(0, ti.min(gx, GRID_N - 1))
        gy = ti.max(0, ti.min(gy, GRID_N - 1))

        slot = ti.atomic_add(cell_count[gx, gy], 1)
        if slot < MAX_PARTICLES_PER_CELL:
            cell_particles[gx, gy, slot] = i
        else:
            ti.atomic_add(overflow_counter[None], 1)


@ti.func
def periodic_delta(xj, xi):
    d = xj - xi
    if d > 0.5 * BOX_SIZE:
        d -= BOX_SIZE
    elif d < -0.5 * BOX_SIZE:
        d += BOX_SIZE
    return d


@ti.func
def middle_band_profile(r):
    # Triangular profile: 0 at r1, 1 at midpoint, 0 at r2
    t = (r - R1) / (R2 - R1)
    return 1.0 - ti.abs(2.0 * t - 1.0)


@ti.kernel
def compute_forces():  # 计算受力
    for i in range(N_PARTICLES):
        xi = pos[i]
        ti_i = ptype[i]
        fi = ti.Vector([0.0, 0.0])

        cx = ti.cast(xi[0] / CELL_SIZE, ti.i32)
        cy = ti.cast(xi[1] / CELL_SIZE, ti.i32)
        cx = ti.max(0, ti.min(cx, GRID_N - 1))
        cy = ti.max(0, ti.min(cy, GRID_N - 1))

        for ox, oy in ti.ndrange((-1, 2), (-1, 2)):
            nx = (cx + ox + GRID_N) % GRID_N
            ny = (cy + oy + GRID_N) % GRID_N
            count = ti.min(cell_count[nx, ny], MAX_PARTICLES_PER_CELL)

            for k in range(count):
                j = cell_particles[nx, ny, k]
                if j != i:
                    xj = pos[j]
                    dx = periodic_delta(xj[0], xi[0])
                    dy = periodic_delta(xj[1], xi[1])
                    rij = ti.Vector([dx, dy])
                    r = rij.norm()

                    if r < R2:
                        unit = rij / r
                        mag = 0.0

                        if r < R1:
                            mag = -REPULSION_STRENGTH * (1.0 - r / R1)
                        else:
                            tri = middle_band_profile(r)
                            ti_j = ptype[j]
                            mag = interaction[ti_i, ti_j] * tri

                        fi += mag * unit

        force[i] = fi


@ti.kernel
def integrate_overdamped():  # 更新位置
    for i in range(N_PARTICLES):
        pos[i] += force[i] * DT

        # Periodic boundary condition on a square domain
        if pos[i][0] >= BOX_SIZE:
            pos[i][0] -= BOX_SIZE
        elif pos[i][0] < 0.0:
            pos[i][0] += BOX_SIZE

        if pos[i][1] >= BOX_SIZE:
            pos[i][1] -= BOX_SIZE
        elif pos[i][1] < 0.0:
            pos[i][1] += BOX_SIZE


# ---------- Main ----------
init_particles_and_types()
interaction.from_numpy(INTERACTION_INIT)

ptype_np = ptype.to_numpy()
colors_np = PARTICLE_COLORS[ptype_np]

print("Interaction matrix:")
print(INTERACTION_INIT)

window = ti.GUI("Particle Life", res=(WINDOW_RES, WINDOW_RES), background_color=BACKGROUND_COLOR)  # 创建窗口
center_window_on_screen_windows("Particle Life")  # 居中窗口

frame = 0
last_time = time.perf_counter()
fps = 0.0
while window.running:
    for _ in range(SUBSTEPS_PER_FRAME):
        clear_grid()
        build_grid()
        compute_forces()
        integrate_overdamped()

    positions_np = pos.to_numpy() / BOX_SIZE
    window.circles(positions_np, radius=PARTICLE_RADIUS, color=colors_np)

    now = time.perf_counter()
    dt_frame = now - last_time
    last_time = now
    if dt_frame > 0:
        fps = fps * 0.9 + (1.0 / dt_frame) * 0.1

    window.text(f"FPS: {fps:.1f}", pos=(0.01, 0.03), font_size=18, color=0xFFFFFF)  # 显示帧率

    overflow = overflow_counter[None]
    if overflow > 0 and frame % 120 == 0:
        print(f"[Warning] Grid overflow count = {overflow}. Consider increasing MAX_PARTICLES_PER_CELL.")

    window.show()
    frame += 1
