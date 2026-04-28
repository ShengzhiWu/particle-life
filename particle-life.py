import numpy as np
import taichi as ti

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

# Render settings
WINDOW_RES = 900
PARTICLE_RADIUS = 1.3
PARTICLE_COLORS = np.array([0x3EA6FF, 0xFF7043], dtype=np.uint32)

BACKGROUND_COLOR = 0x101014

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

window = ti.GUI("Particle Life", res=(WINDOW_RES, WINDOW_RES), background_color=BACKGROUND_COLOR)

frame = 0
while window.running:
    for _ in range(SUBSTEPS_PER_FRAME):
        clear_grid()
        build_grid()
        compute_forces()
        integrate_overdamped()

    positions_np = pos.to_numpy() / BOX_SIZE
    window.circles(positions_np, radius=PARTICLE_RADIUS, color=colors_np)

    overflow = overflow_counter[None]
    if overflow > 0 and frame % 120 == 0:
        print(f"[Warning] Grid overflow count = {overflow}. Consider increasing MAX_PARTICLES_PER_CELL.")

    window.show()
    frame += 1
