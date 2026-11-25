import numpy as np
from scipy.spatial import cKDTree

from .eqs import accumulate_contribs, tait
from .kernels import wendland, wendland_gradient
from .time_integration import symplectic, rk4

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

class SPHSolver:
    def __init__(self, dim:int=1, dx:float=.02, H: float=1, L:float=None, 
        T:float=0.5, rho0:float=1000, gm:int=7, c0:float=None, drho_max:float=0.01, 
        accel:np.ndarray=None, alpha:float=0.1, nu_phys:float=0, delta:float=0,
        h_factor:float=1.3, cfl:float=0.25, rho_ref:np.ndarray=None, 
        wall_density:callable=None, wall_noslip:bool=False, wall:np.ndarray=None,
        viscosity:str="artificial", stepper:str="symplectic", 
        periodic_axes=(), kernel=wendland, kernel_gradient=wendland_gradient,
    ):
        
        # tests and verifications
        if dim not in (1, 2):
            raise ValueError("only dim=1 and dim=2 is valid")
        if accel is None:                     
            accel = np.zeros(dim); accel[-1] = -9.81
        accel = np.array(accel, dtype=float) 
        if accel.shape != (dim,) or not np.all(np.isfinite(accel)):
            raise ValueError(f"accel needs shape ({dim},), got {accel}")
        if viscosity not in ("artificial", "laminar"):
            raise ValueError(f"unknown viscosity: {viscosity}")
        if stepper == "symplectic":
            self.stepper = symplectic
        elif stepper == "rk4":
            self.stepper = rk4
        else:
            raise ValueError(f"unknown stepper: {stepper}")
        
        self.periodic_axes = tuple(periodic_axes)
        self.L = tuple(L) if L is not None else tuple(None for _ in range(dim))
        for axis in self.periodic_axes:
            if not 0 <= axis < dim:
                raise ValueError(f"invalid periodic axis {axis}")
            if self.L[axis] is None:
                raise ValueError(f"missing domain length for periodic axis {axis}")

        # assign all values to class
        self.dim = dim;             self.dx = dx;           self.h = h_factor*self.dx
        self.support = 2*self.h;    self.accel = accel;     self.nu_phys = nu_phys
        self.rho0 = rho0;           self.gm = gm;           self.T = T
        self.delta = delta;         self.rho_ref = rho_ref
        self.wall = None if wall is None else np.asarray(wall, dtype=bool)
        self.viscosity = viscosity; self.alpha = alpha;     
        
        if self.rho_ref is None:
            self.rho_ref = lambda x: np.full(len(x), self.rho0)
        if not c0:                              
            c0 = np.sqrt(abs(self.accel[-1])*H/drho_max)
        if not c0 > 0:
            raise ValueError("c0 must be > 0")
        self.c0 = c0
        self.B = self.rho0*self.c0**2/self.gm
        self.mass = self.rho0*self.dx**dim
        self.nu = (self.nu_phys if viscosity == "laminar"
                   else self.alpha*self.h*self.c0/8)

        limits = [self.h/self.c0]
        if self.nu > 0:
            limits.append(self.h**2/self.nu)
        f = np.linalg.norm(self.accel)
        if f > 0:
            limits.append(np.sqrt(self.h/f))
        self.cfl = cfl
        self.dt = self.cfl*min(limits)
        self.n_steps = int(np.ceil(self.T/self.dt))

        self.kernel = kernel
        self.kernel_gradient = kernel_gradient
        self.wall_density = wall_density
        self.wall_noslip = wall_noslip

    def W(self):
        rr = self.r[..., 0] if self.dim == 1 and self.r.ndim > 1 else self.r
        return self.kernel(rr, self.h, self.dim)

    def gradW(self):
        return self.kernel_gradient(self.r, self.h, self.dim)

    def neighbours(self, x):
        boxsize = tuple(self.L[axis] if axis in self.periodic_axes 
                    else 0. for axis in range(self.dim))
        xt = x.copy()
        for axis in self.periodic_axes:
            xt[:, axis] = np.mod(xt[:, axis], self.L[axis])
            xt[xt[:, axis] >= self.L[axis], axis] = 0.

        pairs = cKDTree(xt, boxsize=boxsize) \
                    .query_pairs(self.support, output_type="ndarray")
        return pairs[:, 0], pairs[:, 1]

<<<<<<< HEAD
    def treat_walls(self, pres, dens):
        # Adami self.wall pressure/density
        wall_fluid_pair  = self.wall[self.i] != self.wall[self.j]          # self.wall-fluid pairs
        self.Ws  = self.Wij[wall_fluid_pair] # kernel for fluid - self.wall interaction
        self.wall_particle_index   = np.where(self.wall[self.i], self.i, self.j)[wall_fluid_pair] # self.wall particle index 
        self.fluid_particle_index  = np.where(self.wall[self.i], self.j, self.i)[wall_fluid_pair] # fluid particle index
        rwf = np.where(self.wall[self.i], 1., -1.)[wall_fluid_pair, None]*self.r[wall_fluid_pair] # relative position
        pres_wall = pres[self.fluid_particle_index]*self.Ws + np.sum(self.accel*rwf, axis=1)*self.rho[self.fluid_particle_index]*self.Ws        
=======
    def treat_walls(self, W, n, rho, pres, dens):
        # Adami self.wall pressure/density
        wall_fluid_pair  = self.wall[self.i] != self.wall[self.j]          # self.wall-fluid pairs
        Ws  = W[wall_fluid_pair] # kernel for fluid - self.wall interaction
        wall_particle_index   = np.where(self.wall[self.i], self.i, self.j)[wall_fluid_pair] # self.wall particle index 
        fluid_particle_index  = np.where(self.wall[self.i], self.j, self.i)[wall_fluid_pair] # fluid particle index
        rwf = np.where(self.wall[self.i], 1., -1.)[wall_fluid_pair, None]*self.r[wall_fluid_pair] # relative position
        pres_wall = pres[fluid_particle_index]*Ws + np.sum(self.accel*rwf, axis=1)*rho[fluid_particle_index]*Ws        
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a
                        # first order pressure extrapolation
        self.sw  = accumulate_contribs(self.wall_particle_index, self.fluid_particle_index, self.Ws, np.zeros_like(self.Ws), self.n)
        num = accumulate_contribs(self.wall_particle_index, self.fluid_particle_index, pres_wall, np.zeros_like(self.Ws), self.n)

        self.valid_wall_particle = self.wall & (self.sw > 0)
        pw = np.zeros(self.n)
        # weighted average pressure
        pw[self.valid_wall_particle] = num[self.valid_wall_particle]/self.sw[self.valid_wall_particle]

        pres[self.wall] = pw[self.wall]
        dens[self.wall] = self.rho0*np.maximum(1 + pw[self.wall]/self.B, 1e-3)**(1/self.gm)

        if self.wall_density is not None:
            rho_w = np.asarray(self.wall_density(self.x[self.wall]), dtype=float)
            p_w = tait(self.B, self.rho0, 
                        np.where(np.isfinite(rho_w), rho_w, self.rho0), self.gm)
            dens[self.wall] = np.where(np.isfinite(rho_w), rho_w, dens[self.wall])
            pres[self.wall] = np.where(np.isfinite(rho_w), p_w, pres[self.wall])
        return pres, dens

<<<<<<< HEAD
    def density_diffusion(self, dens, drho):
        ff  = ~self.wall[self.i] & ~self.wall[self.j]
        ref = self.rho_ref(self.x)
        psi = (dens[self.i] - dens[self.j]) - (ref[self.i] - ref[self.j])
        D   = (2*self.delta*self.h*self.c0*psi*np.sum(self.dWij*self.r, axis=1)
                / np.maximum(self.rn**2, 1e-30)*ff)

        drho += accumulate_contribs(
            self.i, self.j, self.mass/dens[self.j]*D, -self.mass/dens[self.i]*D, self.n)
=======
    def density_diffusion(self, x, dens, n, rn, dW, drho):
        ff  = ~self.wall[self.i] & ~self.wall[self.j]
        ref = self.rho_ref(x)
        psi = (dens[self.i] - dens[self.j]) - (ref[self.i] - ref[self.j])
        D   = (2*self.delta*self.h*self.c0*psi*np.sum(dW*self.r, axis=1)
                / np.maximum(rn**2, 1e-30)*ff)

        drho += accumulate_contribs(
            self.i, self.j, self.mass/dens[self.j]*D, -self.mass/dens[self.i]*D, n)
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a

        return drho

    def equations(self, x, v, rho, t):
        self.x = x; self.rho = rho
        n = self.n = len(rho)
        if self.wall is None:
            self.wall = np.zeros(n, dtype=bool)

        self.i, self.j = self.neighbours(x)
        self.r = x[self.i] - x[self.j]
        for axis in self.periodic_axes:
            self.r[:, axis] -= self.L[axis]*np.round(self.r[:, axis]/self.L[axis])

<<<<<<< HEAD
        self.rn = np.linalg.norm(self.r, axis=1)
        self.Wij, self.dWij = self.W(self.r), self.gradW(self.r)
=======
        rn = np.linalg.norm(self.r, axis=1)
        W, dW = self.W(), self.gradW()
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a

        # equation of state: pres from rho
        pres_0 = tait(self.B, self.rho0, rho, self.gm)
        dens_0 = rho.copy()

<<<<<<< HEAD
        pres, dens = self.treat_walls(pres_0, dens_0)
=======
        pres, dens = self.treat_walls(W, n, rho, pres_0, dens_0)
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a

        veff = v
        if self.wall_noslip:
            vw = accumulate_contribs(self.wall_particle_index, self.fluid_particle_index, v[self.fluid_particle_index, :]*self.Ws[:, None], np.zeros_like(v[self.fluid_particle_index, :]), n)
            veff = v.copy()
            veff[self.valid_wall_particle] = -vw[self.valid_wall_particle]/self.sw[self.valid_wall_particle, None]

        # Continuity eq.
        vij = v[self.i] - v[self.j]
<<<<<<< HEAD
        c = self.mass*np.sum(vij*self.dWij, axis=1)
        drho = accumulate_contribs(self.i, self.j, c, c, n)

        if self.delta > 0:
            # Delta-SPH density diffusion, can be done depending on a ref. profile.
            drho = self.density_diffusion(dens, drho)
=======
        c = self.mass*np.sum(vij*dW, axis=1)
        drho = accumulate_contribs(self.i, self.j, c, c, n)


        if self.delta > 0:
            # Delta-SPH density diffusion, can be done depending on a ref. profile.
            drho = self.density_diffusion(x, dens, n, rn, dW, drho)
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a
        
        # Momentum equation
        vv = veff[self.i] - veff[self.j]
        Pi = 0.
        if self.viscosity == "artificial":
            vr = np.sum(vv*self.r, axis=1)
<<<<<<< HEAD
            mu = self.h*vr/(self.rn**2 + 0.01*self.h**2)
=======
            mu = self.h*vr/(rn**2 + 0.01*self.h**2)
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a
            Pi = np.where(vr < 0,
                -self.alpha*self.c0*mu/(0.5*(dens[self.i] + dens[self.j])), 0.
            )

<<<<<<< HEAD
        f = -(self.mass*(pres[self.i]/dens[self.i]**2 + pres[self.j]/dens[self.j]**2 + Pi))[:, None]*self.dWij
        if self.viscosity == "laminar":
            K = (self.mass*4*self.nu_phys*np.sum(self.r*self.dWij, axis=1)
                 / ((dens[self.i] + dens[self.j])*(self.rn**2 + 0.01*self.h**2)))
=======
        f = -(self.mass*(pres[self.i]/dens[self.i]**2 + pres[self.j]/dens[self.j]**2 + Pi))[:, None]*dW
        if self.viscosity == "laminar":
            K = (self.mass*4*self.nu_phys*np.sum(self.r*dW, axis=1)
                 / ((dens[self.i] + dens[self.j])*(rn**2 + 0.01*self.h**2)))
>>>>>>> 3443ee4ae347d91213f60074ed775aecbf12e00a
            f += K[:, None]*vv

        acc = accumulate_contribs(self.i, self.j, f, -f, n) + self.accel
        
        # fixed walls
        acc[self.wall] = 0.
        drho[self.wall] = 0.
        return acc, drho

    def step(self, x, v, rho, dt, t=0.0):
        return self.stepper(self.equations, x, v, rho, dt, t)

    def run(self, x, v, rho, wall=None, T=None, save_every=10):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if wall is not None:
            self.wall = np.asarray(wall, dtype=bool)
        if self.wall is None:
            self.wall = np.zeros(len(rho), dtype=bool)
        if self.wall.shape != rho.shape:
            raise ValueError(f"wall must have shape {rho.shape}, got {self.wall.shape}")

        T = self.T if T is None else T
        ns = int(np.ceil(T/self.dt))
        saved = np.unique(np.append(np.arange(0, ns + 1, save_every), ns))
        to_save = set(saved.tolist())

        xs, vs, rs = [], [], []
        for n in tqdm(range(ns + 1)):
            if n in to_save:
                xs.append(x.copy())
                vs.append(v.copy())
                rs.append(rho.copy())
            if n < ns:
                x, v, rho = self.step(x, v, rho, self.dt, n*self.dt)

        return dict(t=saved*self.dt, x=np.array(xs), v=np.array(vs), 
                                rho=np.array(rs), wall=self.wall, dt=self.dt)