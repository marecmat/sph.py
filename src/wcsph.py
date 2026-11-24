import numpy as np
from scipy.spatial import cKDTree

from .eqs import scatter, tait
from .kernels import wendland, wendland_gradient
from .time_integration import symplectic, rk4

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

class SPHSolver:

    def __init__(
        self, *, dim, dx, h_factor=1.3, rho0=1000., gm=7.,
        c0=None, H=None, drho_max=0.01, g=9.81, body_force=0., accel=None,
        viscosity="artificial", alpha=0.1, nu_phys=0., delta=0.,
        rho_ref=None, wall_density=None, wall_noslip=False,
        stepper="symplectic", cfl=0.25,
        T=0.5, periodic_axes=(), L=None,
        kernel=wendland, kernel_gradient=wendland_gradient,
    ):
        if dim not in (1, 2):
            raise ValueError("only dim=1 and dim=2 are supported")
        if c0 is None:
            if H is None:
                raise ValueError("pass c0 or H to set the numerical speed of sound")
            c0 = np.sqrt(g*H/drho_max)


        self.dim = dim
        self.dx = float(dx)
        self.h = h_factor*self.dx
        self.support = 2*self.h
        self.rho0 = float(rho0)
        self.gm = float(gm)
        self.c0 = float(c0)
        self.B = self.rho0*self.c0**2/self.gm
        self.T = float(T)
        self.cfl = float(cfl)

        self.mass = self.rho0*self.dx**dim
        self.viscosity = viscosity
        self.alpha = float(alpha)
        self.nu_phys = float(nu_phys)

        if viscosity not in ("artificial", "laminar"):
            raise ValueError(f"unknown viscosity: {viscosity}")

        self.nu = (self.nu_phys if viscosity == "laminar"
                   else self.alpha*self.h*self.c0/8)
        self.delta = float(delta)
        self.rho_ref = rho_ref
        self.wall_density = wall_density
        self.wall_noslip = bool(wall_noslip)
        
        if stepper == "symplectic":
            self.stepper = symplectic
        elif stepper == "rk4":
            self.stepper = rk4
        else:
            raise ValueError(f"unknown stepper: {stepper}")

        self.accel = self._body_acceleration(g, body_force, accel)
        self.dt = self.cfl*min(self._time_limits())
        self.n_steps = int(np.ceil(self.T/self.dt))

        self.periodic_axes = tuple(periodic_axes)
        self.L = tuple(L) if L is not None else tuple(None for _ in range(dim))
        for axis in self.periodic_axes:
            if not 0 <= axis < dim:
                raise ValueError(f"invalid periodic axis {axis}")
            if self.L[axis] is None:
                raise ValueError(f"missing domain length for periodic axis {axis}")

        self.kernel = kernel
        self.kernel_gradient = kernel_gradient
        self.wall = None

    def _body_acceleration(self, g, body_force, accel):
        if accel is not None:
            a = np.asarray(accel, dtype=float)
            if a.shape != (self.dim,):
                raise ValueError(f"accel must have shape ({self.dim},)")
            return a

        a = np.zeros(self.dim)
        a[-1] = -g                       # last coordinate is vertical
        a[0] += body_force               # body force drives the first coordinate
        return a

    def _time_limits(self):
        limits = [self.h/self.c0]
        if self.nu > 0:
            limits.append(self.h**2/self.nu)
        f = np.linalg.norm(self.accel)
        if f > 0:
            limits.append(np.sqrt(self.h/f))
        return limits

    def W(self, r):
        rr = np.asarray(r)
        if self.dim == 1 and rr.ndim > 1:
            rr = rr[..., 0]
        return self.kernel(rr, self.h, self.dim)

    def gradW(self, r):
        return self.kernel_gradient(r, self.h, self.dim)

    def neighbours(self, x):
        """Return undirected pair indices with periodic images handled."""
        boxsize = tuple(
            self.L[axis] if axis in self.periodic_axes else 0.
            for axis in range(self.dim))
        xt = x.copy()
        for axis in self.periodic_axes:
            xt[:, axis] = np.mod(xt[:, axis], self.L[axis])
            xt[xt[:, axis] >= self.L[axis], axis] = 0.

        pairs = cKDTree(xt, boxsize=boxsize) \
                    .query_pairs(self.support, output_type="ndarray")
        return pairs[:, 0], pairs[:, 1]

    def equations(self, x, v, rho, t):
        n = len(rho)
        wall = self.wall if self.wall is not None else np.zeros(n, dtype=bool)

        i, j = self.neighbours(x)
        r = x[i] - x[j]
        for axis in self.periodic_axes:
            r[:, axis] -= self.L[axis]*np.round(r[:, axis]/self.L[axis])

        rn = np.linalg.norm(r, axis=1)
        W, dW = self.W(r), self.gradW(r)
        pres, dens = tait(self.B, self.rho0, rho, self.gm), rho.copy()

        # Adami-style wall pressure/density from fluid neighbours.
        wf = wall[i] != wall[j]
        iw = np.where(wall[i], i, j)[wf]
        jf = np.where(wall[i], j, i)[wf]
        rwf = np.where(wall[i], 1., -1.)[wf, None]*r[wf]
        Ws = W[wf]

        sw = np.bincount(iw, Ws, n)
        num = np.bincount(iw, pres[jf]*Ws, n) \
              + np.bincount(iw, np.sum(self.accel*rwf, axis=1)*rho[jf]*Ws, n)
        ok = wall & (sw > 0)
        pw = np.zeros(n)
        pw[ok] = num[ok]/sw[ok]

        pres[wall] = pw[wall]
        dens[wall] = self.rho0*np.maximum(
            1 + pw[wall]/self.B, 1e-3
        )**(1/self.gm)

        # Override selected wall particles (e.g. top ghosts) with a prescribed
        # density.  NaN in the returned array means "use the Adami value".
        if self.wall_density is not None:
            rho_w = np.asarray(self.wall_density(x[wall]), dtype=float)
            given = np.isfinite(rho_w)
            p_w = tait(self.B, self.rho0, np.where(given, rho_w, self.rho0), self.gm)
            dens[wall] = np.where(given, rho_w, dens[wall])
            pres[wall] = np.where(given, p_w, pres[wall])

        veff = v
        if self.wall_noslip:
            vw = np.stack(
                [np.bincount(iw, v[jf, a]*Ws, n) for a in range(self.dim)],
                axis=1,
            )
            veff = v.copy()
            veff[ok] = -vw[ok]/sw[ok, None]

        # Continuity.
        vij = v[i] - v[j]
        c = self.mass*np.sum(vij*dW, axis=1)
        drho = scatter(i, j, c, c, n)

        # Delta-SPH density diffusion, optionally around a reference profile.
        if self.delta > 0:
            ff = ~wall[i] & ~wall[j]
            ref = self.rho_ref(x) if self.rho_ref is not None \
                  else np.full(n, self.rho0)
            psi = (dens[i] - dens[j]) - (ref[i] - ref[j])
            D = (2*self.delta*self.h*self.c0*psi*np.sum(dW*r, axis=1)
                 / np.maximum(rn**2, 1e-30)*ff)
            drho += np.bincount(i, self.mass/dens[j]*D, n) \
                    - np.bincount(j, self.mass/dens[i]*D, n)

        # Momentum: pressure, artificial/laminar viscosity.
        vv = veff[i] - veff[j]
        Pi = 0.
        if self.viscosity == "artificial":
            vr = np.sum(vv*r, axis=1)
            mu = self.h*vr/(rn**2 + 0.01*self.h**2)
            Pi = np.where(
                vr < 0,
                -self.alpha*self.c0*mu/(0.5*(dens[i] + dens[j])),
                0.,
            )

        f = -(self.mass*(pres[i]/dens[i]**2 + pres[j]/dens[j]**2 + Pi))[:, None]*dW
        if self.viscosity == "laminar":
            K = (self.mass*4*self.nu_phys*np.sum(r*dW, axis=1)
                 / ((dens[i] + dens[j])*(rn**2 + 0.01*self.h**2)))
            f += K[:, None]*vv

        acc = scatter(i, j, f, -f, n) + self.accel
        acc[wall] = 0.
        drho[wall] = 0.
        return acc, drho

    def step(self, x, v, rho, dt, t=0.0):
        return self.stepper(self.equations, x, v, rho, dt, t)

    def run(self, x, v, rho, wall=None, T=None, save_every=10):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if wall is None:
            wall = np.zeros(len(rho), dtype=bool)
        else:
            wall = np.asarray(wall, dtype=bool)
        self.wall = wall

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

        return dict(
            t=saved*self.dt,
            x=np.array(xs),
            v=np.array(vs),
            rho=np.array(rs),
            wall=wall,
            dt=self.dt,
        )
