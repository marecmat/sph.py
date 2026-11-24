
import numpy as np
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from utility import scatter

class SPHSolver:

    def __init__(
        self,
        H=1., N=50, Nx=8, n_wall=4, walls="bottom", h_factor=1.3,
        rho0=1000., gm=7., g=9.81, body_force=0., c0=None, drho_max=.01,
        viscosity="artificial", alpha=.1, nu_phys=0., delta=.1,
        start="hydrostatic", hydrostatic_correction=True,
        wall_noslip=False, stepper="symplectic", cfl=.25, T=.5
    ):

        self.H, self.N, self.Nx = H, N, Nx
        self.n_wall, self.walls, self.h_factor = n_wall, walls, h_factor
        self.rho0, self.gm, self.g = rho0, gm, g
        self.body_force = body_force
        self.c0 = np.sqrt(g*H/drho_max) if c0 is None else c0
        self.viscosity, self.alpha, self.nu_phys = viscosity, alpha, nu_phys
        self.delta = delta
        self.start, self.hydrostatic_correction = start, hydrostatic_correction
        self.wall_noslip, self.stepper = wall_noslip, stepper
        self.cfl, self.T = cfl, T

        assert self.c0 > 0
        assert walls in ("bottom", "both")
        assert viscosity in ("artificial", "laminar")
        assert start in ("hydrostatic", "uniform")
        assert stepper in ("symplectic", "rk4")
        assert Nx >= 2*self.support/self.dy, "Strip too narrow"

        rows = [True]*n_wall + [False]*N
        if walls == "both": rows += [True]*n_wall
        self.wall = np.repeat(rows, Nx)

        self.dy = self.H/self.N
        self.h = self.h_factor*self.dy
        self.support = 2*self.h
        self.Lx = self.Nx*self.dy
        self.mass = self.rho0*self.dy**2
        self.B = self.rho0*self.c0**2/self.gm
        if self.viscosity == "laminar":
            self.nu = self.nu_phys  
        else:
            self.nu = self.alpha*self.h*self.c0/8

        self.accel = np.array([self.body_force, -self.g])
        
        limits = [self.h/self.c0]
        if self.nu > 0: 
            limits.append(self.h**2/self.nu)
        f = np.linalg.norm(self.accel)
        if f > 0: 
            limits.append(np.sqrt(self.h/f))
        self.dt = self.cfl*min(limits)

        self.n_steps = int(np.ceil(self.T/self.dt))
        self.kp = (self.gm - 1)*self.rho0*self.g/(self.B*self.gm)

        ub = (1 + self.H*self.kp*self.gm/(self.gm - 1))**((self.gm - 1)/self.gm)
        self.ys = (ub-1)/self.kp

        self.W = kernel
        self.gradW = kernel_gradient

    def pressure(self, y):
        return self.B*((1+self.kp*(self.ys-y))**(self.gm/(self.gm-1))-1)

    def rho_hydro(self, y):
        return self.rho0*(1+self.pressure(y)/self.B)**(1/self.gm)

    def tait(self, rho):
        return self.B*((rho/self.rho0)**self.gm-1)

    def particles(self):
        k = np.arange(self.N)

        if self.start == "uniform":
            yf = (k+.5)*self.dy
            rf = np.full(self.N, self.rho0)

        else:
            assert self.walls == "bottom" and self.g > 0
            u = (1+self.kp*self.gm*(self.N-1-k+.5)*self.dy/(self.gm-1))**((self.gm-1)/self.gm)
            yf = self.ys-(u-1)/self.kp
            rf = self.rho0*u**(1/(self.gm-1))

        y = [-(np.arange(self.n_wall)+.5)*self.dy, yf]
        rho = [np.full(self.n_wall, self.rho0), rf]

        if self.walls == "both":
            y.append(self.H+(np.arange(self.n_wall)+.5)*self.dy)
            rho.append(np.full(self.n_wall, self.rho0))

        y, rho = np.concatenate(y), np.concatenate(rho)
        xc = (np.arange(self.Nx)+.5)*self.dy

        x = np.column_stack([np.tile(xc, len(y)), np.repeat(y, self.Nx)])
        return x, np.zeros_like(x), np.repeat(rho, self.Nx)

    def neighbours(self, x):
        xt = x.copy()
        xt[:, 0] = np.mod(xt[:, 0], self.Lx)
        xt[xt[:, 0] >= self.Lx, 0] = 0.
        pairs = cKDTree(xt, boxsize=[self.Lx, 0.]) \
                    .query_pairs(self.support)
        return pairs[:, 0], pairs[:, 1]

    def equations(self, x, v, rho):
        n, h, m = len(rho), self.h, self.mass
        i, j = self.neighbours(x)

        r = x[i] - x[j]
        r[:, 0] -= self.Lx*np.round(r[:, 0]/self.Lx)

        rn = np.linalg.norm(r, axis=1)
        W, dW = self.W(rn), self.gradW(r)

        pres, dens = self.tait(rho), rho.copy()

        # Wall pressure/density (Adami et al.)
        wf = self.wall[i] != self.wall[j]
        iw = np.where(self.wall[i], i, j)[wf]
        jf = np.where(self.wall[i], j, i)[wf]
        rwf = np.where(self.wall[i], 1., -1.)[wf, None]*r[wf]
        Ws = W[wf]

        sw = np.bincount(iw, Ws, n)
        num = np.bincount(iw, pres[jf]*Ws, n) \
                + np.bincount(iw, np.sum(self.accel*rwf, axis=1)*rho[jf]*Ws, n)

        ok = self.wall & (sw > 0)
        pw = np.zeros(n)
        pw[ok] = num[ok]/sw[ok]

        pres[self.wall] = pw[self.wall]
        dens[self.wall] = self.rho0*np.maximum(1+pw[self.wall]\
                                        /self.B, 1e-3)**(1/self.gm)

        veff = v
        if self.wall_noslip:
            vw = np.stack([np.bincount(iw, v[jf, a]*Ws, n) for a in range(2)], axis=1)
            veff = v.copy()
            veff[ok] = -vw[ok]/sw[ok, None]

        # Continuity
        vij = v[i]-v[j]
        c = m*np.sum(vij*dW, axis=1)
        drho = self.scatter(i, j, c, c, n)

        # Delta-SPH
        if self.delta > 0:
            ff = ~self.wall[i] & ~self.wall[j]
            rh = self.rho_hydro(x[:, 1]) if self.hydrostatic_correction else np.full(n, self.rho0)
            psi = (dens[i]-dens[j])-(rh[i]-rh[j])
            D = 2*self.delta*h*self.c0*psi*np.sum(dW*r, axis=1)/np.maximum(rn**2, 1e-30)*ff
            drho += np.bincount(i, m/dens[j]*D, n)-np.bincount(j, m/dens[i]*D, n)

        # Momentum: pressure + viscosity
        vv = veff[i]-veff[j]
        Pi = 0.

        if self.viscosity == "artificial":
            vr = np.sum(vv*r, axis=1)
            mu = h*vr/(rn**2+.01*h**2)
            Pi = np.where(vr < 0, -self.alpha*self.c0*mu/(.5*(dens[i]+dens[j])), 0.)

        f = -(m*(pres[i]/dens[i]**2+pres[j]/dens[j]**2+Pi))[:, None]*dW

        if self.viscosity == "laminar":
            K = m*4*self.nu_phys*np.sum(r*dW, axis=1)/((dens[i]+dens[j])*(rn**2+.01*h**2))
            f += K[:, None]*vv

        acc = self.scatter(i, j, f, -f, n)+self.accel
        acc[self.wall] = 0.
        drho[self.wall] = 0.

        return acc, drho

    # def symplectic(self, x, v, rho, dt):
    #     a, d = self.equations(x, v, rho)
    #     vh, xh, rh = v+.5*dt*a, x+.5*dt*v, rho+.5*dt*d
    #     ah, dh = self.equations(xh, vh, rh)
    #     eps = -dh/rh
    #     rn = rho*(2-eps*dt)/(2+eps*dt)
    #     vn = v+dt*ah
    #     xn = xh+.5*dt*vn
    #     return xn, vn, rn

    # def step(self, x, v, rho, dt):
    #     if self.stepper == "symplectic":
    #         return self.symplectic(x, v, rho, dt)
    #     return self.rk4(x, v, rho, dt)

    def run(self, save_every=10):
        x, v, rho = self.particles()
        dt, ns = self.dt, self.n_steps
        saved = np.unique(np.append(np.arange(0, ns+1, save_every), ns))
        xs, vs, rs = [], [], []

        for n in tqdm(range(ns+1)):
            if n in saved:
                xs.append(x.copy())
                vs.append(v.copy())
                rs.append(rho.copy())

            if n < ns:
                x, v, rho = self.step(x, v, rho, dt)

        xs, vs, rs = np.array(xs), np.array(vs), np.array(rs)
        fluid = ~self.wall

        return dict(
            t=saved*dt, x=xs, v=vs, rho=rs, fluid=fluid, dt=dt,
            KE=.5*self.mass*np.sum(vs[:, fluid]**2, axis=(1, 2))
        )

    def poiseuille(cls, H=1., N=20, Nx=8, nu=.1, F=1., T=3., mach=.1, **kw):
        umax = F*H**2/(8*nu)
        opts = dict(
            H=H, N=N, Nx=Nx, T=T, g=0., body_force=F,
            walls="both", start="uniform", h_factor=2**.5,
            n_wall=3, viscosity="laminar", nu_phys=nu,
            c0=umax/mach, hydrostatic_correction=False,
            wall_noslip=True
        )

        opts.update(kw)
        return cls(**opts)

if __name__ == "__main__":
    solver = SPHSolver(N=50, Nx=8, T=0.5)
    out = solver.run(save_every=10)

    print(out.keys())