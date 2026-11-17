# Learning SPH

- Implementation of elementary bricks of the method to gain insight on the workings of DualSPHysics

# Preliminary tests 
- Partition of unity test: the kernel is normalized such that,
$$
\int W(\boldsymbol{r}', h) dv' = 1,
$$
and the particle sum, 
$$
\sum_j V_j W(x_i - x_j, h) \approx 1.
$$

A field $f = 1$ should be properly retrieved, and consist at zeroth-order, $\tilde f \longrightarrow f$ through,
$$
\tilde{f}(x_i) = \sum_j V_j f_j W_{ij} = 1,
$$ 
isolating the interpolation weights and ensuring that, 
$$
\sum_j V W_{ij} = 1
$$

- gradient check: discrete discretization, requires the kernel gradient (analytical expression)
$$
\nabla A_i \approx \sum_j A_j \frac{m_j}{\rho_j} \nabla W_{ij}
$$


# Done 
- gradient check
- partition of unity test 
- neighbour search using `scipy.spatial.cKDTree`
- ghost particles: enforce boundary conditions: periodic, and dirichlet 

# TODO 
- implement fluid equations
- simple hydrostatic 
- symplectic time integration scheme