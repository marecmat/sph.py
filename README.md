# My own implementation of SPH in Python

Implementation of the core equations with delta-SPH (density diffusion) and a few boundary conditions

- Minimal features, developed to understand the method and get to use DualSPHyics smoothly 
- Code should kind of understandable
- Will only support selected test cases in 1D and 2D : hydrostatics, dam break and Poiseuille flow tests

## TODO 
- [x] toy model with 1D hydrostatics
- [x] core code refactored to prepare 2D tests
- [ ] finish the README
- [ ] validate 2D hydrostatics
- [ ] validate Poiseuille flow to analytics to DualSPHysics
- [ ] validate dam break 2D to DualSPHysics