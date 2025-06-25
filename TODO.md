## ✅ TODO / Roadmap

A set of key improvements and extensions planned for this project.

### 🔧 1. Memory Optimization and Performance

- [ ] **Implement shared memory usage** in key CUDA kernels:
  - Especially in interface kernels in `lbm_phase.cu` and the core LBM routines in `lbm_core.cu`;
  - Objective: reduce global memory bandwidth usage and improve data locality;
  - Will probably have to implement ghost interfaces to share data between blocks.

- [ ] **Reduce register pressure** in `gpuMomCollisionStream`:
  - Bad workaround implemented via:
    - Compilation flag `--maxrregcount` to limit register usage;
  - **GPU occupancy is still behind by ~30% of its potential.**

### 🧩 2. Codebase Generalization and Modularity

- [ ] **Merge** this repository (`MULTIC-TS-CUDA`) with the `MULTIC-BUBBLE-CUDA` project:
  - Goal: make the codebase general for **multicomponent LBM flows**, regardless of geometry or injection scenario.

- [ ] **Refactor simulation logic** to support multiple case types via:
  - `#define` macros;
  - Encapsulation of case-specific setup (inflow/boundary conditions, initial fields, etc.).

### 🌊 3. Boundary Conditions

- [X] **Implement boundary reconstruction** to enable physical boundary behavior:
  - [x] **Periodic** behavior implemented on lateral faces (`x` and `y` directions);
  - [x] **Outflow** implemented at the domain exit (`z = NZ-1`);
  - [x] **Inflow** implemented explicitly at `z = 0`;
  - [ ] **EXTRA**: prevent outflow from recirculating when too much fluid tries to exit.

### 🔬 4. Physics Extensions

- [ ] **Associate physical properties** to each fluid component:
  - Assign **oil** properties to the injected jet and **water** to the background medium (already possible).

- [ ] **Allow dynamic oil properties**:
  - Parametrize oil characteristics (density, viscosity, surface tension, interface width) for multiple types or API grades;
  - Possibly via external config or compile-time macros.

### 📦 5. Code Usability

- [ ] Move post-processing loop to C++ for better performance and integration:
  - Rewrite `process_steps.py` processing loop to C++ due to the large time needed to process big simulations;
  - Automate variable detection in `get_sim_info.py` based on the `copyAndSaveToBinary` function calls.

---

Feel free to contribute, discuss or pick up tasks from this list!
