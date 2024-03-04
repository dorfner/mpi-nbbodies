# 2D N-Bodies Parallel Simulation with MPI

This repository contains Python implementations of sequential and parallel algorithms for simulating the evolution of a set of bodies interacting with each other in a 2D space. The simulation is based on the N-Bodies problem, commonly found in astrophysics, where the motion of stars is calculated considering gravitational forces.

This project was developed as part of a practical work assignment for my MPI class during the first semester of my master's degree (Data Science degree at University Paul Sabatier of Toulouse). It served as an opportunity to deepen my understanding of MIMD parallelism. You can access the assignement description in the root of this repository.

The instructors provided the initial framework of the project, and only the parts below the line `# Modify only starting here (and in the imports)` have been written by me.

## Overview

The N-Bodies problem involves calculating the forces between pairs of stars and updating their positions, velocities, and accelerations at each time step. The basic solution includes calculating forces, summing up resultant forces, and updating the star data accordingly. The sequential and parallel versions of the algorithm are implemented to simulate this process.

## Features

- Implements sequential and parallel MPI versions of the N-Bodies simulation algorithm.
- Provides options for running simulations with different numbers of bodies and iterations.
- Supports display of simulation results and signature verification for validation.
- Includes utility functions for initialization, force calculation, and result display.

## Usage

```
python3 n-bodies-seq.py <num_bodies> <num_iterations>
```

```
mpirun -n <num_processes> python3 n-bodies-opt.py <num_bodies> <num_iterations>
```

Replace `<num_bodies>` with the number of bodies and `<num_iterations>` with the number of iterations. For the parallel version, specify the number of MPI processes using `-n <num_processes>`. If you add a `-nodisplay` argument, the function to display the stars should not be used.

## Example

Small example using 2 cores :
```
mpirun -n 2 python3 n-bodies-opt.py 12 60
```

Larger example with 4 cores :
```
mpirun -n 4 python3 n-bodies-opt.py 128 60
```

## Given Files

- `n-bodies-base.py`: Base file.
- `n-bodies-test.sh`: Simple shell test.

## Assignement files

- `n-bodies-seq.py`: Sequential implementation of the N-Bodies algorithm.
- `n-bodies-par.py`: Parallel MPI implementation of the N-Bodies algorithm.
- `n-bodies-opt.py`: Optimized parallel MPI version considering symmetrical forces.

## Additional Notes

- This README.md was created with the assistance of ChatGPT 3.5.

- As a learning project, the focus was on applying algorithmic concepts rather than producing perfectly polished code.

- This program was tested on Linux only.
