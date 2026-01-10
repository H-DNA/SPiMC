# Implementations

Prototype SPMC queue implementations using MPI-3 RMA.

## Contents

- [segment_queue](./segment_queue) - Segment-based SPMC queue using marked pointers

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
mpirun -n <num_processes> ./main
```
