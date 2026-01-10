# Porting shared memory SPMC queues to distributed context using MPI-3 RMA

![SPiMC](https://img.shields.io/badge/SPiMC-blue?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsMTAgNSAxMC01TTIgMTJsMTAgNSAxMC01Ii8+PC9zdmc+) ![Status](https://img.shields.io/badge/status-prototype-yellow)

This project ports lock-free Single-Producer Multiple-Consumer (SPMC) queue algorithms from shared-memory to distributed systems using MPI-3 Remote Memory Access (RMA).

## Table of Contents

- [Overview](#overview)
- [Related Work](#related-work)

## Overview

SPMC queues are the dual of MPSC queues—a single producer feeds data to multiple consumers. This pattern is common in work distribution scenarios where one coordinator dispatches tasks to a pool of workers.

This project prototypes a porting direction for a student group studying SPMC queues. It claims no correctness guarantee. The approach mirrors the MPSC porting effort, leveraging MPI-3 RMA to translate shared-memory algorithms to distributed contexts.

## Related Work

See [MPiSC](../MPiSC) for the completed MPSC queue porting project, which uses the same methodology and serves as the foundation for this work.
