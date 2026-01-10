# Segment Queue

A distributed SPMC queue using fixed-size segments organized as a linked list.

## Overview

The queue allocates data in contiguous segments of 2048 elements. When a segment fills, a new segment is allocated and linked. Consumers traverse segments via `next` pointers, incrementing a shared `head` index to claim slots.

## Design

### Structure

```
segment_t {
  head: uint64        // next slot for dequeue (FAA by consumers)
  tail: uint64        // next slot for enqueue (producer only)
  next: markable_ptr  // link to next segment
  data: gptr[2048]    // global pointers to enqueued data
}
```

### Marked Pointers

The `next` pointers use LSB stealing for a marker bit (same idea as Java's [AtomicMarkableReference](https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/atomic/AtomicMarkableReference.html)). The marker indicates the segment is being reclaimed—consumers seeing a marked pointer must retry from the head segment.

### Memory Reclamation

Hazard pointers protect segments during traversal. Each consumer maintains two hazard pointer slots. The producer periodically scans hazard pointers and reclaims unmarked segments not protected by any consumer.

## API

```cpp
SegmentQueue(MPI_Aint enqueuer_rank, MPI_Comm comm)
bool enqueue(const T& data)   // producer only
bool dequeue(T* output)       // consumers only
```

## Diagram

<img width="2540" height="834" alt="Segment queue structure" src="https://github.com/user-attachments/assets/ede0f2a2-bf1e-43ec-9ce9-929ada6a6eb3" />
