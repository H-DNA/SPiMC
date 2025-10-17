#pragma once

#include <bclx/bclx.hpp>
#include <mpi.h>

#include "bcl/backends/mpi/backend.hpp"
#include "bcl/backends/mpi/comm.hpp"
#include "bcl/core/alloc.hpp"
#include "bcl/core/teams.hpp"
#include "bclx/core/comm.hpp"
#include "bclx/core/definition.hpp"

template <typename data_t> class SegmentQueue {
  int _self_rank;
  const MPI_Aint _enqueuer_rank;

public:
  SegmentQueue(MPI_Aint enqueuer_rank, MPI_Comm comm)
      : _enqueuer_rank{enqueuer_rank} {}

  SegmentQueue(SegmentQueue &&other) noexcept {}

  SegmentQueue(const SegmentQueue &) = delete;
  SegmentQueue &operator=(const SegmentQueue &) = delete;
  SegmentQueue &operator=(SegmentQueue &&) = delete;

  ~SegmentQueue() {
    // free later
  }

  bool enqueue(const data_t &data) {}

  bool dequeue(data_t *output) {}
};
