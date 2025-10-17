#pragma once

#include <bclx/bclx.hpp>
#include <cstdint>
#include <mpi.h>

#include "bcl/backends/mpi/backend.hpp"
#include "bcl/backends/mpi/comm.hpp"
#include "bcl/core/alloc.hpp"
#include "bcl/core/teams.hpp"
#include "bclx/core/comm.hpp"
#include "bclx/core/definition.hpp"

template <typename T> struct markable_gptr {
  bclx::gptr<T> _inner;
};

template <typename T> constexpr bool get_marker(markable_gptr<T> ptr) {
  return ptr._inner & (uint64_t)1;
}

template <typename T> constexpr bclx::gptr<T> get_ptr(markable_gptr<T> ptr) {
  return ptr._inner & ~(uint64_t)1;
}

template <typename data_t> class SegmentQueue {
  static constexpr MPI_Aint SEGMENT_CAPACITY = 2048;

  struct segment_t {
    uint64_t head;
    uint64_t tail;
    markable_gptr<segment_t> next;
    bclx::gptr<data_t> data[SEGMENT_CAPACITY];
  };

  int _self_rank;
  const MPI_Aint _enqueuer_rank;
  MPI_Comm _comm;

  // enqueuer-use only
  bclx::gptr<segment_t> _tail_segment;
  bclx::gptr<segment_t> _e_head_segment;

  // shared
  bclx::gptr<bclx::gptr<segment_t>> _d_head_segment;

public:
  SegmentQueue(MPI_Aint enqueuer_rank, MPI_Comm comm)
      : _enqueuer_rank{enqueuer_rank}, _comm{comm} {
    MPI_Comm_rank(comm, &this->_self_rank);

    if (this->_self_rank == enqueuer_rank) {
      bclx::gptr<segment_t> empty_segment = BCL::alloc<segment_t>(1);
      this->_tail_segment = empty_segment;
      this->_e_head_segment = empty_segment;
      this->_d_head_segment = BCL::alloc<bclx::gptr<segment_t>>(1);
      *this->_d_head_segment.local() = empty_segment;
      BCL::broadcast(this->_d_head_segment, enqueuer_rank);
    } else {
      this->_tail_segment = nullptr;
      this->_e_head_segment = nullptr;
      this->_d_head_segment =
          BCL::broadcast(this->_d_head_segment, enqueuer_rank);
    }
  }

  SegmentQueue(SegmentQueue &&other) noexcept
      : _self_rank{other._self_rank}, _enqueuer_rank{other._enqueuer_rank},
        _comm{other._comm}, _tail_segment{other._tail_segment},
        _e_head_segment{other._e_head_segment},
        _d_head_segment{other._d_head_segment} {
    other._tail_segment = nullptr;
    other._e_head_segment = nullptr;
    other._d_head_segment = nullptr;
    other._comm = MPI_COMM_NULL;
  }

  SegmentQueue(const SegmentQueue &) = delete;
  SegmentQueue &operator=(const SegmentQueue &) = delete;
  SegmentQueue &operator=(SegmentQueue &&) = delete;

  ~SegmentQueue() {
    // free later
  }

  bool enqueue(const data_t &data) {}

  bool dequeue(data_t *output) {}
};
