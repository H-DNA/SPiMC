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
  int _size;
  MPI_Comm _comm;

  // enqueuer-use only
  bclx::gptr<segment_t> _tail_segment;
  bclx::gptr<segment_t> _e_head_segment;
  bclx::gptr<bclx::gptr<segment_t>>
      *_e_hazard_pointers; // 2 hazard pointers each for each dequeuer

  // shared
  bclx::gptr<bclx::gptr<segment_t>> _d_head_segment;
  bclx::gptr<bclx::gptr<segment_t>> _d_hazard_pointers; // 2 hazard pointers

public:
  SegmentQueue(MPI_Aint enqueuer_rank, MPI_Comm comm)
      : _enqueuer_rank{enqueuer_rank}, _comm{comm} {
    MPI_Comm_rank(comm, &this->_self_rank);
    MPI_Comm_size(comm, &this->_size);

    if (this->_self_rank == enqueuer_rank) {
      bclx::gptr<segment_t> empty_segment = BCL::alloc<segment_t>(1);
      this->_tail_segment = empty_segment;
      this->_e_head_segment = empty_segment;
      this->_d_head_segment = BCL::alloc<bclx::gptr<segment_t>>(1);
      *this->_d_head_segment.local() = empty_segment;
      BCL::broadcast(this->_d_head_segment, enqueuer_rank);

      this->_d_hazard_pointers = BCL::alloc<bclx::gptr<segment_t>>(2);
      this->_d_hazard_pointers.local()[0] = nullptr;
      this->_d_hazard_pointers.local()[1] = nullptr;
      this->_e_hazard_pointers =
          new bclx::gptr<bclx::gptr<segment_t>>[this->_size];
      for (int i = 0; i < this->_size; ++i) {
        this->_e_hazard_pointers[i] =
            BCL::broadcast(this->_d_hazard_pointers, i);
      }
    } else {
      this->_tail_segment = nullptr;
      this->_e_head_segment = nullptr;
      this->_d_head_segment =
          BCL::broadcast(this->_d_head_segment, enqueuer_rank);

      this->_e_hazard_pointers = nullptr;
      this->_d_hazard_pointers = BCL::alloc<bclx::gptr<segment_t>>(2);
      this->_d_hazard_pointers.local()[0] = nullptr;
      this->_d_hazard_pointers.local()[1] = nullptr;
      for (int i = 0; i < this->_size; ++i) {
        BCL::broadcast(this->_d_hazard_pointers, i);
      }
    }
  }

  SegmentQueue(SegmentQueue &&other) noexcept
      : _self_rank{other._self_rank}, _size{other._size},
        _enqueuer_rank{other._enqueuer_rank}, _comm{other._comm},
        _tail_segment{other._tail_segment},
        _e_head_segment{other._e_head_segment},
        _d_head_segment{other._d_head_segment},
        _e_hazard_pointers{other._e_hazard_pointers},
        _d_hazard_pointers{other._d_hazard_pointers} {
    other._tail_segment = nullptr;
    other._e_head_segment = nullptr;
    other._d_head_segment = nullptr;
    other._d_hazard_pointers = nullptr;
    other._e_hazard_pointers = nullptr;
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

private:
  markable_gptr<segment_t>
  _reserve_if_not_marked(bclx::gptr<markable_gptr<segment_t>> ptr,
                         int hazard_slot) {
    markable_gptr<segment_t> m_ptr = bclx::aget_sync(ptr);
    if (get_marker(m_ptr)) {
      return m_ptr;
    }
    if (get_ptr(m_ptr) == nullptr) {
      return m_ptr;
    }
    bclx::aput_sync(get_ptr(m_ptr), this->_d_hazard_pointers + hazard_slot);
    markable_gptr<segment_t> m_ptr_again;
    while ((m_ptr_again = bclx::aget_sync(ptr)) != m_ptr) {
      m_ptr = m_ptr_again;
      if (get_marker(m_ptr)) {
        bclx::aput_sync(nullptr, this->_d_hazard_pointers + hazard_slot);
        return m_ptr;
      }
      bclx::aput_sync(get_ptr(m_ptr), this->_d_hazard_pointers + hazard_slot);
      if (get_ptr(m_ptr) == nullptr) {
        return m_ptr;
      }
    }
    return m_ptr;
  }

  void _release_hazard_pointer(int hazard_slot) {
    bclx::aput_sync(nullptr, this->_d_hazard_pointers + hazard_slot);
  }
};
