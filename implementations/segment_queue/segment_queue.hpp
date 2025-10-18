#pragma once

#include <algorithm>
#include <bclx/bclx.hpp>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "bcl/backends/mpi/atomics.hpp"
#include "bcl/backends/mpi/backend.hpp"
#include "bcl/backends/mpi/comm.hpp"
#include "bcl/backends/mpi/ops.hpp"
#include "bcl/core/alloc.hpp"
#include "bcl/core/teams.hpp"
#include "bclx/backends/mpi/comm.hpp"
#include "bclx/core/comm.hpp"
#include "bclx/core/definition.hpp"

template <typename T> struct markable_gptr {
  bclx::gptr<T> _inner;
};

template <typename T> bool get_marker(markable_gptr<T> ptr) {
  return ptr._inner.ptr & (uint64_t)1;
}

template <typename T> bclx::gptr<T> get_ptr(markable_gptr<T> ptr) {
  bclx::gptr<T> res;
  res.ptr = ptr._inner.ptr & ~(uint64_t)1;
  res.rank = ptr._inner.rank;
  return res;
}

template <typename T>
markable_gptr<T> create_markable_ptr(bclx::gptr<T> ptr, bool marker = false) {
  if (marker) {
    ptr.ptr = ptr.ptr | (uint64_t)1;
  }
  return markable_gptr<T>{ptr};
}

template <typename T> bclx::gptr<T> create_nullptr() {
  bclx::gptr<T> ptr;
  ptr.rank = 0;
  ptr.ptr = 0;
  return ptr;
}

template <typename data_t> class SegmentQueue {
  static constexpr MPI_Aint SEGMENT_CAPACITY = 2048;
  inline static const bclx::gptr<data_t> BOTTOM_PTR = nullptr;
  inline static const bclx::gptr<data_t> TOP_PTR = bclx::gptr<data_t>(0, 2);

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
  std::vector<bclx::gptr<segment_t>> _free_list;

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
        bclx::gptr<bclx::gptr<segment_t>> tmp = this->_d_hazard_pointers;
        this->_e_hazard_pointers[i] = BCL::broadcast(tmp, i);
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
        bclx::gptr<bclx::gptr<segment_t>> tmp = this->_d_hazard_pointers;
        BCL::broadcast(tmp, i);
      }
    }
  }

  SegmentQueue(SegmentQueue &&other) noexcept
      : _self_rank{other._self_rank}, _size{other._size},
        _enqueuer_rank{other._enqueuer_rank}, _comm{other._comm},
        _free_list{std::move(other._free_list)},
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

  bool enqueue(const data_t &data) {
    static int enqueue_count = 0;
    ++enqueue_count;

    bclx::gptr<data_t> data_ptr = BCL::alloc<data_t>(1);
    *data_ptr.local() = data;

    uint64_t tail =
        bclx::aget_sync(_get_gptr_to_tail_of_segment(this->_tail_segment));

    bclx::gptr<data_t> value;

    if (tail < SEGMENT_CAPACITY) {
      value = bclx::swap_sync(
          _get_gptr_to_data_of_segment(this->_tail_segment) + tail, data_ptr);
    }

    if (tail >= SEGMENT_CAPACITY || value == TOP_PTR) {
      bclx::gptr<segment_t> new_segment = BCL::alloc<segment_t>(1);
      new_segment.local()->head = 0;
      new_segment.local()->tail = 0;
      new_segment.local()->next = markable_gptr<segment_t>{nullptr};
      tail = 0;
      for (int i = 0; i < SEGMENT_CAPACITY; ++i) {
        new_segment.local()->data[i] = BOTTOM_PTR;
      }
      new_segment.local()->data[0] = data_ptr;

      if (value == TOP_PTR ||
          bclx::aget_sync(this->_d_head_segment) == nullptr) {
        bclx::aput_sync(new_segment, this->_d_head_segment);
      }
      if (this->_e_head_segment == nullptr) {
        this->_e_head_segment = new_segment;
      }
      bclx::aput_sync(
          create_markable_ptr(new_segment, value == TOP_PTR ||
                                               get_marker(bclx::aget_sync(
                                                   _get_gptr_to_next_of_segment(
                                                       this->_tail_segment)))),
          _get_gptr_to_next_of_segment(this->_tail_segment));
      this->_tail_segment = new_segment;
    }
    bclx::aput_sync(tail + 1,
                    _get_gptr_to_tail_of_segment(this->_tail_segment));

    if (enqueue_count > SEGMENT_CAPACITY * 4) {
      enqueue_count = 0;
      this->_try_reclaim_segments();
    }

    return true;
  }

  bool dequeue(data_t *output) {
  retry:
    bclx::gptr<segment_t> current_segment =
        this->_reserve(this->_d_head_segment);
    if (current_segment == nullptr) {
      return false;
    }
  next:
    uint64_t head =
        bclx::fao_sync(_get_gptr_to_head_of_segment(current_segment),
                       (uint64_t)1, BCL::plus<uint64_t>{});
    if (head >= SEGMENT_CAPACITY) {
      markable_gptr<segment_t> next_segment = this->_reserve_if_not_marked(
          _get_gptr_to_next_of_segment(current_segment));
      bclx::cas_sync(this->_d_head_segment, current_segment,
                     get_ptr(next_segment));
      if (get_ptr(next_segment) == nullptr) {
        this->_release_hazard_pointer(current_segment);
        return false;
      }
      if (get_marker(next_segment)) {
        this->_release_hazard_pointer(current_segment);
        goto retry;
      }
      this->_release_hazard_pointer(current_segment);
      current_segment = get_ptr(next_segment);
      goto next;
    }
    bclx::gptr<data_t> data_ptr = bclx::swap_sync(
        _get_gptr_to_data_of_segment(current_segment) + head, TOP_PTR);
    this->_release_hazard_pointer(current_segment);
    if (data_ptr == BOTTOM_PTR) {
      return false;
    }
    bclx::rread_sync(data_ptr, output, 1);
    return true;
  }

private:
  void _try_reclaim_segments() {
    bclx::gptr<segment_t> last_marked_segment = nullptr;
    while (true) {
      bclx::gptr<segment_t> old_head_segment =
          bclx::aget_sync(this->_d_head_segment);

      bclx::gptr<segment_t> current_segment;
      if (last_marked_segment == nullptr) {
        current_segment = this->_e_head_segment;
      } else {
        current_segment = get_ptr(
            bclx::aget_sync(_get_gptr_to_next_of_segment(last_marked_segment)));
      }

      while (current_segment != nullptr) {
        uint64_t cur_head =
            bclx::aget_sync(_get_gptr_to_head_of_segment(current_segment));
        if (cur_head >= SEGMENT_CAPACITY) {
          last_marked_segment = current_segment;
        }
        current_segment = get_ptr(
            bclx::aget_sync(_get_gptr_to_next_of_segment(current_segment)));
      }

      if (last_marked_segment == nullptr) {
        return;
      }

      markable_gptr<segment_t> last_marked_next =
          bclx::aget_sync(_get_gptr_to_next_of_segment(last_marked_segment));
      if (get_ptr(last_marked_next) == nullptr) {
        bclx::gptr<segment_t> new_segment = BCL::alloc<segment_t>(1);
        new_segment.local()->head = 0;
        new_segment.local()->tail = 0;
        new_segment.local()->next = markable_gptr<segment_t>{nullptr};
        for (int i = 0; i < SEGMENT_CAPACITY; ++i) {
          new_segment.local()->data[i] = BOTTOM_PTR;
        }
        bclx::aput_sync(create_nullptr<segment_t>(), this->_d_head_segment);
        this->_mark_and_free_upto(last_marked_segment);
        bclx::aput_sync(new_segment, this->_d_head_segment);
        this->_tail_segment = new_segment;
        this->_e_head_segment = new_segment;
        return;
      }

      if (bclx::cas_sync(this->_d_head_segment, old_head_segment,
                         get_ptr(last_marked_next)) == old_head_segment) {
        break;
      }
    }
    this->_mark_and_free_upto(last_marked_segment);
    this->_e_head_segment = last_marked_segment;
  }

  void _mark_and_free_upto(bclx::gptr<segment_t> ptr) {
    bclx::gptr<segment_t> cur_segment = this->_e_head_segment;
    while (true) {
      bclx::gptr<markable_gptr<segment_t>> next_segment_ptr =
          _get_gptr_to_next_of_segment(cur_segment);
      markable_gptr<segment_t> next_segment = bclx::aget_sync(next_segment_ptr);
      bclx::aput_sync(create_markable_ptr(get_ptr(next_segment), true),
                      next_segment_ptr);
      _free(cur_segment);
      if (cur_segment == ptr) {
        break;
      }
      cur_segment = get_ptr(next_segment);
    }
  }

private:
  void _free(bclx::gptr<segment_t> ptr) {
    this->_free_list.push_back(ptr);
    if (this->_free_list.size() >= 4 * this->_size) {
      this->_scan();
    }
  }

  void _scan() {
    std::vector<bclx::gptr<segment_t>> list_temp(2 * this->_size, nullptr);
    for (int i = 0; i < this->_size; ++i) {
      bclx::aread_async(this->_e_hazard_pointers[i], &list_temp[2 * i], 1);
      bclx::aread_async(this->_e_hazard_pointers[i] + 1, &list_temp[2 * i + 1],
                        1);
    }
    MPI_Win_flush_all(BCL::win);
    std::vector<bclx::gptr<segment_t>> dlist_temp;
    while (this->_free_list.size() > 0) {
      bclx::gptr<segment_t> ptr = this->_free_list.back();
      this->_free_list.pop_back();
      if (std::find(list_temp.begin(), list_temp.end(), ptr) !=
          list_temp.end()) {
        dlist_temp.push_back(ptr);
      } else {
        BCL::dealloc(ptr);
      }
    }
    this->_free_list = std::move(dlist_temp);
  }

  bclx::gptr<segment_t> _reserve(bclx::gptr<bclx::gptr<segment_t>> pptr) {
    int hazard_slot;
    if (bclx::aget_sync(this->_d_hazard_pointers) == nullptr) {
      hazard_slot = 0;
    } else if (bclx::aget_sync(this->_d_hazard_pointers + 1) == nullptr) {
      hazard_slot = 1;
    } else {
      printf("calling reserve without empty hazard slot!");
      exit(1);
    }
    bclx::gptr<segment_t> ptr = bclx::aget_sync(pptr);
    if (ptr == nullptr) {
      return nullptr;
    }
    bclx::aput_sync(ptr, this->_d_hazard_pointers + hazard_slot);
    bclx::gptr<segment_t> ptr_again;
    while ((ptr_again = bclx::aget_sync(pptr)) != ptr) {
      ptr = ptr_again;
      bclx::aput_sync(ptr, this->_d_hazard_pointers + hazard_slot);
      if (ptr == nullptr) {
        return nullptr;
      }
    }
    return ptr;
  }

  markable_gptr<segment_t>
  _reserve_if_not_marked(bclx::gptr<markable_gptr<segment_t>> ptr) {
    int hazard_slot;
    if (bclx::aget_sync(this->_d_hazard_pointers) == nullptr) {
      hazard_slot = 0;
    } else if (bclx::aget_sync(this->_d_hazard_pointers + 1) == nullptr) {
      hazard_slot = 1;
    } else {
      printf("calling reserve without empty hazard slot!");
      exit(1);
    }
    markable_gptr<segment_t> m_ptr = bclx::aget_sync(ptr);
    if (get_marker(m_ptr)) {
      return m_ptr;
    }
    if (get_ptr(m_ptr) == nullptr) {
      return m_ptr;
    }
    bclx::aput_sync(get_ptr(m_ptr), this->_d_hazard_pointers + hazard_slot);
    markable_gptr<segment_t> m_ptr_again;
    while ((m_ptr_again = bclx::aget_sync(ptr))._inner != m_ptr._inner) {
      m_ptr = m_ptr_again;
      if (get_marker(m_ptr)) {
        bclx::aput_sync(create_nullptr<segment_t>(),
                        this->_d_hazard_pointers + hazard_slot);
        return m_ptr;
      }
      bclx::aput_sync(get_ptr(m_ptr), this->_d_hazard_pointers + hazard_slot);
      if (get_ptr(m_ptr) == nullptr) {
        return m_ptr;
      }
    }
    return m_ptr;
  }

  void _release_hazard_pointer(bclx::gptr<segment_t> ptr) {
    bclx::gptr<segment_t> ptr0 = bclx::aget_sync(this->_d_hazard_pointers);
    bclx::gptr<segment_t> ptr1 = bclx::aget_sync(this->_d_hazard_pointers + 1);
    if (ptr0 == ptr) {
      bclx::aput_sync(create_nullptr<segment_t>(), this->_d_hazard_pointers);
    } else if (ptr1 == ptr) {
      bclx::aput_sync(create_nullptr<segment_t>(),
                      this->_d_hazard_pointers + 1);
    } else {
      printf("release a non-reserved pointer!");
      exit(1);
    }
  }

  inline bclx::gptr<uint64_t>
  _get_gptr_to_head_of_segment(bclx::gptr<segment_t> ptr) {
    bclx::gptr<uint64_t> head_ptr;
    head_ptr.rank = ptr.rank;
    head_ptr.ptr = ptr.ptr;
    return head_ptr;
  }

  inline bclx::gptr<uint64_t>
  _get_gptr_to_tail_of_segment(bclx::gptr<segment_t> ptr) {
    bclx::gptr<uint64_t> tail_ptr;
    tail_ptr.rank = ptr.rank;
    tail_ptr.ptr = ptr.ptr + offsetof(segment_t, tail);
    return tail_ptr;
  }

  inline bclx::gptr<markable_gptr<segment_t>>
  _get_gptr_to_next_of_segment(bclx::gptr<segment_t> ptr) {
    bclx::gptr<markable_gptr<segment_t>> next_ptr;
    next_ptr.rank = ptr.rank;
    next_ptr.ptr = ptr.ptr + offsetof(segment_t, next);
    return next_ptr;
  }

  inline bclx::gptr<bclx::gptr<data_t>>
  _get_gptr_to_data_of_segment(bclx::gptr<segment_t> ptr) {
    bclx::gptr<bclx::gptr<data_t>> data_ptr;
    data_ptr.rank = ptr.rank;
    data_ptr.ptr = ptr.ptr + offsetof(segment_t, data);
    return data_ptr;
  }
};
