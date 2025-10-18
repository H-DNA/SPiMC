#include "./segment_queue/segment_queue.hpp"
#include "bcl/backends/mpi/backend.hpp"
#include "bcl/core/teams.hpp"
#include <bclx/bclx.hpp>
#include <cstring>
#include <iostream>
#include <ostream>

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [option]\n"
            << "Options:\n"
            << "  example      - Run example\n"
            << "  help         - Show this help message\n";
}

void run_example() {
  int rank = BCL::my_rank;
  int size = BCL::nprocs();
  const int poison_pill = -1;

  if (rank == 0) {
    SegmentQueue<int> queue(0, MPI_COMM_WORLD);
    for (int i = 0; i < 50; ++i) {
      if (queue.enqueue(i)) {
        std::cout << "-- Enqueue " << i << std::endl;
      }
    }
    // Enqueue a poison pill for each consumer
    for (int i = 0; i < size - 1; ++i) {
      if (queue.enqueue(poison_pill)) {
        std::cout << "-- Enqueue " << poison_pill << std::endl;
      }
    }
  } else {
    SegmentQueue<int> queue(0, MPI_COMM_WORLD);
    while (true) {
      int value;
      if (queue.dequeue(&value)) {
        std::cout << "Rank " << rank << ": dequeue " << value << std::endl;
        // Stop if a poison pill is dequeued
        if (value == poison_pill) {
          break;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  BCL::init();

  if (argc == 1 || (argc == 2 && strcmp(argv[1], "example") == 0)) {
    run_example();
  } else if (argc == 2 && strcmp(argv[1], "help") == 0) {
    print_usage(argv[0]);
  } else {
    print_usage(argv[0]);
    BCL::finalize();
    return 1;
  }

  BCL::finalize();
  return 0;
}
