#include "./segment_queue/segment_queue.hpp"
#include "bcl/backends/mpi/backend.hpp"
#include "bcl/core/teams.hpp"
#include <bclx/bclx.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <ostream>

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [option]\n"
            << "Options:\n"
            << "  example      - Run example\n"
            << "  micro - Run microbenchmark\n"
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

void run_microbenchmark() {
  int rank = BCL::my_rank;
  int size = BCL::nprocs();
  const int num_items = 100000;
  const int poison_pill = -1;

  SegmentQueue<int> queue(0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < 100; ++i) {
      queue.enqueue(i);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 100; i < num_items; ++i) {
      queue.enqueue(i);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    double throughput = (num_items - 100) / (duration.count() / 1e6) / 1e5;
    double latency = duration.count() / (double)(num_items - 100);

    std::cout << "Enqueuer (Rank 0):\n";
    std::cout << "  Enqueue latency: " << latency << " us/op\n";
    std::cout << "  Enqueue throughput: " << throughput << " 10^5 ops/s\n";

    for (int i = 0; i < size - 1; ++i) {
      queue.enqueue(poison_pill);
    }

    if (size > 1) {
      double total_latency = 0;
      double total_throughput = 0;
      for (int i = 1; i < size; ++i) {
        double remote_latency, remote_throughput;
        MPI_Recv(&remote_latency, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&remote_throughput, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        total_latency += remote_latency;
        total_throughput += remote_throughput;
      }

      double avg_latency = total_latency / (size - 1);
      double avg_throughput = total_throughput;

      std::cout << "Dequeuers (Total):\n";
      std::cout << "  Average dequeue latency: " << avg_latency
                << " us/op\n";
      std::cout << "  Total dequeue throughput: " << avg_throughput
                << " 10^5 ops/s\n";
    }
  } else {
    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::high_resolution_clock::now();

    int value;
    int count = 0;
    while (true) {
      if (queue.dequeue(&value)) {
        if (value == poison_pill) {
          break;
        }
        count++;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    if (count > 0) {
      double latency = duration.count() / (double)count;
      double throughput = count / (duration.count() / 1e6) / 1e5;
      MPI_Send(&latency, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&throughput, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
      double latency = 0;
      double throughput = 0;
      MPI_Send(&latency, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&throughput, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
  }
}

int main(int argc, char **argv) {
  BCL::init();

  if (argc == 2 && strcmp(argv[1], "example") == 0) {
    run_example();
  } else if (argc == 2 && strcmp(argv[1], "micro") == 0) {
    run_microbenchmark();
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
