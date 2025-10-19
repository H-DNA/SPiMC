#!/usr/bin/env bash

set -e

# Define colors and symbols
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

CHECK_MARK="✔"
CROSS_MARK="✖"

# ---

# Function to print a centered header
print_header() {
  local text="$1"
  local width=80
  local padding=$(((width - ${#text}) / 2))
  printf "%${padding}s" ""
  echo -e "${YELLOW}$text${NC}"
}

# ---

# Build the project
print_header "BUILDING PROJECT"
cd ../implementations
cmake .
cmake --build .
echo -e "${GREEN}Build successful!${NC}\n"

# ---

# Run the tests
print_header "RUNNING TESTS"
for i in {1..100}; do
  echo -e "${YELLOW}Iteration $i:${NC}"
  for n in {2..8}; do
    output=$(mpiexec -n "$n" ./out example 2> /dev/null)
    
    dequeued_numbers=$(echo "$output" | grep "dequeue" | awk '{print $4}' | grep -v -- "-1" | sort -n | tr -d '\n')
    
    expected_numbers=$(seq 0 49 | sort -n | tr -d '\n')
  
    if [ "$dequeued_numbers" != "$expected_numbers" ]; then
      echo -e "${RED}${CROSS_MARK}  Test failed for $n processes.${NC}"
      
      expected_file=$(mktemp)
      echo "$expected_numbers" > "$expected_file"
      
      actual_file=$(mktemp)
      echo "$dequeued_numbers" > "$actual_file"
      
      echo "Diff:"
      diff -u "$expected_file" "$actual_file" || true
      
      echo "Expected: $expected_numbers"
      echo "Actual:   $dequeued_numbers"
      
      rm "$expected_file" "$actual_file"
      
      exit 1
    fi
    echo -e "${GREEN}${CHECK_MARK}  Test passed for $n processes.${NC}"
  done
done

echo -e "\n${GREEN}All tests passed!${NC}"
