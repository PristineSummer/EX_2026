CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp
# Keep the bundled ARM executable untouched. All new artifacts stay in EX_2026.
OUTPUT = ../../build/lab1/make/quicksort

.PHONY: all clean
all: $(OUTPUT)

$(OUTPUT): quick_sort.cpp sort_core.cpp sort_core.hpp
	mkdir -p $(dir $(OUTPUT))
	$(CXX) $(CXXFLAGS) quick_sort.cpp sort_core.cpp -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)

