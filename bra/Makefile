SHELL = /bin/sh
.SUFFIXES: 

KET_DIR = $(pwd)/../ket
YAMPI_DIR = $(pwd)/../yampi

target = bra
nodename := $(shell uname -n)

pwd = .
src_dir = $(pwd)/src
include_dir = $(pwd)/include
build_dir = $(pwd)/build
bin_dir = $(pwd)/bin

library_dirs = $(KET_DIR) $(YAMPI_DIR)
idirs = $(include_dir) $(addsuffix /include,$(library_dirs))
ldirs = $(addsuffix /lib,$(library_dirs))

sources := $(shell find $(src_dir) -name *.cpp)
objects = $(sources:$(src_dir)/%.cpp=$(build_dir)/%.o)

macros = KET_PRINT_LOG
macros += KET_USE_OPENMP
#macros += KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
macros += KET_PREFER_POINTER_TO_VECTOR_ITERATOR
macros += KET_USE_DIAGONAL_LOOP
libraries =

CPPFLAGS = $(addprefix -I,$(idirs)) $(addprefix -D,$(macros))
CXXFLAGS = $(common_flags) $(cxx_flags)
LDFLAGS = $(common_flags) $(addprefix -L,$(ldirs))
ifeq ($(strip $(libraries)),)
  library_flags =
else
  library_flags = $(addprefix -l,$(libraries))
endif


.PHONY: release
ifneq ($(findstring fn01sv,$(nodename)),)
release: common_flags += -Nclang,libomp -Kfast -std=c++11
else
release: common_flags += -O3 -std=c++17
endif
release: macros += NDEBUG
release: $(bin_dir)/$(target)

.PHONY: debug
ifneq ($(findstring fn01sv,$(nodename)),)
debug: common_flags += -Nclang,libomp -O2 -std=c++11
else
debug: common_flags += -O0 -std=c++17
endif
debug: $(bin_dir)/$(target)

.PHONY: nompi
nompi: nompi-release

.PHONY: nompi-release
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release: common_flags += -Nclang,libomp -Kfast -std=c++11
else
nompi-release: common_flags += -O3 -std=c++17
endif
nompi-release: macros += NDEBUG BRA_NO_MPI
nompi-release: $(bin_dir)/$(target)

.PHONY: nompi-debug
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug: common_flags += -Nclang,libomp -O2 -std=c++11
else
nompi-debug: common_flags += -O0 -std=c++17
endif
nompi-debug: macros += BRA_NO_MPI
nompi-debug: $(bin_dir)/$(target)

.PHONY: release-float
ifneq ($(findstring fn01sv,$(nodename)),)
release-float: common_flags += -Nclang,libomp -Kfast -std=c++11
else
release-float: common_flags += -O3 -std=c++17
endif
release-float: macros += NDEBUG BRA_REAL_TYPE=2
release-float: $(bin_dir)/$(target)

.PHONY: debug-float
ifneq ($(findstring fn01sv,$(nodename)),)
debug-float: common_flags += -Nclang,libomp -O2 -std=c++11
else
debug-float: common_flags += -O0 -std=c++17
endif
debug-float: macros += BRA_REAL_TYPE=2
debug-float: $(bin_dir)/$(target)

.PHONY: nompi-float
nompi-float: nompi-release-float

.PHONY: nompi-release-float
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release-float: common_flags += -Nclang,libomp -Kfast -std=c++11
else
nompi-release-float: common_flags += -O3 -std=c++17
endif
nompi-release-float: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=2
nompi-release-float: $(bin_dir)/$(target)

.PHONY: nompi-debug-float
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug-float: common_flags += -Nclang,libomp -O2 -std=c++11
else
nompi-debug-float: common_flags += -O0 -std=c++17
endif
nompi-debug-float: macros += BRA_NO_MPI BRA_REAL_TYPE=2
nompi-debug-float: $(bin_dir)/$(target)

.PHONY: release-long
ifneq ($(findstring fn01sv,$(nodename)),)
release-long: common_flags += -Nclang,libomp -Kfast -std=c++11
else
release-long: common_flags += -O3 -std=c++17
endif
release-long: macros += NDEBUG BRA_REAL_TYPE=0
release-long: $(bin_dir)/$(target)

.PHONY: debug-long
ifneq ($(findstring fn01sv,$(nodename)),)
debug-long: common_flags += -Nclang,libomp -O2 -std=c++11
else
debug-long: common_flags += -O0 -std=c++17
endif
debug-long: macros += BRA_REAL_TYPE=0
debug-long: $(bin_dir)/$(target)

.PHONY: nompi-long
nompi-long: nompi-release-long

.PHONY: nompi-release-long
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release-long: common_flags += -Nclang,libomp -Kfast -std=c++11
else
nompi-release-long: common_flags += -O3 -std=c++17
endif
nompi-release-long: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=0
nompi-release-long: $(bin_dir)/$(target)

.PHONY: nompi-debug-long
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug-long: common_flags += -Nclang,libomp -O2 -std=c++11
else
nompi-debug-long: common_flags += -O0 -std=c++17
endif
nompi-debug-long: macros += BRA_NO_MPI BRA_REAL_TYPE=0
nompi-debug-long: $(bin_dir)/$(target)

ifneq ($(findstring fn01sv,$(nodename)),)
  CXX = mpiFCCpx
  common_flags = -Kopenmp
  cxx_flags =
else
  CXX = mpiCC
  common_flags = -fopenmp
  cxx_flags =
endif

$(bin_dir)/$(target): $(objects)
	$(mkdir_p) $(dir $@)
	$(CXX) $(objects) $(library_flags) -o $@ $(LDFLAGS)

$(build_dir)/%.o: $(src_dir)/%.cpp
	$(mkdir_p) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(src_dir)/%.cpp: $(include_dir)/%.hpp ;

.PHONY: clean
clean:
	$(RM) -r $(build_dir) $(bin_dir) $(distfile)

mkdir_p ?= mkdir -p

