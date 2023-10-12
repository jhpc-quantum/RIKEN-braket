SHELL = /bin/sh
.SUFFIXES: 

KET_DIR = $(pwd)/../ket
YAMPI_DIR = $(pwd)/../yampi
CXXOPTS_DIR = $(pwd)/../cxxopts
FMT_DIR = $(pwd)/../fmt

target = bra
nodename := $(shell uname -n)

pwd = .
src_dir = $(pwd)/src
include_dir = $(pwd)/include
build_dir = $(pwd)/build
bin_dir = $(pwd)/bin

library_dirs = $(KET_DIR) $(YAMPI_DIR) $(CXXOPTS_DIR) $(FMT_DIR)
idirs = $(include_dir) $(addsuffix /include,$(library_dirs))
ldirs = $(addsuffix /lib,$(library_dirs))

sources := $(shell find $(src_dir) -name *.cpp)
objects = $(sources:$(src_dir)/%.cpp=$(build_dir)/%.o)

macros = KET_PRINT_LOG
macros += KET_USE_OPENMP
#macros += KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
macros += KET_USE_DIAGONAL_LOOP
macros += KET_USE_BARRIER
#macros += BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
#macros += KET_USE_COLLECTIVE_COMMUNICATIONS
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
release: common_flags += -Nclang
endif
release: common_flags += -Ofast
release: macros += NDEBUG
release: $(bin_dir)/$(target)

.PHONY: debug
ifneq ($(findstring fn01sv,$(nodename)),)
debug: common_flags += -Nclang
endif
debug: common_flags += -O0
debug: $(bin_dir)/$(target)

.PHONY: nompi
nompi: nompi-release

.PHONY: nompi-release
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release: CXX = FCCpx
nompi-release: common_flags += -Nclang
else
nompi-release: CXX = g++
endif
nompi-release: common_flags += -Ofast
nompi-release: macros += NDEBUG BRA_NO_MPI
nompi-release: $(bin_dir)/$(target)

.PHONY: nompi-debug
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug: CXX = FCCpx
nompi-debug: common_flags += -Nclang
else
nompi-debug: CXX = g++
endif
nompi-debug: common_flags += -O0
nompi-debug: macros += BRA_NO_MPI
nompi-debug: $(bin_dir)/$(target)

.PHONY: release-float
ifneq ($(findstring fn01sv,$(nodename)),)
release-float: common_flags += -Nclang
endif
release-float: common_flags += -Ofast
release-float: macros += NDEBUG BRA_REAL_TYPE=2
release-float: $(bin_dir)/$(target)

.PHONY: debug-float
ifneq ($(findstring fn01sv,$(nodename)),)
debug-float: common_flags += -Nclang
endif
debug-float: common_flags += -O0
debug-float: macros += BRA_REAL_TYPE=2
debug-float: $(bin_dir)/$(target)

.PHONY: nompi-float
nompi-float: nompi-release-float

.PHONY: nompi-release-float
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release-float: CXX = FCCpx
nompi-release-float: common_flags += -Nclang
else
nompi-release-float: CXX = g++
endif
nompi-release-float: common_flags += -Ofast
nompi-release-float: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=2
nompi-release-float: $(bin_dir)/$(target)

.PHONY: nompi-debug-float
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug-float: CXX = FCCpx
nompi-debug-float: common_flags += -Nclang
else
nompi-debug-float: CXX = g++
endif
nompi-debug-float: common_flags += -O0
nompi-debug-float: macros += BRA_NO_MPI BRA_REAL_TYPE=2
nompi-debug-float: $(bin_dir)/$(target)

.PHONY: release-long
ifneq ($(findstring fn01sv,$(nodename)),)
release-long: common_flags += -Nclang
endif
release-long: common_flags += -Ofast
release-long: macros += NDEBUG BRA_REAL_TYPE=0
release-long: $(bin_dir)/$(target)

.PHONY: debug-long
ifneq ($(findstring fn01sv,$(nodename)),)
debug-long: common_flags += -Nclang
endif
debug-long: common_flags += -O0
debug-long: macros += BRA_REAL_TYPE=0
debug-long: $(bin_dir)/$(target)

.PHONY: nompi-long
nompi-long: nompi-release-long

.PHONY: nompi-release-long
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-release-long: CXX = FCCpx
nompi-release-long: common_flags += -Nclang
else
nompi-release-long: CXX = g++
endif
nompi-release-long: common_flags += -Ofast
nompi-release-long: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=0
nompi-release-long: $(bin_dir)/$(target)

.PHONY: nompi-debug-long
ifneq ($(findstring fn01sv,$(nodename)),)
nompi-debug-long: CXX = FCCpx
nompi-debug-long: common_flags += -Nclang
else
nompi-debug-long: CXX = g++
endif
nompi-debug-long: common_flags += -O0
nompi-debug-long: macros += BRA_NO_MPI BRA_REAL_TYPE=0
nompi-debug-long: $(bin_dir)/$(target)

ifneq ($(findstring fn01sv,$(nodename)),)
  CXX = mpiFCCpx
else
  CXX = mpiCC
endif
common_flags = -fopenmp -std=c++17 -Wall
cxx_flags =

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

