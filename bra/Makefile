SHELL = /bin/sh
.SUFFIXES: 

KET_DIR = $(pwd)/../ket
YAMPI_DIR = $(pwd)/../yampi

boost_dir = $(pwd)/../boost
boost_submodules = $(shell ls $(boost_dir))
boost_submodule_dirs = $(addprefix $(boost_dir)/,$(boost_submodules))
boost_include_dirs = $(addsuffix /include,$(boost_submodule_dirs))

target = bra
nodename := $(shell uname -n)

pwd = .
src_dir = $(pwd)/src
include_dir = $(pwd)/include
build_dir = $(pwd)/build
bin_dir = $(pwd)/bin

library_dirs = $(KET_DIR) $(YAMPI_DIR)
idirs = $(include_dir) $(k_boost_include_dir) $(addsuffix /include,$(library_dirs)) $(boost_include_dirs)
ldirs = $(addsuffix /lib,$(library_dirs))

sources := $(shell find $(src_dir) -name *.cpp)
objects = $(sources:$(src_dir)/%.cpp=$(build_dir)/%.o)

macros = KET_USE_OPENMP KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS KET_PRINT_LOG
#macros += KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#macros += KET_USE_DIAGONAL_LOOP
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
ifneq ($(findstring juqueen,$(nodename)),)
release: release03
else
  ifneq ($(findstring fe01p,$(nodename)),)
release: release11
  else
release: release14
  endif
endif

.PHONY: debug
ifneq ($(findstring juqueen,$(nodename)),)
debug: debug03
else
  ifneq ($(findstring fe01p,$(nodename)),)
debug: debug11
  else
debug: debug14
  endif
endif

.PHONY: release03
ifneq ($(findstring juqueen,$(nodename)),)
release03: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename)),)
release03: common_flags += -Kfast -std=c++03
  else
release03: common_flags += -O3 -std=c++03
  endif
endif
release03: macros += NDEBUG
release03: $(bin_dir)/$(target)

.PHONY: debug03
ifneq ($(findstring juqueen,$(nodename)),)
debug03: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename)),)
debug03: common_flags += -O2 -std=c++03
  else
debug03: common_flags += -O0 -std=c++03
  endif
endif
debug03: $(bin_dir)/$(target)

.PHONY: release11
ifneq ($(findstring juqueen,$(nodename)),)
release11: release03
else
  ifneq ($(findstring fe01p,$(nodename)),)
release11: common_flags += -Kfast -Xg -std=c++11
  else
release11: common_flags += -O3 -std=c++11
  endif
endif
release11: macros += NDEBUG
release11: $(bin_dir)/$(target)

.PHONY: debug11
ifneq ($(findstring juqueen,$(nodename)),)
debug11: debug03
else
  ifneq ($(findstring fe01p,$(nodename)),)
debug11: common_flags += -O2 -Xg -std=c++11
  else
debug11: common_flags += -O0 -std=c++11
  endif
endif
debug11: $(bin_dir)/$(target)

.PHONY: release14
ifneq ($(findstring juqueen,$(nodename)),)
release14: release11
else
  ifneq ($(findstring fe01p,$(nodename)),)
release14: release11
  else
release14: common_flags += -O3 -std=c++14
  endif
endif
release14: macros += NDEBUG
release14: $(bin_dir)/$(target)

.PHONY: debug14
ifneq ($(findstring juqueen,$(nodename)),)
debug14: debug11
else
  ifneq ($(findstring fe01p,$(nodename)),)
debug14: debug11
  else
debug14: common_flags += -O0 -std=c++14
  endif
endif
debug14: $(bin_dir)/$(target)


.PHONY: nompi
nompi: nompi-release

.PHONY: nompi-release
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release: nompi-release03
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-release: nompi-release11
  else
nompi-release: nompi-release14
  endif
endif

.PHONY: nompi-debug
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug: nompi-debug03
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-debug: nompi-debug11
  else
nompi-debug: nompi-debug14
  endif
endif

.PHONY: nompi-release03
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release03: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-release03: common_flags += -Kfast -std=c++03
  else
nompi-release03: common_flags += -O3 -std=c++03
  endif
endif
nompi-release03: macros += NDEBUG BRA_NO_MPI
nompi-release03: $(bin_dir)/$(target)

.PHONY: nompi-debug03
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug03: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-debug03: common_flags += -O2 -std=c++03
  else
nompi-debug03: common_flags += -O0 -std=c++03
  endif
endif
nompi-debug03: macros += BRA_NO_MPI
nompi-debug03: $(bin_dir)/$(target)

.PHONY: release11
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release11: release03
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-release11: common_flags += -Kfast -Xg -std=c++11
  else
nompi-release11: common_flags += -O3 -std=c++11
  endif
endif
nompi-release11: macros += NDEBUG BRA_NO_MPI
nompi-release11: $(bin_dir)/$(target)

.PHONY: nompi-debug11
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug11: debug03
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-debug11: common_flags += -O2 -Xg -std=c++11
  else
nompi-debug11: common_flags += -O0 -std=c++11
  endif
endif
nompi-debug11: macros += BRA_NO_MPI
nompi-debug11: $(bin_dir)/$(target)

.PHONY: nompi-release14
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release14: release11
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-release14: release11
  else
nompi-release14: common_flags += -O3 -std=c++14
  endif
endif
nompi-release14: macros += NDEBUG BRA_NO_MPI
nompi-release14: $(bin_dir)/$(target)

.PHONY: nompi-debug14
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug14: debug11
else
  ifneq ($(findstring fe01p,$(nodename)),)
nompi-debug14: debug11
  else
nompi-debug14: common_flags += -O0 -std=c++14
  endif
endif
nompi-debug14: macros += BRA_NO_MPI
nompi-debug14: $(bin_dir)/$(target)


ifneq ($(findstring juqueen,$(nodename)),)
  CXX = mpixlcxx
  common_flags = -qsmp=omp
  cxx_flags =
else
  ifneq ($(findstring fe01p,$(nodename)),)
    CXX = mpiFCCpx
    common_flags = -Kparallel,openmp
    cxx_flags = --alternative_tokens
    k_boost_include_dir = $(pwd)/../k_boost/include
  else
    CXX = mpiCC
    common_flags = -fopenmp
    cxx_flags =
  endif
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

