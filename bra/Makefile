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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release: release11
  else
release: release14
  endif
endif

.PHONY: debug
ifneq ($(findstring juqueen,$(nodename)),)
debug: debug03
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug: debug11
  else
debug: debug14
  endif
endif

.PHONY: release03
ifneq ($(findstring juqueen,$(nodename)),)
release03: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release: nompi-release11
  else
nompi-release: nompi-release14
  endif
endif

.PHONY: nompi-debug
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug: nompi-debug03
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug: nompi-debug11
  else
nompi-debug: nompi-debug14
  endif
endif

.PHONY: nompi-release03
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release03: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug03: common_flags += -O2 -std=c++03
  else
nompi-debug03: common_flags += -O0 -std=c++03
  endif
endif
nompi-debug03: macros += BRA_NO_MPI
nompi-debug03: $(bin_dir)/$(target)

.PHONY: nompi-release11
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release11: nompi-release03
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release11: common_flags += -Kfast -Xg -std=c++11
  else
nompi-release11: common_flags += -O3 -std=c++11
  endif
endif
nompi-release11: macros += NDEBUG BRA_NO_MPI
nompi-release11: $(bin_dir)/$(target)

.PHONY: nompi-debug11
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug11: nompi-debug03
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug11: common_flags += -O2 -Xg -std=c++11
  else
nompi-debug11: common_flags += -O0 -std=c++11
  endif
endif
nompi-debug11: macros += BRA_NO_MPI
nompi-debug11: $(bin_dir)/$(target)

.PHONY: nompi-release14
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release14: nompi-release11
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release14: nompi-release11
  else
nompi-release14: common_flags += -O3 -std=c++14
  endif
endif
nompi-release14: macros += NDEBUG BRA_NO_MPI
nompi-release14: $(bin_dir)/$(target)

.PHONY: nompi-debug14
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug14: nompi-debug11
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug14: nompi-debug11
  else
nompi-debug14: common_flags += -O0 -std=c++14
  endif
endif
nompi-debug14: macros += BRA_NO_MPI
nompi-debug14: $(bin_dir)/$(target)


.PHONY: release-float
ifneq ($(findstring juqueen,$(nodename)),)
release-float: release03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release-float: release11-float
  else
release-float: release14-float
  endif
endif

.PHONY: debug-float
ifneq ($(findstring juqueen,$(nodename)),)
debug-float: debug03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug-float: debug11-float
  else
debug-float: debug14-float
  endif
endif

.PHONY: release03-float
ifneq ($(findstring juqueen,$(nodename)),)
release03-float: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release03-float: common_flags += -Kfast -std=c++03
  else
release03-float: common_flags += -O3 -std=c++03
  endif
endif
release03-float: macros += NDEBUG BRA_REAL_TYPE=2
release03-float: $(bin_dir)/$(target)

.PHONY: debug03-float
ifneq ($(findstring juqueen,$(nodename)),)
debug03-float: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug03-float: common_flags += -O2 -std=c++03
  else
debug03-float: common_flags += -O0 -std=c++03
  endif
endif
debug03-float: macros += BRA_REAL_TYPE=2
debug03-float: $(bin_dir)/$(target)

.PHONY: release11-float
ifneq ($(findstring juqueen,$(nodename)),)
release11-float: release03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release11-float: common_flags += -Kfast -Xg -std=c++11
  else
release11-float: common_flags += -O3 -std=c++11
  endif
endif
release11-float: macros += NDEBUG BRA_REAL_TYPE=2
release11-float: $(bin_dir)/$(target)

.PHONY: debug11-float
ifneq ($(findstring juqueen,$(nodename)),)
debug11-float: debug03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug11-float: common_flags += -O2 -Xg -std=c++11
  else
debug11-float: common_flags += -O0 -std=c++11
  endif
endif
debug11-float: macros += BRA_REAL_TYPE=2
debug11-float: $(bin_dir)/$(target)

.PHONY: release14-float
ifneq ($(findstring juqueen,$(nodename)),)
release14-float: release11-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release14-float: release11-float
  else
release14-float: common_flags += -O3 -std=c++14
  endif
endif
release14-float: macros += NDEBUG BRA_REAL_TYPE=2
release14-float: $(bin_dir)/$(target)

.PHONY: debug14-float
ifneq ($(findstring juqueen,$(nodename)),)
debug14-float: debug11-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug14-float: debug11-float
  else
debug14-float: common_flags += -O0 -std=c++14
  endif
endif
debug14-float: macros += BRA_REAL_TYPE=2
debug14-float: $(bin_dir)/$(target)


.PHONY: nompi-float
nompi-float: nompi-release-float

.PHONY: nompi-release-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release-float: nompi-release03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release-float: nompi-release11-float
  else
nompi-release-float: nompi-release14-float
  endif
endif

.PHONY: nompi-debug-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug-float: nompi-debug03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug-float: nompi-debug11-float
  else
nompi-debug-float: nompi-debug14-float
  endif
endif

.PHONY: nompi-release03-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release03-float: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release03-float: common_flags += -Kfast -std=c++03
  else
nompi-release03-float: common_flags += -O3 -std=c++03
  endif
endif
nompi-release03-float: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=2
nompi-release03-float: $(bin_dir)/$(target)

.PHONY: nompi-debug03-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug03-float: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug03-float: common_flags += -O2 -std=c++03
  else
nompi-debug03-float: common_flags += -O0 -std=c++03
  endif
endif
nompi-debug03-float: macros += BRA_NO_MPI BRA_REAL_TYPE=2
nompi-debug03-float: $(bin_dir)/$(target)

.PHONY: nompi-release11-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release11-float: nompi-release03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release11-float: common_flags += -Kfast -Xg -std=c++11
  else
nompi-release11-float: common_flags += -O3 -std=c++11
  endif
endif
nompi-release11-float: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=2
nompi-release11-float: $(bin_dir)/$(target)

.PHONY: nompi-debug11-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug11-float: nompi-debug03-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug11-float: common_flags += -O2 -Xg -std=c++11
  else
nompi-debug11-float: common_flags += -O0 -std=c++11
  endif
endif
nompi-debug11-float: macros += BRA_NO_MPI BRA_REAL_TYPE=2
nompi-debug11-float: $(bin_dir)/$(target)

.PHONY: nompi-release14-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release14-float: nompi-release11-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release14-float: nompi-release11-float
  else
nompi-release14-float: common_flags += -O3 -std=c++14
  endif
endif
nompi-release14-float: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=2
nompi-release14-float: $(bin_dir)/$(target)

.PHONY: nompi-debug14-float
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug14-float: nompi-debug11-float
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug14-float: nompi-debug11-float
  else
nompi-debug14-float: common_flags += -O0 -std=c++14
  endif
endif
nompi-debug14-float: macros += BRA_NO_MPI BRA_REAL_TYPE=2
nompi-debug14-float: $(bin_dir)/$(target)


.PHONY: release-long
ifneq ($(findstring juqueen,$(nodename)),)
release-long: release03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release-long: release11-long
  else
release-long: release14-long
  endif
endif

.PHONY: debug-long
ifneq ($(findstring juqueen,$(nodename)),)
debug-long: debug03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug-long: debug11-long
  else
debug-long: debug14-long
  endif
endif

.PHONY: release03-long
ifneq ($(findstring juqueen,$(nodename)),)
release03-long: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release03-long: common_flags += -Kfast -std=c++03
  else
release03-long: common_flags += -O3 -std=c++03
  endif
endif
release03-long: macros += NDEBUG BRA_REAL_TYPE=0
release03-long: $(bin_dir)/$(target)

.PHONY: debug03-long
ifneq ($(findstring juqueen,$(nodename)),)
debug03-long: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug03-long: common_flags += -O2 -std=c++03
  else
debug03-long: common_flags += -O0 -std=c++03
  endif
endif
debug03-long: macros += BRA_REAL_TYPE=0
debug03-long: $(bin_dir)/$(target)

.PHONY: release11-long
ifneq ($(findstring juqueen,$(nodename)),)
release11-long: release03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release11-long: common_flags += -Kfast -Xg -std=c++11
  else
release11-long: common_flags += -O3 -std=c++11
  endif
endif
release11-long: macros += NDEBUG BRA_REAL_TYPE=0
release11-long: $(bin_dir)/$(target)

.PHONY: debug11-long
ifneq ($(findstring juqueen,$(nodename)),)
debug11-long: debug03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug11-long: common_flags += -O2 -Xg -std=c++11
  else
debug11-long: common_flags += -O0 -std=c++11
  endif
endif
debug11-long: macros += BRA_REAL_TYPE=0
debug11-long: $(bin_dir)/$(target)

.PHONY: release14-long
ifneq ($(findstring juqueen,$(nodename)),)
release14-long: release11-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
release14-long: release11-long
  else
release14-long: common_flags += -O3 -std=c++14
  endif
endif
release14-long: macros += NDEBUG BRA_REAL_TYPE=0
release14-long: $(bin_dir)/$(target)

.PHONY: debug14-long
ifneq ($(findstring juqueen,$(nodename)),)
debug14-long: debug11-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
debug14-long: debug11-long
  else
debug14-long: common_flags += -O0 -std=c++14
  endif
endif
debug14-long: macros += BRA_REAL_TYPE=0
debug14-long: $(bin_dir)/$(target)


.PHONY: nompi-long
nompi-long: nompi-release-long

.PHONY: nompi-release-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release-long: nompi-release03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release-long: nompi-release11-long
  else
nompi-release-long: nompi-release14-long
  endif
endif

.PHONY: nompi-debug-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug-long: nompi-debug03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug-long: nompi-debug11-long
  else
nompi-debug-long: nompi-debug14-long
  endif
endif

.PHONY: nompi-release03-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release03-long: common_flags += -O4 -qarch=qp -qtune=qp
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release03-long: common_flags += -Kfast -std=c++03
  else
nompi-release03-long: common_flags += -O3 -std=c++03
  endif
endif
nompi-release03-long: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=0
nompi-release03-long: $(bin_dir)/$(target)

.PHONY: nompi-debug03-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug03-long: common_flags += -O0
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug03-long: common_flags += -O2 -std=c++03
  else
nompi-debug03-long: common_flags += -O0 -std=c++03
  endif
endif
nompi-debug03-long: macros += BRA_NO_MPI BRA_REAL_TYPE=0
nompi-debug03-long: $(bin_dir)/$(target)

.PHONY: nompi-release11-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release11-long: nompi-release03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release11-long: common_flags += -Kfast -Xg -std=c++11
  else
nompi-release11-long: common_flags += -O3 -std=c++11
  endif
endif
nompi-release11-long: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=0
nompi-release11-long: $(bin_dir)/$(target)

.PHONY: nompi-debug11-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug11-long: nompi-debug03-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug11-long: common_flags += -O2 -Xg -std=c++11
  else
nompi-debug11-long: common_flags += -O0 -std=c++11
  endif
endif
nompi-debug11-long: macros += BRA_NO_MPI BRA_REAL_TYPE=0
nompi-debug11-long: $(bin_dir)/$(target)

.PHONY: nompi-release14-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-release14-long: nompi-release11-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-release14-long: nompi-release11-long
  else
nompi-release14-long: common_flags += -O3 -std=c++14
  endif
endif
nompi-release14-long: macros += NDEBUG BRA_NO_MPI BRA_REAL_TYPE=0
nompi-release14-long: $(bin_dir)/$(target)

.PHONY: nompi-debug14-long
ifneq ($(findstring juqueen,$(nodename)),)
nompi-debug14-long: nompi-debug11-long
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
nompi-debug14-long: nompi-debug11-long
  else
nompi-debug14-long: common_flags += -O0 -std=c++14
  endif
endif
nompi-debug14-long: macros += BRA_NO_MPI BRA_REAL_TYPE=0
nompi-debug14-long: $(bin_dir)/$(target)


ifneq ($(findstring juqueen,$(nodename)),)
  CXX = mpixlcxx
  common_flags = -qsmp=omp
  cxx_flags =
else
  ifneq ($(findstring fe01p,$(nodename))$(findstring hpcifx,$(nodename))$(findstring fn01sv,$(nodename)),)
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

