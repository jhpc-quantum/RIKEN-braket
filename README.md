# bracket

## Introduction

**bracket** is a tool for simulations of quantum gates on (classical) computers.
It contains an interpreter of "quantum assembler" *bra* and a C++ template library *ket*.
It also contains a C++ MPI wrapper library *yampi* for users' convenience.

## Getting Started

**bracket** requires a MPI2-supported C++03 compliant compiler and [Boost library](http://www.boost.org/).
If you do not have Boost library, get download [the latest version](http://www.boost.org/users/download/) of the library, or get it by using a package manager of your Linux environment.
Because **bracket** does not require any compiled Boost libraries, you have just to extract the downloaded file.

You can retrieve the current status of **bracket** by cloning the repository:

```bash
git clone https://github.com/naoki-yoshioka/bracket.git
```

