#ifndef YAMPI_WALL_CLOCK_HPP
# define YAMPI_WALL_CLOCK_HPP

# include <boost/config.hpp>

# if !defined(BOOST_NO_CXX11_HDR_CHRONO) and !defined(BOOST_NO_CXX11_HDR_RATIO)
#   include <chrono>
#   include <ratio>
# else
#   include <boost/chrono/duration.hpp>
#   include <boost/chrono/time_point.hpp>
#   include <boost/ratio.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>

# if !defined(BOOST_NO_CXX11_HDR_CHRONO) and !defined(BOOST_NO_CXX11_HDR_RATIO)
#   define YAMPI_chrono std::chrono
#   define YAMPI_ratio std::ratio
# else
#   define YAMPI_chrono boost::chrono
#   define YAMPI_ratio boost::ratio
# endif


namespace yampi
{
  struct wall_clock
  {
    typedef double rep;
    typedef YAMPI_ratio<1> period;
    typedef YAMPI_chrono::duration<rep, period> duration;
    typedef YAMPI_chrono::time_point< ::yampi::wall_clock > time_point;
    BOOST_STATIC_CONSTEXPR bool is_steady = false;

    static time_point now(::yampi::environment const&)
    { return static_cast<time_point>(static_cast<duration>(MPI_Wtime())); }
    static duration tick(::yampi::environment const&)
    { return static_cast<duration>(MPI_Wtick()); }
  };
}


# undef YAMPI_chrono
# undef YAMPI_ratio

#endif

