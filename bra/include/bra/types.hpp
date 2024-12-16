#ifndef BRA_TYPES_HPP
# define BRA_TYPES_HPP

# include <cstdint>
# include <complex>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifndef BRA_NO_MPI
#   include <ket/mpi/permutated.hpp>
# endif // BRA_NO_MPI


namespace bra
{
  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  using control_qubit_type = ket::control<qubit_type>;
# ifndef BRA_NO_MPI
  using permutated_qubit_type = ket::mpi::permutated< ::bra::qubit_type >;
  using permutated_control_qubit_type = ket::mpi::permutated< ::bra::control_qubit_type >;
# endif // BRA_NO_MPI

# ifdef BRA_REAL_TYPE
#   if BRA_REAL_TYPE == 0
  using real_type = long double;
#   elif BRA_REAL_TYPE == 1
  using real_type = double;
#   elif BRA_REAL_TYPE == 2
  using real_type = float;
#   else
  using real_type = double;
#   endif
# else // BRA_REAL_TYPE
  using real_type = double;
# endif // BRA_REAL_TYPE
  using complex_type = std::complex<real_type>;
} // namespace bra


#endif // BRA_TYPES_HPP
