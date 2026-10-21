#include <complex>

#include <ket/utility/exp_i.hpp>

#if KET_TEST_PRECISION == 0
using real_type = float;
#elif KET_TEST_PRECISION == 1
using real_type = double;
#else
using real_type = long double;
#endif

// Compile this translation unit with the optimization settings under test.
auto exp_i_subject(real_type const phase) -> std::complex<real_type>
{ return ket::utility::exp_i<std::complex<real_type>>(phase); }
