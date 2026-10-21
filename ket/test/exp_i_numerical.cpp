#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#if KET_TEST_PRECISION == 0
using real_type = float;
#elif KET_TEST_PRECISION == 1
using real_type = double;
#else
using real_type = long double;
#endif

auto exp_i_subject(real_type phase) -> std::complex<real_type>;

int main()
{
  using limits = std::numeric_limits<real_type>;
  auto const pi = std::acos(real_type{-1});
  std::vector<real_type> const magnitudes{
    real_type{0}, real_type{0.7L}, pi / 2, pi,
    std::nextafter(pi, real_type{0}), std::nextafter(pi, limits::infinity()),
    real_type{1e10L}, real_type{1e20L}, limits::max() / 2, limits::max()};
  auto const tolerance = real_type{64} * limits::epsilon();
  unsigned failures = 0;
  unsigned count = 0;
  std::cout << std::setprecision(limits::max_digits10);
  for (auto const magnitude : magnitudes)
    for (auto const negative : {false, true})
    {
      auto const phase = negative ? -magnitude : magnitude;
      // The oracle and checks are built without fast-math, separately from
      // the subject, and with no LTO. The link still uses release flags.
      auto const expected = std::exp(std::complex<real_type>{0, phase});
      auto const trigonometric = std::complex<real_type>{std::cos(phase), std::sin(phase)};
      auto const actual = exp_i_subject(phase);
      ++count;
      if (not std::isfinite(expected.real()) or not std::isfinite(expected.imag())
          or std::abs(expected - trigonometric) > tolerance)
      {
        std::cerr << "Invalid oracle for phase " << phase << '\n';
        return 2;
      }
      auto const valid = std::isfinite(actual.real()) and std::isfinite(actual.imag());
      auto const signed_zero_matches = phase != 0
        or (actual.imag() == 0 and std::signbit(actual.imag()) == std::signbit(phase));
      if (not valid or std::abs(actual - expected) > tolerance
          or std::abs(std::abs(actual) - real_type{1}) > tolerance
          or not signed_zero_matches)
      {
        ++failures;
        std::cout << "FAIL phase=" << phase << " actual=" << actual
                  << " expected=" << expected
                  << " signed-zero=" << signed_zero_matches << '\n';
      }
    }
  std::cout << count << " cases, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
