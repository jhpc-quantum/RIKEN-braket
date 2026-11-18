// Experimental type-level probe for #636; this is not a bra build or proof.
// It uses bra's lexical parsing mechanism and ket's exp_i primitive.
#include <boost/lexical_cast.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <ket/utility/exp_i.hpp>

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using high_type = boost::multiprecision::cpp_dec_float_100;

template <typename Real>
void probe(char const* name)
{
  std::cout << "type=" << name << " radix=" << std::numeric_limits<Real>::radix
            << " digits=" << std::numeric_limits<Real>::digits
            << " min_exponent=" << std::numeric_limits<Real>::min_exponent
            << " max_exponent=" << std::numeric_limits<Real>::max_exponent
            << " has_denorm=" << static_cast<int>(std::numeric_limits<Real>::has_denorm) << '\n';
  struct example { char const* name; char const* angle; };
  example const cases[] = {
    {"endpoint", "3.1415926535897932384626433832795028841971693993751"},
    {"below-cut", "3.1415926535897932384526433832795028841971693993751"},
    {"above-cut", "3.1415926535897932384726433832795028841971693993751"},
    {"near-degenerate", "0.00000000000000000001"},
    {"large-phase", "3141592.6535897932384626433832795028841971693993751"}};
  for (auto const& item: cases)
  {
    auto const parsed = boost::lexical_cast<Real>(item.angle);
    auto const parsed_high = high_type{parsed};
    auto const original = high_type{item.angle};
    auto const actual = ket::utility::exp_i<std::complex<Real>>(parsed);
    using boost::multiprecision::abs;
    using boost::multiprecision::cos;
    using boost::multiprecision::sin;
    using boost::multiprecision::sqrt;
    auto const parse_error = abs(parsed_high - original);
    // Independent high-precision evaluation at the parsed value isolates the
    // exp_i calculation from the input's decimal-to-native conversion.
    auto const real_error = high_type{actual.real()} - cos(parsed_high);
    auto const imag_error = high_type{actual.imag()} - sin(parsed_high);
    high_type const execution_error = sqrt(real_error * real_error + imag_error * imag_error);
    std::cout << item.name << " parsed=" << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << std::scientific << parsed
              << " parse_error=" << std::setprecision(12) << parse_error
              << " exp_i_error=" << execution_error << '\n';
  }
}

int main()
{
  probe<float>("float");
  probe<double>("double");
  probe<long double>("long double");
}
