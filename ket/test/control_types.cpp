// g++ -std=c++14 -Wall -Wextra -pedantic -Iket/include ket/test/control_types.cpp -o /tmp/control_types
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#include <ket/control.hpp>
#include <ket/control_io.hpp>
#include <ket/qubit_io.hpp>
#include <ket/runtime_control.hpp>
#include <ket/meta/bit_integer_of.hpp>
#include <ket/utility/none_in_state_vector.hpp>
#include <ket/utility/meta/is_control_qubits_range.hpp>

namespace
{
  using qubit = ket::qubit<>;
  using positive = ket::control<qubit>;
  using negative = ket::control<qubit, 0u>;
  using runtime = ket::runtime_control<qubit>;
  using namespace ket::literals;

  template <typename Original, typename Positive, typename Negative>
  constexpr auto check_literals(Original const original, Positive const positive_control, Negative const negative_control) -> bool
  {
    static_assert(std::is_same<Original, Positive>::value, "positive literal preserves original type");
    static_assert(std::is_same<Negative, decltype(ket::make_control<0u>(ket::remove_control(original)))>::value,
                  "negative literal preserves underlying qubit type");
    return ket::control_value(original) == 1u
      and ket::control_value(positive_control) == 1u
      and ket::control_value(negative_control) == 0u
      and ket::remove_control(original) == ket::remove_control(positive_control)
      and ket::remove_control(original) == ket::remove_control(negative_control)
      and ket::control_value(ket::make_runtime_control(positive_control)) == 1u
      and ket::control_value(ket::make_runtime_control(negative_control)) == 0u;
  }

  static_assert(check_literals(3_cq, 3_pcq, 3_ncq), "literal family _cq");
  static_assert(noexcept(3_pcq) and noexcept(3_ncq), "literal construction is noexcept");
  static_assert(check_literals(3_cqs, 3_pcqs, 3_ncqs), "literal family _cqs");
  static_assert(noexcept(3_pcqs) and noexcept(3_ncqs), "literal construction is noexcept");
  static_assert(check_literals(3_cql, 3_pcql, 3_ncql), "literal family _cql");
  static_assert(noexcept(3_pcql) and noexcept(3_ncql), "literal construction is noexcept");
  static_assert(check_literals(3_cqll, 3_pcqll, 3_ncqll), "literal family _cqll");
  static_assert(noexcept(3_pcqll) and noexcept(3_ncqll), "literal construction is noexcept");
  static_assert(check_literals(3_cq8, 3_pcq8, 3_ncq8), "literal family _cq8");
  static_assert(noexcept(3_pcq8) and noexcept(3_ncq8), "literal construction is noexcept");
  static_assert(check_literals(3_cq16, 3_pcq16, 3_ncq16), "literal family _cq16");
  static_assert(noexcept(3_pcq16) and noexcept(3_ncq16), "literal construction is noexcept");
  static_assert(check_literals(3_cq32, 3_pcq32, 3_ncq32), "literal family _cq32");
  static_assert(noexcept(3_pcq32) and noexcept(3_ncq32), "literal construction is noexcept");
  static_assert(check_literals(3_cq64, 3_pcq64, 3_ncq64), "literal family _cq64");
  static_assert(noexcept(3_pcq64) and noexcept(3_ncq64), "literal construction is noexcept");

  static_assert(std::is_same<positive, ket::control<qubit, 1u>>::value, "compatible default");
  static_assert(ket::meta::base_v<qubit> == 2u, "qubit base");
  static_assert(ket::meta::base_v<negative> == 2u, "control base");
  static_assert(ket::meta::base_v<runtime> == 2u, "runtime base");
  static_assert(ket::control_value(3_cq) == 1u, "compatible literal");
  static_assert(ket::control_value(negative{3_q}) == 0u, "negative control");
  static_assert(ket::control_value(ket::make_control(3_q)) == 1u, "compatible helper");
  static_assert(ket::control_value(ket::make_control<0u>(3_q)) == 0u, "negative helper");
  static_assert(ket::control_value(runtime{}) == 1u, "runtime default");
  static_assert(ket::control_value(runtime{3_q}) == 1u, "runtime qubit constructor");
  static_assert(ket::control_value(runtime{3_q, 0}) == 0u, "constexpr runtime value");
  static_assert(ket::control_value(runtime{negative{3_q}}) == 0u, "explicit conversion");
  static_assert(ket::control_value(ket::make_runtime_control(negative{3_q})) == 0u, "conversion helper");
  static_assert(ket::control_value(ket::make_runtime_control(3_q, 0)) == 0u, "runtime helper");
  static_assert(ket::utility::none_in_state_vector(3u, runtime{3_q, 0u}), "Negative runtime wire outside vector");
  static_assert(not ket::utility::none_in_state_vector(3u, runtime{2_q, 1u}), "Positive runtime wire inside vector");
  static_assert(ket::utility::none_in_state_vector(3u, negative{3_q}, positive{4_q}), "Bounds ignore control values");
  static_assert(ket::utility::none_in_state_vector(3u, 4u), "Raw positional bounds remain supported");
  static_assert(ket::remove_control(negative{3_q}) == 3_q, "explicit removal");
  static_assert(ket::remove_control(runtime{3_q, 0}) == 3_q, "runtime removal");
  static_assert(ket::meta::is_control<negative>::value, "negative trait");
  static_assert(ket::meta::is_control<runtime>::value, "runtime trait");
  static_assert(ket::meta::is_control_cvref<negative const&>::value, "negative cvref");
  static_assert(ket::meta::is_control_cvref<runtime const&>::value, "runtime cvref");
  static_assert(not ket::meta::is_control<qubit>::value, "target is not a control");
  static_assert(ket::utility::meta::is_control_qubits_range<std::vector<positive>>::value, "old range");
  static_assert(ket::utility::meta::is_control_qubits_range<std::vector<negative>>::value, "negative range");
  static_assert(ket::utility::meta::is_control_qubits_range<std::vector<runtime>>::value, "mixed range");
  static_assert(std::is_same<ket::meta::state_integer_t<runtime>, std::uint64_t>::value, "state integer");
  static_assert(std::is_same<ket::meta::bit_integer_t<negative>, unsigned int>::value, "bit integer");
  static_assert(not std::is_convertible<negative, positive>::value, "no polarity loss");
  static_assert(not std::is_constructible<positive, negative>::value, "no explicit polarity loss");
  static_assert(not std::is_constructible<negative, positive>::value, "no explicit reverse polarity loss");
  static_assert(not std::is_convertible<negative, runtime>::value, "conversion is explicit");
  static_assert(not std::is_convertible<runtime, positive>::value, "no runtime value loss");
  static_assert(not std::is_constructible<positive, runtime>::value, "no explicit runtime value loss");
  static_assert(not std::is_convertible<negative, qubit>::value, "no implicit removal");
  static_assert(not std::is_convertible<runtime, qubit>::value, "no implicit runtime removal");
  static_assert(not std::is_constructible<runtime, qubit, double>::value, "integer values only");

  template <typename Control>
  constexpr auto check_arithmetic(Control control) -> bool
  {
    auto const value = ket::control_value(control);
    if (ket::remove_control(control++) != 3_q or ket::remove_control(control) != 4_q) return false;
    if (ket::remove_control(control--) != 4_q or ket::remove_control(control) != 3_q) return false;
    ++control; --control; control += 2; control -= 1;
    return ket::remove_control(control) == 4_q and ket::control_value(control) == value
      and ket::remove_control(control + 1) == 5_q and ket::remove_control(1 + control) == 5_q
      and ket::remove_control(control - 1) == 3_q and control - Control{2_q} == 2u
      and control - 2_q == 2u
      and control > 3_q and 3_q < control and control >= 4_q and 4_q <= control
      and control == 4_q and 4_q == control and control != 3_q and 3_q != control
      and control <= Control{4_q} and control >= Control{4_q}
      and control < Control{5_q} and control > Control{3_q}
      and (std::uint64_t{1u} << control) == 16u and (std::uint64_t{16u} >> control) == 1u;
  }
  static_assert(check_arithmetic(positive{3_q}), "positive arithmetic");
  static_assert(check_arithmetic(negative{3_q}), "negative arithmetic");
  static_assert(check_arithmetic(runtime{3_q, 0}), "runtime arithmetic preserves value");
  static_assert(6_q - runtime{4_q, 0} == 2u, "qubit minus runtime control");
  static_assert(runtime{3_q, 0} == runtime{3_q, 1}, "comparison refers to wire position");

  template <typename Integer>
  auto check_invalid(Integer const value) -> void
  {
    auto threw = false;
    try { auto const control = runtime{3_q, value}; (void)control; }
    catch (std::out_of_range const&) { threw = true; }
    assert(threw);
  }
}

// Trait-only mock: this does not implement qudit execution.
struct digit
{
  using state_integer_type = unsigned char;
  using bit_integer_type = unsigned int;
};
namespace ket { namespace meta {
  template <> struct base<digit> : std::integral_constant<unsigned char, 3u> { };
} }
static_assert(ket::control_value(ket::control<digit>{digit{}}) == 2u, "base-dependent default");
static_assert(ket::control_value(ket::runtime_control<digit>{digit{}, 1}) == 1u, "nonbinary value");

int main()
{
  auto controls = std::vector<runtime>{runtime{3_q, 1}, runtime{4_q, 0}, runtime{5_q, 1}};
  assert(ket::control_value(controls[0]) == 1u);
  assert(ket::control_value(controls[1]) == 0u);
  assert(ket::remove_control(controls[2]) == 5_q);
  controls[1].qubit() = 6_q;
  assert(ket::control_value(controls[1]) == 0u and ket::remove_control(controls[1]) == 6_q);
  std::sort(controls.begin(), controls.end());
  assert(ket::remove_control(controls.back()) == 6_q and ket::control_value(controls.back()) == 0u);
  assert(ket::utility::runtime::ranges::none_in_state_vector(3u, controls));
  assert(not ket::utility::runtime::ranges::none_in_state_vector(4u, controls));
  auto empty_controls = std::vector<runtime>{};
  assert(ket::utility::runtime::ranges::none_in_state_vector(3u, empty_controls));
  check_invalid(-1);
  check_invalid(2);
  check_invalid(2u);
  check_invalid(std::numeric_limits<std::uint64_t>::max());
  check_invalid(std::numeric_limits<int>::min());
  auto threw = false;
  try { ket::runtime_control<digit> const control{digit{}, 256u}; (void)control; }
  catch (std::out_of_range const&) { threw = true; }
  assert(threw); // Narrowing must not turn an invalid value into zero.

  auto negative_control = negative{3_q};
  auto stream = std::stringstream{};
  stream << negative_control;
  assert(stream.str() == "3");
}
