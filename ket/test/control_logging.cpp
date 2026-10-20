// Compile with and without -DKET_PRINT_LOG; no MPI installation is needed.
// Include first to check self-containment in both configurations.
#include <ket/mpi/gate/detail/append_qubits_string.hpp>

#include <cassert>
#include <string>
#include <vector>
#include <ket/control.hpp>
#include <ket/runtime_control.hpp>

namespace
{
  template <typename Character>
  auto text(std::string const& value) -> std::basic_string<Character>
  { return {value.begin(), value.end()}; }

  template <typename Character>
  auto check(std::basic_string<Character> const& actual, std::string const& suffix) -> void
  {
#ifdef KET_PRINT_LOG
    assert(actual == text<Character>("Gate" + suffix));
#else
    (void)suffix;
    assert(actual == text<Character>("Gate"));
#endif
  }

  template <typename Character>
  auto check_logging() -> void
  {
    namespace detail = ket::mpi::gate::detail;
    using qubit = ket::qubit<>;
    using positive = ket::control<qubit, 1u>;
    using negative = ket::control<qubit, 0u>;
    using dynamic = ket::runtime_control<qubit>;
    auto const base = text<Character>("Gate");

    check(detail::append_qubits_string(base), "");
    check(detail::append_qubits_string(base, qubit{3u}), " 3");
    check(detail::append_qubits_string(base, qubit{0u}, qubit{1u}), " 0 1");
    check(detail::append_qubits_string(base, positive{3u}), " 3#1");
    check(detail::append_qubits_string(base, negative{3u}), " 3#0");
    check(detail::append_qubits_string(base, dynamic{qubit{3u}, 0u}), " 3#0");
    check(detail::append_qubits_string(
      base, qubit{0u}, qubit{1u}, positive{2u}, negative{3u}, dynamic{qubit{4u}, 1u}),
      " 0 1 2#1 3#0 4#1");
    check(detail::append_qubits_string(base, negative{2u}, positive{3u}), " 2#0 3#1");

    auto const targets = std::vector<qubit>{qubit{0u}, qubit{1u}};
    auto const positives = std::vector<positive>{positive{2u}, positive{3u}};
    auto const negatives = std::vector<negative>{negative{2u}, negative{3u}};
    auto const mixed = std::vector<dynamic>{dynamic{qubit{2u}, 0u}, dynamic{qubit{3u}, 1u}};
    auto const no_targets = std::vector<qubit>{};
    auto const no_controls = std::vector<dynamic>{};
    check(detail::runtime::append_qubits_string(base, targets), " 0 1");
    check(detail::runtime::append_qubits_string(base, positives), " 2#1 3#1");
    check(detail::runtime::append_qubits_string(base, negatives), " 2#0 3#0");
    check(detail::runtime::append_qubits_string(base, mixed), " 2#0 3#1");
    check(detail::runtime::append_qubits_string(base, qubit{0u}, positives), " 0 2#1 3#1");
    check(detail::runtime::append_qubits_string(base, qubit{0u}, negatives), " 0 2#0 3#0");
    check(detail::runtime::append_qubits_string(base, qubit{0u}, mixed), " 0 2#0 3#1");
    check(detail::runtime::append_qubits_string(base, targets, positives), " 0 1 2#1 3#1");
    check(detail::runtime::append_qubits_string(base, targets, negatives), " 0 1 2#0 3#0");
    check(detail::runtime::append_qubits_string(base, targets, mixed), " 0 1 2#0 3#1");
    check(detail::runtime::append_qubits_string(base, no_targets), "");
    check(detail::runtime::append_qubits_string(base, no_controls), "");
    check(detail::runtime::append_qubits_string(base, qubit{0u}, no_controls), " 0");
    check(detail::runtime::append_qubits_string(base, targets, no_controls), " 0 1");
    check(detail::runtime::append_qubits_string(base, no_targets, mixed), " 2#0 3#1");
    check(detail::runtime::append_qubits_string(base, no_targets, no_controls), "");

    // Prefixes (including adjoint names and parameters) must remain untouched.
    auto const prefix = text<Character>("Adj(CCH) theta=0.5");
    auto const result = detail::append_qubits_string(prefix, qubit{0u}, negative{2u}, positive{3u});
#ifdef KET_PRINT_LOG
    assert(result == prefix + text<Character>(" 0 2#0 3#1"));
#else
    assert(result == prefix);
#endif
  }
}

int main()
{
  check_logging<char>();
  check_logging<wchar_t>();
}
