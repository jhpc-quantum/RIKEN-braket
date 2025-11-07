#ifndef BRA_GATE_MULTI_CONTROLLED_EXPONENTIAL_SWAP_HPP
# define BRA_GATE_MULTI_CONTROLLED_EXPONENTIAL_SWAP_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class multi_controlled_exponential_swap final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      qubit_type target_qubit1_;
      qubit_type target_qubit2_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      multi_controlled_exponential_swap(
        boost::variant<real_type, std::string> const& phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_exponential_swap(
        boost::variant<real_type, std::string> const& phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_exponential_swap() = default;
      multi_controlled_exponential_swap(multi_controlled_exponential_swap const&) = delete;
      multi_controlled_exponential_swap& operator=(multi_controlled_exponential_swap const&) = delete;
      multi_controlled_exponential_swap(multi_controlled_exponential_swap&&) = delete;
      multi_controlled_exponential_swap& operator=(multi_controlled_exponential_swap&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_exponential_swap
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_EXPONENTIAL_SWAP_HPP
