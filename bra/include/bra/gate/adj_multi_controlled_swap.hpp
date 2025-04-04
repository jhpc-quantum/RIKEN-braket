#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_SWAP_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_SWAP_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_swap final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit1_;
      qubit_type target_qubit2_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_swap(
        qubit_type const target_qubit1, qubit_type const target_qubit2,
        std::vector<control_qubit_type> const& control_qubits);

      adj_multi_controlled_swap(
        qubit_type const target_qubit1, qubit_type const target_qubit2,
        std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_swap() = default;
      adj_multi_controlled_swap(adj_multi_controlled_swap const&) = delete;
      adj_multi_controlled_swap& operator=(adj_multi_controlled_swap const&) = delete;
      adj_multi_controlled_swap(adj_multi_controlled_swap&&) = delete;
      adj_multi_controlled_swap& operator=(adj_multi_controlled_swap&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_swap
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_SWAP_HPP
