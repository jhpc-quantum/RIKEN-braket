#ifndef BRA_GATE_ADJ_MULTI_CONTROLLED_U2_HPP
# define BRA_GATE_ADJ_MULTI_CONTROLLED_U2_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_multi_controlled_u2 final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase1_;
      real_type phase2_;
      qubit_type target_qubit_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      adj_multi_controlled_u2(
        real_type const& phase1, real_type const& phase2,
        qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits);

      ~adj_multi_controlled_u2() = default;
      adj_multi_controlled_u2(adj_multi_controlled_u2 const&) = delete;
      adj_multi_controlled_u2& operator=(adj_multi_controlled_u2 const&) = delete;
      adj_multi_controlled_u2(adj_multi_controlled_u2&&) = delete;
      adj_multi_controlled_u2& operator=(adj_multi_controlled_u2&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_multi_controlled_u2
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_MULTI_CONTROLLED_U2_HPP
