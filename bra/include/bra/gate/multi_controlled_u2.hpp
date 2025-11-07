#ifndef BRA_GATE_MULTI_CONTROLLED_U2_HPP
# define BRA_GATE_MULTI_CONTROLLED_U2_HPP

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
    class multi_controlled_u2 final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase1_;
      boost::variant<real_type, std::string> phase2_;
      qubit_type target_qubit_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      multi_controlled_u2(
        boost::variant<real_type, std::string> const& phase1, boost::variant<real_type, std::string> const& phase2,
        qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_u2(
        boost::variant<real_type, std::string> const& phase1, boost::variant<real_type, std::string> const& phase2,
        qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_u2() = default;
      multi_controlled_u2(multi_controlled_u2 const&) = delete;
      multi_controlled_u2& operator=(multi_controlled_u2 const&) = delete;
      multi_controlled_u2(multi_controlled_u2&&) = delete;
      multi_controlled_u2& operator=(multi_controlled_u2&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_u2
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_U2_HPP
