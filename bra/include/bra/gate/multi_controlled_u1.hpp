#ifndef BRA_GATE_MULTI_CONTROLLED_U1_HPP
# define BRA_GATE_MULTI_CONTROLLED_U1_HPP

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
    class multi_controlled_u1 final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      std::vector<control_qubit_type> control_qubits_;

      std::string name_;

     public:
      multi_controlled_u1(
        boost::variant<real_type, std::string> const& phase,
        std::vector<control_qubit_type> const& control_qubits);

      multi_controlled_u1(
        boost::variant<real_type, std::string> const& phase,
        std::vector<control_qubit_type>&& control_qubits);

      ~multi_controlled_u1() = default;
      multi_controlled_u1(multi_controlled_u1 const&) = delete;
      multi_controlled_u1& operator=(multi_controlled_u1 const&) = delete;
      multi_controlled_u1(multi_controlled_u1&&) = delete;
      multi_controlled_u1& operator=(multi_controlled_u1&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class multi_controlled_u1
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MULTI_CONTROLLED_U1_HPP
