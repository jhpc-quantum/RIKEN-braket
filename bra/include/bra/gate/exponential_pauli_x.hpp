#ifndef BRA_GATE_EXPONENTIAL_PAULI_X_HPP
# define BRA_GATE_EXPONENTIAL_PAULI_X_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_pauli_x final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      exponential_pauli_x(real_type const phase, qubit_type const qubit);

      ~exponential_pauli_x() = default;
      exponential_pauli_x(exponential_pauli_x const&) = delete;
      exponential_pauli_x& operator=(exponential_pauli_x const&) = delete;
      exponential_pauli_x(exponential_pauli_x&&) = delete;
      exponential_pauli_x& operator=(exponential_pauli_x&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_pauli_x
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_PAULI_X_HPP
