#ifndef BRA_GATE_FIDELITY_HPP
# define BRA_GATE_FIDELITY_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class fidelity final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string remote_circuit_index_or_all_;

      static std::string const name_;

     public:
      fidelity(std::string const& remote_circuit_index_or_all);

      ~fidelity() = default;
      fidelity(fidelity const&) = delete;
      fidelity& operator=(fidelity const&) = delete;
      fidelity(fidelity&&) = delete;
      fidelity& operator=(fidelity&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class fidelity
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_FIDELITY_HPP
