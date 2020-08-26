#ifndef BRA_GATE_SHOR_BOX_HPP
# define BRA_GATE_SHOR_BOX_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class shor_box final
      : public ::bra::gate::gate
    {
     public:
      using bit_integer_type = ::bra::state::bit_integer_type;
      using state_integer_type = ::bra::state::state_integer_type;

     private:
      bit_integer_type num_exponent_qubits_;
      state_integer_type divisor_;
      state_integer_type base_;

      static std::string const name_;

     public:
      shor_box(
        bit_integer_type const num_exponent_qubits,
        state_integer_type const divisor, state_integer_type const base);

      ~shor_box() = default;
      shor_box(shor_box const&) = delete;
      shor_box& operator=(shor_box const&) = delete;
      shor_box(shor_box&&) = delete;
      shor_box& operator=(shor_box&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class shor_box
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SHOR_BOX_HPP
