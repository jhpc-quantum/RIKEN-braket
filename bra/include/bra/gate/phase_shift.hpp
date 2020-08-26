#ifndef BRA_GATE_PHASE_SHIFT_HPP
# define BRA_GATE_PHASE_SHIFT_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class phase_shift final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using complex_type = ::bra::state::complex_type;

     private:
      int phase_exponent_;
      complex_type phase_coefficient_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      phase_shift(
        int const phase_exponent,
        complex_type const phase_coefficient,
        qubit_type const qubit);

      ~phase_shift() = default;
      phase_shift(phase_shift const&) = delete;
      phase_shift& operator=(phase_shift const&) = delete;
      phase_shift(phase_shift&&) = delete;
      phase_shift& operator=(phase_shift&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class phase_shift
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PHASE_SHIFT_HPP
