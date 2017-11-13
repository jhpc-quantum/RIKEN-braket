#ifndef BRA_GATE_CONTROLLED_PHASE_SHIFT_HPP
# define BRA_GATE_CONTROLLED_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>

# ifdef BOOST_NO_CXX11_FINAL
#  define final 
#  define override 
# endif // BOOST_NO_CXX11_FINAL


namespace bra
{
  namespace gate
  {
    class controlled_phase_shift final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;
      typedef ::bra::state::control_qubit_type control_qubit_type;
      typedef ::bra::state::complex_type complex_type;

     private:
      int phase_exponent_;
      complex_type phase_coefficient_;
      qubit_type target_qubit_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      controlled_phase_shift(
        int const phase_exponent,
        complex_type const phase_coefficient,
        qubit_type const target_qubit,
        control_qubit_type const control_qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~controlled_phase_shift() = default;
# else
      ~controlled_phase_shift() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      controlled_phase_shift(controlled_phase_shift const&) = delete;
      controlled_phase_shift& operator=(controlled_phase_shift const&) = delete;
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      controlled_phase_shift(controlled_phase_shift&&) = delete;
      controlled_phase_shift& operator=(controlled_phase_shift&&) = delete;
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      controlled_phase_shift(controlled_phase_shift const&);
      controlled_phase_shift& operator=(controlled_phase_shift const&);
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      controlled_phase_shift(controlled_phase_shift&&);
      controlled_phase_shift& operator=(controlled_phase_shift&&);
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    };
  }
}


# ifdef BOOST_NO_CXX11_FINAL
#   undef final 
#   undef override 
# endif // BOOST_NO_CXX11_FINAL

#endif

