#ifndef BRA_GATE_S_GATE_HPP
# define BRA_GATE_S_GATE_HPP

# include <boost/config.hpp>

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>

# ifdef BOOST_NO_CXX11_FINAL
#   define final 
#   define override 
# endif // BOOST_NO_CXX11_FINAL


namespace bra
{
  namespace gate
  {
    class s_gate final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;
      typedef ::bra::state::complex_type complex_type;

     private:
      complex_type phase_coefficient_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      s_gate(
        complex_type const phase_coefficient,
        qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~s_gate() = default;
# else
      ~s_gate() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      s_gate(s_gate const&) = delete;
      s_gate& operator=(s_gate const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      s_gate(s_gate&&) = delete;
      s_gate& operator=(s_gate&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      s_gate(s_gate const&);
      s_gate& operator=(s_gate const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      s_gate(s_gate&&);
      s_gate& operator=(s_gate&&);
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
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

