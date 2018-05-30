#ifndef BRA_GATE_PAULI_Z_HPP
# define BRA_GATE_PAULI_Z_HPP

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
    class pauli_z final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit pauli_z(qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~pauli_z() = default;
# else
      ~pauli_z() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      pauli_z(pauli_z const&) = delete;
      pauli_z& operator=(pauli_z const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      pauli_z(pauli_z&&) = delete;
      pauli_z& operator=(pauli_z&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      pauli_z(pauli_z const&);
      pauli_z& operator=(pauli_z const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      pauli_z(pauli_z&&);
      pauli_z& operator=(pauli_z&&);
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

