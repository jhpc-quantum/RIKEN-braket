#ifndef BRA_GATE_PAULI_Y_HPP
# define BRA_GATE_PAULI_Y_HPP

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
    class pauli_y final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit pauli_y(qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~pauli_y() = default;
# else
      ~pauli_y() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      pauli_y(pauli_y const&) = delete;
      pauli_y& operator=(pauli_y const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      pauli_y(pauli_y&&) = delete;
      pauli_y& operator=(pauli_y&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      pauli_y(pauli_y const&);
      pauli_y& operator=(pauli_y const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      pauli_y(pauli_y&&);
      pauli_y& operator=(pauli_y&&);
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
