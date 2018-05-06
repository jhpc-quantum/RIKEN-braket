#ifndef BRA_GATE_ADJ_PAULI_Z_HPP
# define BRA_GATE_ADJ_PAULI_Z_HPP

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
    class adj_pauli_z final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_pauli_z(qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~adj_pauli_z() = default;
# else
      ~adj_pauli_z() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_pauli_z(adj_pauli_z const&) = delete;
      adj_pauli_z& operator=(adj_pauli_z const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_pauli_z(adj_pauli_z&&) = delete;
      adj_pauli_z& operator=(adj_pauli_z&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_pauli_z(adj_pauli_z const&);
      adj_pauli_z& operator=(adj_pauli_z const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_pauli_z(adj_pauli_z&&);
      adj_pauli_z& operator=(adj_pauli_z&&);
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

