#ifndef BRA_GATE_ADJ_PAULI_X_HPP
# define BRA_GATE_ADJ_PAULI_X_HPP

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
    class adj_pauli_x final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_pauli_x(qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~adj_pauli_x() = default;
# else
      ~adj_pauli_x() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_pauli_x(adj_pauli_x const&) = delete;
      adj_pauli_x& operator=(adj_pauli_x const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_pauli_x(adj_pauli_x&&) = delete;
      adj_pauli_x& operator=(adj_pauli_x&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_pauli_x(adj_pauli_x const&);
      adj_pauli_x& operator=(adj_pauli_x const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_pauli_x(adj_pauli_x&&);
      adj_pauli_x& operator=(adj_pauli_x&&);
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

