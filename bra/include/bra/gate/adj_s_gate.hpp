#ifndef BRA_GATE_ADJ_S_GATE_HPP
# define BRA_GATE_ADJ_S_GATE_HPP

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
    class adj_s_gate final
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
      adj_s_gate(
        complex_type const phase_coefficient,
        qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~adj_s_gate() = default;
# else
      ~adj_s_gate() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_s_gate(adj_s_gate const&) = delete;
      adj_s_gate& operator=(adj_s_gate const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_s_gate(adj_s_gate&&) = delete;
      adj_s_gate& operator=(adj_s_gate&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      adj_s_gate(adj_s_gate const&);
      adj_s_gate& operator=(adj_s_gate const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      adj_s_gate(adj_s_gate&&);
      adj_s_gate& operator=(adj_s_gate&&);
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

