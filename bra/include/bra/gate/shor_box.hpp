#ifndef BRA_GATE_SHOR_BOX_HPP
# define BRA_GATE_SHOR_BOX_HPP

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
    class shor_box final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::bit_integer_type bit_integer_type;
      typedef ::bra::state::state_integer_type state_integer_type;

     private:
      bit_integer_type num_exponent_qubits_;
      state_integer_type divisor_;
      state_integer_type base_;

      static std::string const name_;

     public:
      shor_box(
        bit_integer_type const num_exponent_qubits,
        state_integer_type const divisor, state_integer_type const base);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~shor_box() = default;
# else
      ~shor_box() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      shor_box(shor_box const&) = delete;
      shor_box& operator=(shor_box const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      shor_box(shor_box&&) = delete;
      shor_box& operator=(shor_box&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      shor_box(shor_box const&);
      shor_box& operator=(shor_box const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      shor_box(shor_box&&);
      shor_box& operator=(shor_box&&);
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    };
  }
}


# ifdef BOOST_NO_CXX11_FINAL
#   undef final 
#   undef override 
# endif // BOOST_NO_CXX11_FINAL

#endif
