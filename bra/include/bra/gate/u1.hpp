#ifndef BRA_GATE_U1_HPP
# define BRA_GATE_U1_HPP

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
    class u1 final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;
      typedef ::bra::state::real_type real_type;

     private:
      real_type phase_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      u1(real_type const phase, qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~u1() = default;
# else
      ~u1() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      u1(u1 const&) = delete;
      u1& operator=(u1 const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      u1(u1&&) = delete;
      u1& operator=(u1&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      u1(u1 const&);
      u1& operator=(u1 const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      u1(u1&&);
      u1& operator=(u1&&);
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

