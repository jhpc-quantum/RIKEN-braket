#ifndef BRA_GATE_PROJECTIVE_MEASUREMENT_HPP
# define BRA_GATE_PROJECTIVE_MEASUREMENT_HPP

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
    class projective_measurement final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit projective_measurement(qubit_type const qubit);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~projective_measurement() = default;
# else
      ~projective_measurement() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      projective_measurement(projective_measurement const&) = delete;
      projective_measurement& operator=(projective_measurement const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      projective_measurement(projective_measurement&&) = delete;
      projective_measurement& operator=(projective_measurement&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      projective_measurement(projective_measurement const&);
      projective_measurement& operator=(projective_measurement const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      projective_measurement(projective_measurement&&);
      projective_measurement& operator=(projective_measurement&&);
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

