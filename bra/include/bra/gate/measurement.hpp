#ifndef BRA_GATE_MEASUREMENT_HPP
# define BRA_GATE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <string>
# include <iosfwd>

# include <yampi/rank.hpp>

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
    class measurement final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      yampi::rank root_;

      static std::string const name_;

     public:
      explicit measurement(yampi::rank const root);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~measurement() = default;
# else
      ~measurement() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      measurement(measurement const&) = delete;
      measurement& operator=(measurement const&) = delete;
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      measurement(measurement&&) = delete;
      measurement& operator=(measurement&&) = delete;
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      measurement(measurement const&);
      measurement& operator=(measurement const&);
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      measurement(measurement&&);
      measurement& operator=(measurement&&);
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES
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
