#ifndef BRA_GATE_DEPOLARIZING_CHANNEL_HPP
# define BRA_GATE_DEPOLARIZING_CHANNEL_HPP

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
    class depolarizing_channel final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;
      typedef ::bra::state::real_type real_type;

     private:
      real_type px_;
      real_type py_;
      real_type pz_;
      int seed_;

      static std::string const name_;

     public:
      depolarizing_channel(real_type const px, real_type const py, real_type const pz, int seed);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~depolarizing_channel() = default;
# else
      ~depolarizing_channel() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      depolarizing_channel(depolarizing_channel const&) = delete;
      depolarizing_channel& operator=(depolarizing_channel const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      depolarizing_channel(depolarizing_channel&&) = delete;
      depolarizing_channel& operator=(depolarizing_channel&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      depolarizing_channel(depolarizing_channel const&);
      depolarizing_channel& operator=(depolarizing_channel const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      depolarizing_channel(depolarizing_channel&&);
      depolarizing_channel& operator=(depolarizing_channel&&);
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
