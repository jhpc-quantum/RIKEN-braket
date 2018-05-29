#ifndef BRA_GATE_GENERATE_EVENTS_HPP
# define BRA_GATE_GENERATE_EVENTS_HPP

# include <boost/config.hpp>

# include <string>
# include <iosfwd>

# include <yampi/rank.hpp>

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
    class generate_events final
      : public ::bra::gate::gate
    {
     public:
      typedef ::bra::state::qubit_type qubit_type;

     private:
      yampi::rank root_;
      int num_events_;
      int seed_;

      static std::string const name_;

     public:
      generate_events(yampi::rank const root, int num_events, int seed);
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~generate_events() = default;
# else
      ~generate_events() { }
# endif

     private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      generate_events(generate_events const&) = delete;
      generate_events& operator=(generate_events const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      generate_events(generate_events&&) = delete;
      generate_events& operator=(generate_events&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
      generate_events(generate_events const&);
      generate_events& operator=(generate_events const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      generate_events(generate_events&&);
      generate_events& operator=(generate_events&&);
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
