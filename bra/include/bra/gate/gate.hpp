#ifndef BRA_GATE_GATE_HPP
# define BRA_GATE_GATE_HPP

# include <boost/config.hpp>

# include <string>
# include <iosfwd>

# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class gate
    {
     public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      gate() = default;
      virtual ~gate() = default;
# else
      gate() { }
      virtual ~gate() { }
# endif

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      gate(gate const&) = delete;
      gate& operator=(gate const&) = delete;
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      gate(gate&&) = delete;
      gate& operator=(gate&&) = delete;
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES

# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
     private:
      gate(gate const&);
      gate& operator=(gate const&);
#  ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      gate(gate&&);
      gate& operator=(gate&&);
#  endif // BOOST_NO_CXX11_RVALUE_REFERENCES

     public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS
      ::bra::state& apply(::bra::state& state) const { return do_apply(state); }
      std::string const& name() const { return do_name(); }
      std::string representation() const;

     protected:
      virtual ::bra::state& do_apply(::bra::state& state) const = 0;
      virtual std::string const& do_name() const = 0;
      virtual std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const = 0;
    };

    inline ::bra::state& operator<<(::bra::state& state, ::bra::gate::gate const& gate)
    { return gate.apply(state); }
  }
}


#endif

