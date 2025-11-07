#ifndef BRA_GATE_GATE_HPP
# define BRA_GATE_GATE_HPP

# include <string>
# include <sstream>
# include <iosfwd>

# include <boost/variant/static_visitor.hpp>

# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    namespace gate_detail
    {
      template <typename T>
      struct output_visitor
        : public boost::static_visitor<std::string>
      {
        std::string operator()(T const value) const
        { std::ostringstream oss; oss << value; return oss.str(); }

        std::string operator()(std::string const& string) const { return string; }
      }; // struct output_visitor<T>
    } // namespace gate_detail

    class gate
    {
     public:
      gate() = default;
      virtual ~gate() = default;

      gate(gate const&) = delete;
      gate& operator=(gate const&) = delete;
      gate(gate&&) = delete;
      gate& operator=(gate&&) = delete;

      ::bra::state& apply(::bra::state& state) const { return do_apply(state); }
      std::string const& name() const { return do_name(); }
      std::string representation() const;

     protected:
      virtual ::bra::state& do_apply(::bra::state& state) const = 0;
      virtual std::string const& do_name() const = 0;
      virtual std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const = 0;
    }; // class gate

    inline ::bra::state& operator<<(::bra::state& state, ::bra::gate::gate const& gate)
    { return gate.apply(state); }
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_GATE_HPP
