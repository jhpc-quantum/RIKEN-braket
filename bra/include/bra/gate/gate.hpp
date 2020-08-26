#ifndef BRA_GATE_GATE_HPP
# define BRA_GATE_GATE_HPP

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
