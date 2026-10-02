#ifndef BRA_GATE_GLOBAL_PHASE_HPP
# define BRA_GATE_GLOBAL_PHASE_HPP

# include <iosfwd>
# include <string>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class global_phase final
      : public ::bra::gate::gate
    {
     public:
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;

      static std::string const name_;

     public:
      explicit global_phase(boost::variant<real_type, std::string> const& phase);

      ~global_phase() = default;
      global_phase(global_phase const&) = delete;
      global_phase& operator=(global_phase const&) = delete;
      global_phase(global_phase&&) = delete;
      global_phase& operator=(global_phase&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class global_phase
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_GLOBAL_PHASE_HPP
