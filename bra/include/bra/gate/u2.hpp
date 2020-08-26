#ifndef BRA_GATE_U2_HPP
# define BRA_GATE_U2_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class u2 final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase1_;
      real_type phase2_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      u2(
        real_type const phase1, real_type const phase2,
        qubit_type const qubit);

      ~u2() = default;
      u2(u2 const&) = delete;
      u2& operator=(u2 const&) = delete;
      u2(u2&&) = delete;
      u2& operator=(u2&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class u2
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_U2_HPP
