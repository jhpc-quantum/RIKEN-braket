#ifndef BRA_GATE_ADJ_U1_HPP
# define BRA_GATE_ADJ_U1_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_u1 final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      adj_u1(real_type const phase, control_qubit_type const control_qubit);

      ~adj_u1() = default;
      adj_u1(adj_u1 const&) = delete;
      adj_u1& operator=(adj_u1 const&) = delete;
      adj_u1(adj_u1&&) = delete;
      adj_u1& operator=(adj_u1&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_u1
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_U1_HPP
