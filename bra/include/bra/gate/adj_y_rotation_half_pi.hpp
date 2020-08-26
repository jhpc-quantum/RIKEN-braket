#ifndef BRA_GATE_ADJ_Y_ROTATION_HALF_PI_HPP
# define BRA_GATE_ADJ_Y_ROTATION_HALF_PI_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_y_rotation_half_pi final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_y_rotation_half_pi(qubit_type const qubit);

      ~adj_y_rotation_half_pi() = default;
      adj_y_rotation_half_pi(adj_y_rotation_half_pi const&) = delete;
      adj_y_rotation_half_pi& operator=(adj_y_rotation_half_pi const&) = delete;
      adj_y_rotation_half_pi(adj_y_rotation_half_pi&&) = delete;
      adj_y_rotation_half_pi& operator=(adj_y_rotation_half_pi&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_y_rotation_half_pi
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_Y_ROTATION_HALF_PI_HPP
