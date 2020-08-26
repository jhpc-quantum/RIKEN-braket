#ifndef BRA_GATE_X_ROTATION_HALF_PI_HPP
# define BRA_GATE_X_ROTATION_HALF_PI_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class x_rotation_half_pi final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit x_rotation_half_pi(qubit_type const qubit);

      ~x_rotation_half_pi() = default;
      x_rotation_half_pi(x_rotation_half_pi const&) = delete;
      x_rotation_half_pi& operator=(x_rotation_half_pi const&) = delete;
      x_rotation_half_pi(x_rotation_half_pi&&) = delete;
      x_rotation_half_pi& operator=(x_rotation_half_pi&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class x_rotation_half_pi
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_X_ROTATION_HALF_PI_HPP
