#ifndef BRA_GATE_DEPOLARIZING_CHANNEL_HPP
# define BRA_GATE_DEPOLARIZING_CHANNEL_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class depolarizing_channel final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type px_;
      real_type py_;
      real_type pz_;
      int seed_;

      static std::string const name_;

     public:
      depolarizing_channel(real_type const px, real_type const py, real_type const pz, int seed);

      ~depolarizing_channel() = default;
      depolarizing_channel(depolarizing_channel const&) = delete;
      depolarizing_channel& operator=(depolarizing_channel const&) = delete;
      depolarizing_channel(depolarizing_channel&&) = delete;
      depolarizing_channel& operator=(depolarizing_channel&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class depolarizing_channel
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_DEPOLARIZING_CHANNEL_HPP
