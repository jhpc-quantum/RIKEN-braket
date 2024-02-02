#ifndef BRA_GATE_EXPONENTIAL_SWAP_HPP
# define BRA_GATE_EXPONENTIAL_SWAP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exponential_swap final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      explicit exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);

      ~exponential_swap() = default;
      exponential_swap(exponential_swap const&) = delete;
      exponential_swap& operator=(exponential_swap const&) = delete;
      exponential_swap(exponential_swap&&) = delete;
      exponential_swap& operator=(exponential_swap&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class exponential_swap
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPONENTIAL_SWAP_HPP
