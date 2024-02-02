#ifndef BRA_GATE_SWAP_HPP
# define BRA_GATE_SWAP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class swap final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      explicit swap(qubit_type const qubit1, qubit_type const qubit2);

      ~swap() = default;
      swap(swap const&) = delete;
      swap& operator=(swap const&) = delete;
      swap(swap&&) = delete;
      swap& operator=(swap&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class swap
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SWAP_HPP
