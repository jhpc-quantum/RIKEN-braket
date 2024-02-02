#ifndef BRA_GATE_ADJ_SWAP_HPP
# define BRA_GATE_ADJ_SWAP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_swap final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      explicit adj_swap(qubit_type const qubit1, qubit_type const qubit2);

      ~adj_swap() = default;
      adj_swap(adj_swap const&) = delete;
      adj_swap& operator=(adj_swap const&) = delete;
      adj_swap(adj_swap&&) = delete;
      adj_swap& operator=(adj_swap&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_swap
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_SWAP_HPP
