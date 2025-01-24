#ifndef BRA_GATE_ADJ_NOT_HPP
# define BRA_GATE_ADJ_NOT_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_not_ final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_not_(qubit_type const qubit);

      ~adj_not_() = default;
      adj_not_(adj_not_ const&) = delete;
      adj_not_& operator=(adj_not_ const&) = delete;
      adj_not_(adj_not_&&) = delete;
      adj_not_& operator=(adj_not_&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_not_
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_NOT_HPP
