#ifndef BRA_GATE_CLEAR_HPP
# define BRA_GATE_CLEAR_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class clear final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit clear(qubit_type const qubit);

      ~clear() = default;
      clear(clear const&) = delete;
      clear& operator=(clear const&) = delete;
      clear(clear&&) = delete;
      clear& operator=(clear&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class clear
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CLEAR_HPP
