#ifndef BRA_GATE_NOT_HPP
# define BRA_GATE_NOT_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class not_ final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit not_(qubit_type const qubit);

      ~not_() = default;
      not_(not_ const&) = delete;
      not_& operator=(not_ const&) = delete;
      not_(not_&&) = delete;
      not_& operator=(not_&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class not_
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_NOT_HPP
