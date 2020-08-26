#ifndef BRA_GATE_SET_HPP
# define BRA_GATE_SET_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class set final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit set(qubit_type const qubit);

      ~set() = default;
      set(set const&) = delete;
      set& operator=(set const&) = delete;
      set(set&&) = delete;
      set& operator=(set&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class set
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SET_HPP
