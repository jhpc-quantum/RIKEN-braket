#ifndef BRA_GATE_RESET_HPP
# define BRA_GATE_RESET_HPP

# include <string>
# include <iosfwd>

# ifndef BRA_NO_MPI
#   include <yampi/rank.hpp>
# endif

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class reset final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      reset(qubit_type const qubit, yampi::rank const root);
# else
      explicit reset(qubit_type const qubit);
# endif

      ~reset() = default;
      reset(reset const&) = delete;
      reset& operator=(reset const&) = delete;
      reset(reset&&) = delete;
      reset& operator=(reset&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class reset
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_RESET_HPP
