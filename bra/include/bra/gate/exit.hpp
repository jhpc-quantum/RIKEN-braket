#ifndef BRA_GATE_EXIT_HPP
# define BRA_GATE_EXIT_HPP

# include <string>
# include <iosfwd>

# ifndef BRA_NO_MPI
#  include <yampi/rank.hpp>
# endif

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class exit final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif // BRA_NO_MPI

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      explicit exit(yampi::rank const root);
# else // BRA_NO_MPI
      exit();
# endif // BRA_NO_MPI

      ~exit() = default;
      exit(exit const&) = delete;
      exit& operator=(exit const&) = delete;
      exit(exit&&) = delete;
      exit& operator=(exit&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class exit
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXIT_HPP
