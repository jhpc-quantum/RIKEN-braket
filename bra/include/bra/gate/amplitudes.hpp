#ifndef BRA_GATE_AMPLITUDES_HPP
# define BRA_GATE_AMPLITUDES_HPP

# include <string>
# include <vector>
# include <iosfwd>

# ifndef BRA_NO_MPI
#   include <yampi/rank.hpp>
# endif

# include <bra/gate/gate.hpp>
# include <bra/types.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class amplitudes final
      : public ::bra::gate::gate
    {
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif
      std::vector< ::bra::state_integer_type > amplitude_indices_;

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      explicit amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type > const& amplitude_indices);
      explicit amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type >&& amplitude_indices);
# else
      amplitudes(std::vector< ::bra::state_integer_type > const& amplitude_indices);
      amplitudes(std::vector< ::bra::state_integer_type >&& amplitude_indices);
# endif

      ~amplitudes() = default;
      amplitudes(amplitudes const&) = delete;
      amplitudes& operator=(amplitudes const&) = delete;
      amplitudes(amplitudes&&) = delete;
      amplitudes& operator=(amplitudes&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class amplitudes
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_AMPLITUDES_HPP
