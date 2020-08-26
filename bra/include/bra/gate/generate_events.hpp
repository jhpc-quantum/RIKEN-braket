#ifndef BRA_GATE_GENERATE_EVENTS_HPP
# define BRA_GATE_GENERATE_EVENTS_HPP

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
    class generate_events final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif
      int num_events_;
      int seed_;

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      generate_events(yampi::rank const root, int num_events, int seed);
# else
      generate_events(int num_events, int seed);
# endif

      ~generate_events() = default;
      generate_events(generate_events const&) = delete;
      generate_events& operator=(generate_events const&) = delete;
      generate_events(generate_events&&) = delete;
      generate_events& operator=(generate_events&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class generate_events
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_GENERATE_EVENTS_HPP
