#ifndef BRA_GATE_MEASUREMENT_HPP
# define BRA_GATE_MEASUREMENT_HPP

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
    class measurement final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      explicit measurement(yampi::rank const root);
# else
      measurement();
# endif

      ~measurement() = default;
      measurement(measurement const&) = delete;
      measurement& operator=(measurement const&) = delete;
      measurement(measurement&&) = delete;
      measurement& operator=(measurement&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const) const override;
    }; // class measurement
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_MEASUREMENT_HPP
