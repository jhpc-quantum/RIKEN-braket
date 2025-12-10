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
# ifndef BRA_NO_MPI
      yampi::rank root_;
# endif
      int const precision_;

      static std::string const name_;

     public:
# ifndef BRA_NO_MPI
      explicit measurement(yampi::rank const root, int const precision);
# else
      measurement(int const precision);
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
