#ifndef BRA_GATE_PROJECTIVE_MEASUREMENT_HPP
# define BRA_GATE_PROJECTIVE_MEASUREMENT_HPP

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
    class projective_measurement final
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
      projective_measurement(qubit_type const qubit, yampi::rank const root);
# else
      explicit projective_measurement(qubit_type const qubit);
# endif

      ~projective_measurement() = default;
      projective_measurement(projective_measurement const&) = delete;
      projective_measurement& operator=(projective_measurement const&) = delete;
      projective_measurement(projective_measurement&&) = delete;
      projective_measurement& operator=(projective_measurement&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class projective_measurement
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PROJECTIVE_MEASUREMENT_HPP
