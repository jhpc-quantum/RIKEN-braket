#ifndef BRA_GATE_END_FUSION_HPP
# define BRA_GATE_END_FUSION_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class end_fusion final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      static std::string const name_;

     public:
      end_fusion();

      ~end_fusion() = default;
      end_fusion(end_fusion const&) = delete;
      end_fusion& operator=(end_fusion const&) = delete;
      end_fusion(end_fusion&&) = delete;
      end_fusion& operator=(end_fusion&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class end_fusion
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_END_FUSION_HPP
