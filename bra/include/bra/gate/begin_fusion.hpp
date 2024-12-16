#ifndef BRA_GATE_BEGIN_FUSION_HPP
# define BRA_GATE_BEGIN_FUSION_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class begin_fusion final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<qubit_type> fused_qubits_;

      static std::string const name_;

     public:
      explicit begin_fusion(std::vector<qubit_type> const& fused_qubits);
      explicit begin_fusion(std::vector<qubit_type>&& fused_qubits);

      ~begin_fusion() = default;
      begin_fusion(begin_fusion const&) = delete;
      begin_fusion& operator=(begin_fusion const&) = delete;
      begin_fusion(begin_fusion&&) = delete;
      begin_fusion& operator=(begin_fusion&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class begin_fusion
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_BEGIN_FUSION_HPP
