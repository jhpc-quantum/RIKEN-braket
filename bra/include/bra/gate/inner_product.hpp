#ifndef BRA_GATE_INNER_PRODUCT_HPP
# define BRA_GATE_INNER_PRODUCT_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class inner_product final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string remote_circuit_index_or_all_;

      static std::string const name_;

     public:
      inner_product(std::string const& remote_circuit_index_or_all);

      ~inner_product() = default;
      inner_product(inner_product const&) = delete;
      inner_product& operator=(inner_product const&) = delete;
      inner_product(inner_product&&) = delete;
      inner_product& operator=(inner_product&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class inner_product
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_INNER_PRODUCT_HPP
