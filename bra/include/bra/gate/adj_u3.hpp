#ifndef BRA_GATE_ADJ_U3_HPP
# define BRA_GATE_ADJ_U3_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_u3 final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase1_;
      boost::variant<real_type, std::string> phase2_;
      boost::variant<real_type, std::string> phase3_;
      qubit_type qubit_;

      static std::string const name_;

     public:
      adj_u3(
        boost::variant<real_type, std::string> const& phase1, boost::variant<real_type, std::string> const& phase2, boost::variant<real_type, std::string> const& phase3,
        qubit_type const qubit);

      ~adj_u3() = default;
      adj_u3(adj_u3 const&) = delete;
      adj_u3& operator=(adj_u3 const&) = delete;
      adj_u3(adj_u3&&) = delete;
      adj_u3& operator=(adj_u3&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_u3
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_U3_HPP
