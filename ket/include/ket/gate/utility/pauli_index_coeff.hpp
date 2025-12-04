#ifndef KET_GATE_UTILITY_PAULI_INDEX_COEFF_HPP
# define KET_GATE_UTILITY_PAULI_INDEX_COEFF_HPP

# include <string>
# include <utility>

# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace utility
    {
      // pauli_index_coeff: get a pair of "Pauli index" and "Coefficient" for a given index
      //   Pauli index: n', or index whose corresponding data exists
      //   Coefficient: C(n)
      //
      // |Psi'> = (\sigma_{i_{N-1}} \otimes ... \otimes \sigma_{i_1} \otimes \sigma_{i_0}) |Psi>
      //   |Psi'> = \sum a'(n) |n>, |Psi> = \sum a(n) |n>
      // a'(n_{N-1} ... n_1 n_0) = C(n_{N-1}) ... C(n_1) C(n_0) a(n'_{N-1} ... n_1 n_0)
      //   n'_k = ^n_k if i_k = X, Y
      //        =  n_k if i_k = Z, I
      //   C(n_k) =  1 if i_k = X
      //          = -i if i_k = Y and n_k = 0
      //          =  i if i_k = Y and n_k = 1
      //          =  1 if i_k = Z and n_k = 0
      //          = -1 if i_k = Z and n_k = 1
      //          =  1 if i_k = I
      template <typename Complex, typename StateInteger>
      inline auto pauli_index_coeff(std::string const& pauli_string, StateInteger const index)
      -> std::pair<StateInteger, Complex>
      {
        auto result_index = index;
        using real_type = ::ket::utility::meta::real_t<Complex>;
        auto result_coeff = Complex{real_type{1}};

        auto const pauli_string_length = pauli_string.size();
        for (auto n = decltype(pauli_string_length){0}; n < pauli_string_length; ++n)
        {
          if (pauli_string[n] == 'X')
            result_index = result_index xor (StateInteger{1u} << n);
          else if (pauli_string[n] == 'Y')
          {
            result_index = result_index xor (StateInteger{1u} << n);

            if ((index bitand (StateInteger{1u} << n)) == StateInteger{0u})
              result_coeff *= ::ket::utility::minus_imaginary_unit<Complex>();
            else
              result_coeff *= ::ket::utility::imaginary_unit<Complex>();
          }
          else if (pauli_string[n] == 'Z' and (index bitand (StateInteger{1u} << n)) != StateInteger{0u})
            result_coeff *= real_type{-1};
        }

        return {result_index, result_coeff};
      }
    } // namespace utility
  } // namespace gate
} // namespace ket


#endif // KET_GATE_UTILITY_PAULI_INDEX_COEFF_HPP
