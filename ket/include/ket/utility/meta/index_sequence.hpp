#ifndef KET_UTILITY_META_INDEX_SEQUENCE
# define KET_UTILITY_META_INDEX_SEQUENCE

# include <cstddef>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <std::size_t... indices>
      struct index_sequence
      { }; // struct index_sequence<indices...>

      namespace index_sequence_detail
      {
        template <typename Result, std::size_t num_indices>
        struct generate_index_sequence;

        template <typename Result>
        struct generate_index_sequence<Result, 0u>
        { using type = Result; };

        template <std::size_t... indices, std::size_t num_residual_indices>
        struct generate_index_sequence< ::ket::utility::meta::index_sequence<indices...>, num_residual_indices >
        { using type = typename ::ket::utility::meta::index_sequence_detail::generate_index_sequence< ::ket::utility::meta::index_sequence<indices..., sizeof...(indices)>, num_residual_indices - 1u >::type; };
      }// namespace index_sequence_detail

      template <std::size_t num_indices>
      struct generate_index_sequence
      { using type = typename ::ket::utility::meta::index_sequence_detail::generate_index_sequence< ::ket::utility::meta::index_sequence<>, num_indices >::type; };
    } // namespace meta
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_META_INDEX_SEQUENCE
