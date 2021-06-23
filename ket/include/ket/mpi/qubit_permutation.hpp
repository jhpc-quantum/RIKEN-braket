#ifndef KET_MPI_QUBIT_PERMUTATION_HPP
# define KET_MPI_QUBIT_PERMUTATION_HPP

# include <cassert>
# include <vector>
# include <algorithm>
# include <iterator>
# include <memory>
# include <utility>
# include <initializer_list>
# include <type_traits>
# if __cplusplus < 201703L // std::is_nothrow_swappable is introduced since C++17
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/mpi/permutated.hpp>

# if __cplusplus >= 201703L
#   define KET_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define KET_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace ket
{
  namespace mpi
  {
    // main interfaces:
    //   permutation[unpermutated_qubit]: get permutated qubit corresponding to unpermutated qubit unpermutated_qubit
    //   inverse(permutation): get inverse view of permutation
    //   inverse(permutation)[permutated_qubit]: get unpermutated qubit corresponding to permutated qubit permutated_qubit
    //   permutate(permutation, unpermutated_qubit1, unpermutated_qubit2): permutate qubits unpermutated_qubit1 and unpermutated_qubit2
    //   permutate_bits(permutation, unsigned_integer): convert unpermutated value unsigned_integer to permutated value
    //   inverse_permutate_bits(permutation, unsigned_integer): convert permutated value unsigned_integer to unpermutated value
    template <
      typename StateInteger = std::uint64_t,
      typename BitInteger = unsigned int,
      typename Allocator
        = std::allocator< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > > >
    class qubit_permutation
    {
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      using control_qubit_type = ::ket::control<qubit_type>;
      using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
      using permutated_control_qubit_type = ::ket::mpi::permutated<control_qubit_type>;

      using data_type
        = std::vector<permutated_qubit_type, typename Allocator::template rebind<permutated_qubit_type>::other>;
      using inverse_data_type
        = std::vector<qubit_type, typename Allocator::template rebind<qubit_type>::other>;

      data_type data_;
      inverse_data_type inverse_data_;

     public:
      using value_type = typename data_type::value_type;
      using reference = typename data_type::const_reference;
      using const_reference = typename data_type::const_reference;
      using pointer = typename data_type::const_pointer;
      using const_pointer = typename data_type::const_pointer;
      using interator = typename data_type::const_iterator;
      using const_iterator = typename data_type::const_iterator;
      using reverse_iterator = typename data_type::const_reverse_iterator;
      using const_reverse_iterator = typename data_type::const_reverse_iterator;
      using size_type = typename data_type::size_type;
      using difference_type = typename data_type::difference_type;
      using allocator_type = typename data_type::allocator_type;

      qubit_permutation() noexcept
        : data_{Allocator()}, inverse_data_{Allocator()}
      { }

      explicit qubit_permutation(Allocator const& allocator) noexcept
        : data_{allocator}, inverse_data_{allocator}
      { }

      explicit qubit_permutation(size_type const num_qubits)
        : data_{generate_identity_permutation(num_qubits)},
          inverse_data_{data_to_inverse_data(data_)}
      { }

      template <typename InputIterator>
      qubit_permutation(
        InputIterator const first, InputIterator const last,
        Allocator const& allocator = Allocator())
        : data_{first, last, allocator},
          inverse_data_{data_to_inverse_data(data_, allocator)}
      { assert(is_valid_permutation(data_)); }

      qubit_permutation(
        qubit_permutation const& other, Allocator const& allocator)
        : data_{other.data_, allocator},
          inverse_data_{other.inverse_data_, allocator}
      { }

      qubit_permutation(qubit_permutation&& other, Allocator const& allocator)
        : data_{std::move(other.data_), allocator},
          inverse_data_{std::move(other.inverse_data_), allocator}
      { }

      qubit_permutation(
        std::initializer_list<value_type> initializer_list,
        Allocator const& allocator = Allocator{})
        : data_{initializer_list, allocator},
          inverse_data_{data_to_inverse_data_(data_, allocator)}
      { assert(is_valid_permutation(data_)); }

      qubit_permutation& operator=(std::initializer_list<value_type> initializer_list)
      {
        data_ = initializer_list;
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
        return *this;
      }

      ~qubit_permutation() = default;
      qubit_permutation(qubit_permutation const&) = default;
      qubit_permutation& operator=(qubit_permutation const&) = default;
      qubit_permutation(qubit_permutation&&) = default;
      qubit_permutation& operator=(qubit_permutation&&) = default;

      template <typename InputIterator>
      void assign(InputIterator const first, InputIterator const last)
      {
        data_.assign(first, last);
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
      }

      void assign(std::initializer_list<value_type> initializer_list)
      {
        data_ = initializer_list;
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
      }

      allocator_type get_allocator() const noexcept
      { return data_.get_allocator(); }

      const_iterator begin() const noexcept { return data_.begin(); }
      const_iterator end() const noexcept { return data_.end(); }
      const_reverse_iterator rbegin() const noexcept  { return data_.rbegin(); }
      const_reverse_iterator rend() const noexcept { return data_.rend(); }

      const_iterator cbegin() const noexcept { return data_.cbegin(); }
      const_iterator cend() const noexcept { return data_.cend(); }
      const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
      const_reverse_iterator crend() const noexcept { return data_.crend(); }

      size_type size() const noexcept { return data_.size(); }
      size_type max_size() const noexcept { return data_.max_size(); }
      size_type capacity() const noexcept { return data_.capacity(); }
      bool empty() const noexcept { return data_.empty(); }
      void reserve(size_type const size)
      { data_.reserve(size); inverse_data_.reserve(size); }
      void shrink_to_fit()
      { data_.shrink_to_fit(); inverse_data_.shrink_to_fit(); }

      const_reference operator[](qubit_type const from) const
      { return data_[static_cast<BitInteger>(from)]; }
      permutated_control_qubit_type operator[](control_qubit_type const from) const
      { return ::ket::mpi::make_permutated(::ket::make_control(data_[static_cast<BitInteger>(from.qubit())].qubit())); }

      const_reference at(qubit_type const from) const
      { return data_.at(static_cast<BitInteger>(from)); }
      permutated_control_qubit_type at(control_qubit_type const from) const
      { return ::ket::mpi::make_permutated(::ket::make_control(data_.at(static_cast<BitInteger>(from.qubit())).qubit())); }

      const_reference front() const { return data_.front(); }
      const_reference back() const { return data_.back(); }

      value_type* data() noexcept { return data_.data(); }
      value_type const* data() const noexcept { return data_.data(); }

      class inverse_view
      {
        inverse_data_type& inverse_data_;

       public:
        inverse_view() = delete;
        inverse_view& operator=(inverse_view const&) = delete;
        inverse_view& operator=(inverse_view&&) = delete;

        inverse_view(inverse_view const&) = default;
        inverse_view(inverse_view&&) = default;

        inverse_view(inverse_data_type& inverse_data)
          : inverse_data_{inverse_data}
        { }

        typename inverse_data_type::const_iterator begin() const noexcept { return inverse_data_.begin(); }
        typename inverse_data_type::const_iterator end() const noexcept { return inverse_data_.end(); }
        typename inverse_data_type::const_reverse_iterator rbegin() const noexcept { return inverse_data_.rbegin(); }
        typename inverse_data_type::const_reverse_iterator rend() const noexcept { return inverse_data_.rend(); }

        typename inverse_data_type::const_iterator cbegin() const noexcept { return inverse_data_.cbegin(); }
        typename inverse_data_type::const_iterator cend() const noexcept { return inverse_data_.cend(); }
        typename inverse_data_type::const_reverse_iterator crbegin() const noexcept { return inverse_data_.crbegin(); }
        typename inverse_data_type::const_reverse_iterator crend() const noexcept { return inverse_data_.crend(); }

        typename inverse_data_type::const_reference operator[](permutated_qubit_type const to) const
        { return inverse_data_[static_cast<BitInteger>(to.qubit())]; }
        control_qubit_type operator[](permutated_control_qubit_type const to) const
        { return ::ket::make_control(inverse_data_[static_cast<BitInteger>(to.qubit().qubit())]); }

        typename inverse_data_type::const_reference at(permutated_qubit_type const to) const
        { return inverse_data_.at(static_cast<BitInteger>(to.qubit())); }
        control_qubit_type at(permutated_control_qubit_type const to) const
        { return ::ket::make_control(inverse_data_.at(static_cast<BitInteger>(to.qubit().qubit()))); }

        typename inverse_data_type::const_reference front() const { return inverse_data_.front(); }
        typename inverse_data_type::const_reference back() const { return inverse_data_.back(); }

        typename inverse_data_type::value_type* data() noexcept { return inverse_data_.data(); }
        typename inverse_data_type::value_type const* data() const noexcept { return inverse_data_.data(); }
      }; // class inverse_view

      inverse_view inverse() { return inverse_view(inverse_data_); }
      inverse_view inverse() const
      { return inverse_view(const_cast<inverse_data_type&>(inverse_data_)); }

      void swap(qubit_permutation& other)
        noexcept(
          KET_is_nothrow_swappable<data_type>::value
          and KET_is_nothrow_swappable<inverse_data_type>::value)
      { data_.swap(other.data_); inverse_data_.swap(other.inverse_data_); }

      void push_back()
      {
        data_.emplace_back(data_.size());
        inverse_data_.emplace_back(inverse_data_.size());
      }

      void clear() noexcept { data_.clear(); inverse_data_.clear(); }

      void permutate(qubit_type const qubit1, qubit_type const qubit2)
      {
        assert(qubit1 != qubit2);
        using std::swap;
        swap(
          inverse_data_[static_cast<BitInteger>(data_[static_cast<BitInteger>(qubit1)].qubit())],
          inverse_data_[static_cast<BitInteger>(data_[static_cast<BitInteger>(qubit2)].qubit())]);
        swap(
          data_[static_cast<BitInteger>(qubit1)],
          data_[static_cast<BitInteger>(qubit2)]);
      }

      void permutate(control_qubit_type const control_qubit, qubit_type const qubit)
      { permutate(control_qubit.qubit(), qubit); }
      void permutate(qubit_type const qubit, control_qubit_type const control_qubit)
      { permutate(qubit, control_qubit.qubit()); }
      void permutate(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
      { permutate(control_qubit1.qubit(), control_qubit2.qubit()); }

     private:
      data_type generate_identity_permutation(size_type const size)
      {
        auto result = data_type(size);
        std::iota(std::begin(result), std::end(result), value_type{0u});
        return result;
      }

      inverse_data_type data_to_inverse_data(
        data_type const& data, Allocator const& allocator = Allocator())
      {
        auto const data_size = data.size();
        auto result = inverse_data_type(data_size, qubit_type{}, allocator);

        for (auto index = size_type{0u}; index < data_size; ++index)
          result[static_cast<typename ::ket::meta::bit_integer_of<value_type>::type>(data[index].qubit())]
            = qubit_type{index};

        return result;
      }

      bool is_valid_permutation(data_type permutation) const
      {
        std::sort(std::begin(permutation), std::end(permutation));

        auto previous_qubit = permutation.front();
        auto const last = std::end(permutation);

        for (auto iter = std::next(std::begin(permutation)); iter != last; ++iter)
          if (*iter == previous_qubit)
            return false;
          else
            previous_qubit = *iter;

        return true;
      }
    }; // class qubit_permutation<StateInteger, BitInteger, Allocator>

    template <typename StateInteger, typename BitInteger, typename Allocator>
    inline void swap(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& lhs,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& rhs)
      noexcept(
        KET_is_nothrow_swappable<
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>>::value)
    { lhs.swap(rhs); }

    template <typename StateInteger, typename BitInteger, typename Allocator>
    inline typename ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>::inverse_view
    inverse(::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation)
    { return permutation.inverse(); }

    template <typename StateInteger, typename BitInteger, typename Allocator, typename AnyQubit1, typename AnyQubit2>
    inline void permutate(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      AnyQubit1 const any_qubit1, AnyQubit2 const any_qubit2)
    { permutation.permutate(any_qubit1, any_qubit2); }

    namespace permutate_bits_detail
    {
      enum class boolean { true_ = true, false_ = false };

      inline constexpr bool to_bool(::ket::mpi::permutate_bits_detail::boolean const b)
      { return static_cast<bool>(b); }

      struct permutate_bits_impl
      {
        template <
          typename StateInteger, typename BitInteger,
          typename Allocator, typename UnsignedInteger>
        static UnsignedInteger call(
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
              permutation,
          UnsignedInteger unsigned_integer)
        {
          static_assert(
            std::is_unsigned<UnsignedInteger>::value, "UnsignedInteger should be unsigned");

          using is_permutated_type = std::vector< ::ket::mpi::permutate_bits_detail::boolean >;
          static auto is_permutated = is_permutated_type{};
          is_permutated.assign(permutation.size(), ::ket::mpi::permutate_bits_detail::boolean::false_);

          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          constexpr auto first_bit = qubit_type{0u};
          auto const last_bit = qubit_type{permutation.size()};

          for (auto bit = first_bit; bit < last_bit; ++bit)
          {
            if (::ket::mpi::permutate_bits_detail::to_bool(
                  is_permutated[static_cast<BitInteger>(bit)]))
              continue;

            auto present_bit = bit;
            // 00000b00000
            auto present_bit_value
              = unsigned_integer bitand (UnsignedInteger{1u} << bit);

            do
            {
              auto const previous_bit = present_bit;
              // 00000b00000
              auto const previous_bit_value = present_bit_value;
              present_bit = permutation[previous_bit].qubit();
              // 00a00000000
              present_bit_value
                = unsigned_integer bitand (UnsignedInteger{1u} << present_bit);

              // xxbxxbxxxxx
              unsigned_integer
                  // xx0xxbxxxxx
                = (unsigned_integer bitand compl (UnsignedInteger{1u} << present_bit))
                  bitor
                  // 00b00000000
                  ((previous_bit_value >> previous_bit) << present_bit);

              is_permutated[static_cast<BitInteger>(present_bit)]
                = ::ket::mpi::permutate_bits_detail::boolean::true_;
            }
            while (present_bit != bit);
          }

          return unsigned_integer;
        }
      }; // struct permutate_bits_impl

      struct inverse_permutate_bits_impl
      {
        template <
          typename StateInteger, typename BitInteger,
          typename Allocator, typename UnsignedInteger>
        static UnsignedInteger call(
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
              permutation,
          UnsignedInteger unsigned_integer)
        {
          static_assert(
            std::is_unsigned<UnsignedInteger>::value, "UnsignedInteger should be unsigned");

          using is_permutated_type = std::vector< ::ket::mpi::permutate_bits_detail::boolean >;
          static auto is_permutated = is_permutated_type{};
          is_permutated.assign(permutation.size(), ::ket::mpi::permutate_bits_detail::boolean::false_);

          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          constexpr auto first_bit = qubit_type{0u};
          auto const last_bit = qubit_type{permutation.size()};

          for (auto bit = first_bit; bit < last_bit; ++bit)
          {
            if (::ket::mpi::permutate_bits_detail::to_bool(
                  is_permutated[static_cast<BitInteger>(bit)]))
              continue;

            auto present_bit = bit;
            // 00000b00000
            auto present_bit_value
              = unsigned_integer bitand (UnsignedInteger{1u} << bit);

            do
            {
              auto const previous_bit = present_bit;
              // 00000b00000
              auto const previous_bit_value = present_bit_value;
              present_bit = ::ket::mpi::inverse(permutation)[::ket::mpi::make_permutated(previous_bit)];
              // 00a00000000
              present_bit_value
                = unsigned_integer bitand (UnsignedInteger{1u} << present_bit);

              // xxbxxbxxxxx
              unsigned_integer
                  // xx0xxbxxxxx
                = (unsigned_integer bitand compl (UnsignedInteger{1u} << present_bit))
                  bitor
                  // 00b00000000
                  ((previous_bit_value >> previous_bit) << present_bit);

              is_permutated[static_cast<BitInteger>(present_bit)]
                = ::ket::mpi::permutate_bits_detail::boolean::true_;
            }
            while (present_bit != bit);
          }

          return unsigned_integer;
        }
      }; // struct inverse_permutate_bits_impl
    } // namespace permutate_bits_detail

    template <
      typename StateInteger, typename BitInteger,
      typename Allocator, typename UnsignedInteger>
    inline UnsignedInteger permutate_bits(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
        permutation,
      UnsignedInteger const unsigned_integer)
    {
      return ::ket::mpi::permutate_bits_detail::permutate_bits_impl::call(
        permutation, unsigned_integer);
    }

    template <
      typename StateInteger, typename BitInteger,
      typename Allocator, typename UnsignedInteger>
    inline UnsignedInteger inverse_permutate_bits(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
        permutation,
      UnsignedInteger const unsigned_integer)
    {
      return ::ket::mpi::permutate_bits_detail::inverse_permutate_bits_impl::call(
        permutation, unsigned_integer);
    }
  } // namespace mpi
} // namespace ket


#undef KET_is_nothrow_swappable

#endif // KET_MPI_QUBIT_PERMUTATION_HPP
