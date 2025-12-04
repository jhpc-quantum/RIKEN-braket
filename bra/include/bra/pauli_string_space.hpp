#ifndef BRA_PAULI_STRING_SPACE_HPP
# define BRA_PAULI_STRING_SPACE_HPP

# include <cstdint>
# include <string>
# include <unordered_map>
# include <algorithm>
# include <iterator>
# include <stdexcept>

# include <bra/types.hpp>


namespace bra
{
  class wrong_pauli_string_error
    : public std::runtime_error
  {
   public:
    wrong_pauli_string_error(std::string const& pauli_string)
      : std::runtime_error{pauli_string + "is not a valid Pauli string or has wrong length"}
    { }
  }; // class wrong_pauli_string_error

  class pauli_string_space
  {
    using map_type = std::unordered_map<std::string, ::bra::complex_type>;

   public:
    using key_type = map_type::key_type;
    using mapped_type = map_type::mapped_type;
    using value_type = map_type::value_type;
    using size_type = map_type::size_type;
    using difference_type = map_type::difference_type;
    using hasher = map_type::hasher;
    using key_equal = map_type::key_equal;
    using allocator_type = map_type::allocator_type;
    using reference = map_type::reference;
    using const_reference = map_type::const_reference;
    using pointer = map_type::pointer;
    using const_pointer = map_type::const_pointer;
    using iterator = map_type::iterator;
    using const_iterator = map_type::const_iterator;

   private:
    size_type num_qubits_;
    map_type basis_scalar_map_;

   public:
    pauli_string_space()
      : pauli_string_space{size_type{0u}}
    { }

    explicit pauli_string_space(std::size_t const num_qubits)
      : num_qubits_{num_qubits}, basis_scalar_map_{{std::string(num_qubits, 'I'), ::bra::complex_type{::bra::real_type{0}}}}
    { }

    pauli_string_space(std::string const& pauli_string, ::bra::complex_type const& scalar)
      : num_qubits_{pauli_string.size()}, basis_scalar_map_{{pauli_string, scalar}}
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};
    }

    template <typename InputIterator>
    pauli_string_space(InputIterator const first, InputIterator const last)
      : num_qubits_{first->first.size()}, basis_scalar_map_(first, last)
    {
      for (auto const& basis_scalar: basis_scalar_map_)
      {
        auto const& pauli_string = basis_scalar.first;

        if (not is_valid_pauli_string(pauli_string))
          throw wrong_pauli_string_error{pauli_string};
      }
    }

    pauli_string_space(std::initializer_list<value_type> init)
      : pauli_string_space(init.begin(), init.end())
    { }

    auto begin() noexcept -> iterator { using std::begin; return begin(basis_scalar_map_); }
    auto begin() const noexcept -> const_iterator { using std::begin; return begin(basis_scalar_map_); }
    auto cbegin() const noexcept -> const_iterator { return basis_scalar_map_.cbegin(); }
    auto end() noexcept -> iterator { using std::end; return end(basis_scalar_map_); }
    auto end() const noexcept -> const_iterator { using std::end; return end(basis_scalar_map_); }
    auto cend() const noexcept -> const_iterator { return basis_scalar_map_.cend(); }

    auto empty() const noexcept -> bool { return basis_scalar_map_.empty(); }
    auto size() const noexcept -> std::size_t { return basis_scalar_map_.size(); }
    auto max_size() const noexcept -> std::size_t { return basis_scalar_map_.max_size(); }

    auto at(std::string const& pauli_string) -> ::bra::complex_type&
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_.at(pauli_string);
    }

    auto at(std::string const& pauli_string) const -> ::bra::complex_type const&
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_.at(pauli_string);
    }

    auto operator[](std::string const& pauli_string) -> ::bra::complex_type&
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_[pauli_string];
    }

    auto operator[](std::string&& pauli_string) -> ::bra::complex_type&
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_[std::move(pauli_string)];
    }
 
    auto find(std::string const& pauli_string) -> iterator
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_.find(pauli_string);
    }

    auto find(std::string const& pauli_string) const -> const_iterator
    {
      if (not is_valid_pauli_string(pauli_string))
        throw wrong_pauli_string_error{pauli_string};

      return basis_scalar_map_.find(pauli_string);
    }

    auto contains(std::string const& pauli_string) const -> bool
    { using std::end; return basis_scalar_map_.find(pauli_string) != end(basis_scalar_map_); }

    auto num_qubits() const -> size_type { return num_qubits_; }

   private:
    auto is_valid_pauli_string(std::string const& pauli_string) const noexcept -> bool
    { return pauli_string.size() == num_qubits_ and pauli_string.find_first_not_of("XYZI") == std::string::npos; }
  }; // class pauli_string_space
} // namespace bra


#endif // BRA_PAULI_STRING_SPACE_HPP
