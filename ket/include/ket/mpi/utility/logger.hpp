#ifndef KET_MPI_UTILITY_LOGGER_HPP
# define KET_MPI_UTILITY_LOGGER_HPP

# ifdef KET_PRINT_LOG
#   include <iostream>
#   include <sstream>
#   include <utility>
# endif // KET_PRINT_LOG
# include <string>
# include <memory>

# ifdef KET_PRINT_LOG
#   include <boost/optional.hpp>
# endif // KET_PRINT_LOG

# include <yampi/environment.hpp>
# ifdef KET_PRINT_LOG
#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/wall_clock.hpp>
#   include <yampi/lowest_io_process.hpp>
# endif // KET_PRINT_LOG


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
# ifdef KET_PRINT_LOG
      class logger
      {
        boost::optional<yampi::rank> maybe_io_rank_;
        yampi::wall_clock::time_point initial_time_;

       public:
        explicit logger(yampi::environment const& environment)
          : maybe_io_rank_{yampi::lowest_io_process(environment)},
            initial_time_{yampi::wall_clock::now(environment)}
        { }

        logger() = delete;
        logger(logger const&) = delete;
        logger& operator=(logger const&) = delete;
        logger(logger&&) = delete;
        logger& operator=(logger&&) = delete;

        auto print(char const* c_str, yampi::environment const& environment) const -> void
        { print(static_cast<std::string>(c_str), environment); }

        auto print(wchar_t const* c_str, yampi::environment const& environment) const -> void
        { print(static_cast<std::wstring>(c_str), environment); }

        auto print(std::string const& string, yampi::environment const& environment) const -> void
        { do_print(std::clog, string, environment); }

        auto print(std::wstring const& string, yampi::environment const& environment) const -> void
        { do_print(std::wclog, string, environment); }

        auto print_with_time(char const* c_str, yampi::environment const& environment) const -> void
        { print_with_time(static_cast<std::string>(c_str), environment); }

        auto print_with_time(wchar_t const* c_str, yampi::environment const& environment) const -> void
        { print_with_time(static_cast<std::wstring>(c_str), environment); }

        auto print_with_time(std::string const& string, yampi::environment const& environment) const -> void
        { do_print_with_time(std::clog, string, environment); }

        auto print_with_time(std::wstring const& string, yampi::environment const& environment) const -> void
        { do_print_with_time(std::wclog, string, environment); }

       private:
        template <typename Character, typename CharacterTraits, typename Allocator>
        auto do_print(
          std::basic_ostream<Character, CharacterTraits>& output_stream,
          std::basic_string<Character, CharacterTraits, Allocator> const& string,
          yampi::environment const& environment) const -> void
        {
          if (!maybe_io_rank_)
            return;

          if (yampi::communicator{yampi::tags::world_communicator}.rank(environment) != *maybe_io_rank_)
            return;

          output_stream << string << std::endl;
        }

        template <typename Character, typename CharacterTraits, typename Allocator>
        auto do_print_with_time(
          std::basic_ostream<Character, CharacterTraits>& output_stream,
          std::basic_string<Character, CharacterTraits, Allocator> const& string,
          yampi::environment const& environment) const -> void
        {
          if (!maybe_io_rank_)
            return;

          if (yampi::communicator{yampi::tags::world_communicator}.rank(environment) != *maybe_io_rank_)
            return;

          output_stream
            << string << ' '
            << (yampi::wall_clock::now(environment) - initial_time_).count()
            << std::endl;
        }
      }; // class logger

      template <
        typename Character,
        typename CharacterTraits = std::char_traits<Character>,
        typename Allocator = std::allocator<Character> >
      class log_guard;

      template <typename CharacterTraits, typename Allocator>
      class log_guard<char, CharacterTraits, Allocator>
      {
        using string_type = std::basic_string<char, CharacterTraits, Allocator>;

        ::ket::mpi::utility::logger logger_;
        string_type string_;
        yampi::environment const* environment_ptr_;

       public:
        log_guard(char const* c_str, yampi::environment const& environment)
          : logger_{environment}, string_{c_str}, environment_ptr_{&environment}
        { logger_.print("[start] " + string_, *environment_ptr_); }

        log_guard(string_type const& string, yampi::environment const& environment)
          : logger_{environment}, string_{string}, environment_ptr_{&environment}
        { logger_.print("[start] " + string_, *environment_ptr_); }

        ~log_guard()
        { logger_.print("[end] " + string_, *environment_ptr_); }

        log_guard() = delete;
        log_guard(log_guard const&) = delete;
        log_guard& operator=(log_guard const&) = delete;
        log_guard(log_guard&&) = delete;
        log_guard& operator=(log_guard&&) = delete;
      }; // class log_guard<char, CharacterTraits, Allocator>

      template <typename CharacterTraits, typename Allocator>
      class log_guard<wchar_t, CharacterTraits, Allocator>
      {
        using string_type = std::basic_string<wchar_t, CharacterTraits, Allocator>;

        ::ket::mpi::utility::logger logger_;
        string_type string_;
        yampi::environment const* environment_ptr_;

       public:
        log_guard(wchar_t const* c_str, yampi::environment const& environment)
          : logger_{environment}, string_{c_str}, environment_ptr_{&environment}
        { logger_.print(L"[start] " + string_, *environment_ptr_); }

        log_guard(string_type const& string, yampi::environment const& environment)
          : logger_{environment}, string_{string}, environment_ptr_{&environment}
        { logger_.print(L"[start] " + string_, *environment_ptr_); }

        ~log_guard()
        { logger_.print(L"[end] " + string_, *environment_ptr_); }

        log_guard() = delete;
        log_guard(log_guard const&) = delete;
        log_guard& operator=(log_guard const&) = delete;
        log_guard(log_guard&&) = delete;
        log_guard& operator=(log_guard&&) = delete;
      }; // class log_guard<wchar_t, CharacterTraits, Allocator>

      template <
        typename Character,
        typename CharacterTraits = std::char_traits<Character>,
        typename Allocator = std::allocator<Character> >
      class log_with_time_guard;

      template <typename CharacterTraits, typename Allocator>
      class log_with_time_guard<char, CharacterTraits, Allocator>
      {
        using string_type = std::basic_string<char, CharacterTraits, Allocator>;

        ::ket::mpi::utility::logger logger_;
        string_type string_;
        yampi::environment const* environment_ptr_;

       public:
        log_with_time_guard(char const* c_str, yampi::environment const& environment)
          : logger_{environment}, string_{c_str}, environment_ptr_{&environment}
        { logger_.print("[start] " + string_, *environment_ptr_); }

        log_with_time_guard(string_type const& string, yampi::environment const& environment)
          : logger_{environment}, string_{string}, environment_ptr_{&environment}
        { logger_.print("[start] " + string_, *environment_ptr_); }

        ~log_with_time_guard()
        { logger_.print_with_time("[end] " + string_, *environment_ptr_); }

        log_with_time_guard() = delete;
        log_with_time_guard(log_with_time_guard const&) = delete;
        log_with_time_guard& operator=(log_with_time_guard const&) = delete;
        log_with_time_guard(log_with_time_guard&&) = delete;
        log_with_time_guard& operator=(log_with_time_guard&&) = delete;
      }; // class log_with_time_guard<char, CharacterTraits, Allocator>

      template <typename CharacterTraits, typename Allocator>
      class log_with_time_guard<wchar_t, CharacterTraits, Allocator>
      {
        using string_type = std::basic_string<wchar_t, CharacterTraits, Allocator>;

        ::ket::mpi::utility::logger logger_;
        string_type string_;
        yampi::environment const* environment_ptr_;

       public:
        log_with_time_guard(wchar_t const* c_str, yampi::environment const& environment)
          : logger_{environment}, string_{c_str}, environment_ptr_{&environment}
        { logger_.print(L"[start] " + string_, *environment_ptr_); }

        log_with_time_guard(string_type const& string, yampi::environment const& environment)
          : logger_{environment}, string_{string}, environment_ptr_{&environment}
        { logger_.print(L"[start] " + string_, *environment_ptr_); }

        ~log_with_time_guard()
        { logger_.print_with_time(L"[end] " + string_, *environment_ptr_); }

        log_with_time_guard() = delete;
        log_with_time_guard(log_with_time_guard const&) = delete;
        log_with_time_guard& operator=(log_with_time_guard const&) = delete;
        log_with_time_guard(log_with_time_guard&&) = delete;
        log_with_time_guard& operator=(log_with_time_guard&&) = delete;
      }; // class log_with_time_guard<wchar_t, CharacterTraits, Allocator>

      namespace logger_detail
      {
        template <typename Character, typename CharacterTraits, typename Allocator>
        inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>&) -> void
        { }

        template <typename Character, typename CharacterTraits, typename Allocator, typename Value, typename... Values>
        inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>& output_string_stream, Value&& value, Values&&... values) -> void
        {
          output_string_stream << std::forward<Value>(value);
          ::ket::mpi::utility::logger_detail::insert(output_string_stream, std::forward<Values>(values)...);
        }
      } // namespace logger_detail

      template <typename Character, typename CharacterTraits, typename Allocator, typename... Values>
      inline auto generate_logger_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_string, Values&&... values)
      -> std::basic_string<Character, CharacterTraits, Allocator>
      {
        auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_string, std::ios_base::ate};
        ::ket::mpi::utility::logger_detail::insert(output_string_stream, std::forward<Values>(values)...);
        return output_string_stream.str();
      }
# else // KET_PRINT_LOG
      class logger
      {
       public:
        explicit logger(yampi::environment const&) { }

        logger() = delete;
        logger(logger const&) = delete;
        logger& operator=(logger const&) = delete;
        logger(logger&&) = delete;
        logger& operator=(logger&&) = delete;

        template <typename Character>
        auto print(Character const*, yampi::environment const&) const -> void
        { }

        template <typename Character, typename CharacterTraits, typename Allocator>
        auto print(
          std::basic_string<Character, CharacterTraits, Allocator> const&,
          yampi::environment const&) const -> void
        { }

        template <typename Character>
        auto print_with_time(Character const*, yampi::environment const&) const -> void
        { }

        template <typename Character, typename CharacterTraits, typename Allocator>
        auto print_with_time(
          std::basic_string<Character, CharacterTraits, Allocator> const&,
          yampi::environment const&) const -> void
        { }
      }; // class logger

      template <
        typename Character,
        typename CharacterTraits = std::char_traits<Character>,
        typename Allocator = std::allocator<Character> >
      class log_guard
      {
       public:
        log_guard(Character const*, yampi::environment const&) { }

        log_guard(
          std::basic_string<Character, CharacterTraits, Allocator> const&,
          yampi::environment const&)
        { }

        ~log_guard() = default;

        log_guard() = delete;
        log_guard(log_guard const&) = delete;
        log_guard& operator=(log_guard const&) = delete;
        log_guard(log_guard&&) = delete;
        log_guard& operator=(log_guard&&) = delete;
      }; // class log_guard<Character, CharacterTraits, Allocator>

      template <
        typename Character,
        typename CharacterTraits = std::char_traits<Character>,
        typename Allocator = std::allocator<Character> >
      class log_with_time_guard
      {
       public:
        log_with_time_guard(Character const*, yampi::environment const&) { }

        log_with_time_guard(
          std::basic_string<Character, CharacterTraits, Allocator> const&,
          yampi::environment const&)
        { }

        ~log_with_time_guard() = default;

        log_with_time_guard() = delete;
        log_with_time_guard(log_with_time_guard const&) = delete;
        log_with_time_guard& operator=(log_with_time_guard const&) = delete;
        log_with_time_guard(log_with_time_guard&&) = delete;
        log_with_time_guard& operator=(log_with_time_guard&&) = delete;
      }; // class log_with_time_guard<Character, CharacterTraits, Allocator>

      template <typename Character, typename CharacterTraits, typename Allocator, typename... Values>
      inline auto generate_logger_string(std::basic_string<Character, CharacterTraits, Allocator> const&, Values&&...)
      -> std::basic_string<Character, CharacterTraits, Allocator>
      { return {}; }
# endif // KET_PRINT_LOG
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_LOGGER_HPP
