#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>

// namespace ft {
// template <class Category, class T, class Distance = ptrdiff_t,
//           class Pointer = T*, class Reference = T&>
//   struct iterator {
//     typedef T         value_type;
//     typedef Distance  difference_type;
//     typedef Pointer   pointer;
//     typedef Reference reference;
//     typedef Category  iterator_category;
//   };
// }

namespace ft {

	template <bool Cond, class T>
	struct enable_if
	{
	  // empty body
	};
	
	template <class T>
	struct enable_if<true, T>
	{
	  typedef T type;
	};

	template <class T> 
	struct is_integral
	{
		static const bool value = false;
	};

	template<>
	struct is_integral<int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<char>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<bool>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<char16_t>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<char32_t>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<wchar_t>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<signed char>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<short int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<long int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<long long int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<unsigned char>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<unsigned short int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<unsigned int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<unsigned long int>
	{
		static const bool value = true;
	};
	template <>
	struct is_integral<unsigned long long int>
	{
		static const bool value = true;
	};

}
#endif