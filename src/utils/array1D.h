#ifndef _ARRAY1D_H
#define _ARRAY1D_H

#include <cstddef>
#include <type_traits>

template <typename T>
class Array1D {
public:
    using value_type = T;
    using pointer_type = value_type*;
    using reference_type = value_type&;
    using creference_type = const value_type&;
    using size_type = std::ptrdiff_t;
private:
    using vref_type = typename std::conditional<std::is_const<T>::value, const T&, T&>::type;
public:
    Array1D() = default;

	Array1D(pointer_type p_array, size_type size)
		: _p_array(p_array), _size(size)
	{ }

	Array1D(const Array1D<T>&) = delete;
	Array1D& operator=(const Array1D<T>&) = delete;

	Array1D& operator=(Array1D<T> &&) noexcept = default;
	Array1D(Array1D<T> &&) noexcept = default;

	creference_type operator[](size_type x) const {
		return _p_array[x];
	}

	vref_type operator[](size_type x) {
		return _p_array[x];
	}

	creference_type operator()(size_type x) const {
		return _p_array[x];
	}

	vref_type operator()(size_type x) {
		return _p_array[x];
	}

	pointer_type get() const {
	    return _p_array;
	}

	pointer_type get(size_type x) const {
	    return _p_array + x;
	}

	size_type get_size() const { return _size; }

private:
	pointer_type _p_array = nullptr;
	size_type _size = 0;
};

#endif //_ARRAY1D_H