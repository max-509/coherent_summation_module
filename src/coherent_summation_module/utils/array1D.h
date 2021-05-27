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
    using vref_type = typename std::conditional<std::is_const<T>::value, creference_type , reference_type>::type;
public:
    Array1D() = default;

	Array1D(pointer_type p_array, size_type size)
		: _p_array(p_array), _size(size), _stride(1)
	{ }

	Array1D(pointer_type p_array, size_type size, size_type stride)
		: _p_array(p_array), _size(size), _stride(stride)
	{ }

	Array1D(const Array1D<T>&) = delete;
	Array1D& operator=(const Array1D<T>&) = delete;

	Array1D& operator=(Array1D<T> &&) noexcept = default;
	Array1D(Array1D<T> &&) noexcept = default;

	creference_type operator[](size_type x) const {
		return _p_array[x * _stride];
	}

	vref_type operator[](size_type x) {
		return _p_array[x * _stride];
	}

	creference_type operator()(size_type x) const {
		return _p_array[x * _stride];
	}

	vref_type operator()(size_type x) {
		return _p_array[x * _stride];
	}

	pointer_type get() const {
	    return _p_array;
	}

	pointer_type get(size_type x) const {
	    return _p_array + x*_stride;
	}

	size_type get_size() const { return _size; }

	size_type get_stride() const { return _stride; }

private:
	pointer_type _p_array = nullptr;
	size_type _size = 0;
	size_type _stride = 0;
};

#endif //_ARRAY1D_H