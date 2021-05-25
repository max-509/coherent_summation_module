#ifndef _ARRAY2D_H
#define _ARRAY2D_H

#include <cstddef>
#include <type_traits>

template <typename T>
class Array2D {
public:
    using value_type = T;
    using pointer_type = value_type*;
    using reference_type = value_type&;
    using creference_type = const value_type&;
    using size_type = std::ptrdiff_t;
private:
    using vref_type = typename std::conditional<std::is_const<T>::value, const T&, T&>::type;
public:
    Array2D() = default;

	Array2D(pointer_type p_array, size_type y_dim, size_type x_dim)
		: _p_array(p_array), _y_dim(y_dim), _x_dim(x_dim)
	{ }

	Array2D(const Array2D<T>&) = delete;
	Array2D& operator=(const Array2D<T>&) = delete;

	Array2D& operator=(Array2D<T> &&) noexcept = default;
	Array2D(Array2D<T> &&) noexcept = default;

	creference_type operator()(size_type y, size_type x) const {
		return _p_array[y*_x_dim+x];
	}

	vref_type operator()(size_type y, size_type x) {
		return _p_array[y*_x_dim+x];
	}

	pointer_type get() const {
	    return _p_array;
	}

	pointer_type get(size_type y, size_type x) const {
	    return _p_array + y*_x_dim + x;
	}

	size_type get_y_dim() const { return _y_dim; }
	size_type get_x_dim() const { return _x_dim; }

private:
	pointer_type _p_array = nullptr;
	size_type _y_dim = 0;
	size_type _x_dim = 0;
};

#endif //_ARRAY2D_H