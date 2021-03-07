#ifndef _ARRAY2D_H
#define _ARRAY2D_H

#include <cstddef>

template <typename T>
class Array2D {
public:
	Array2D(T* p_array, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim) 
		: _p_array(p_array), _y_dim(y_dim), _x_dim(x_dim) 
	{ }

	Array2D(const Array2D<T>&) = delete;
	Array2D& operator=(const Array2D<T>&) = delete;

	const T& operator()(std::ptrdiff_t y, std::ptrdiff_t x) const {
		return _p_array[y*_x_dim+x];
	}

	T& operator()(std::ptrdiff_t y, std::ptrdiff_t x) {
		return _p_array[y*_x_dim+x];
	}

	std::ptrdiff_t get_y_dim() const { return _y_dim; }
	std::ptrdiff_t get_x_dim() const { return _x_dim; }

private:
	T* _p_array;
	std::ptrdiff_t _y_dim;
	std::ptrdiff_t _x_dim;
};

#endif //_ARRAY2D_H