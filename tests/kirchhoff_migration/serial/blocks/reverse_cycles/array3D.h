#ifndef _ARRAY3D_H
#define _ARRAY3D_H

#include <cstddef>

template <typename T>
class Array3D {
public:
	Array3D(T* p_array, std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim) 
		: _p_array(p_array), _z_dim(z_dim), _y_dim(y_dim), _x_dim(x_dim) 
	{ }

	Array3D(const Array3D<T>&) = delete;
	Array3D& operator=(const Array3D<T>&) = delete;

	const T& operator()(std::ptrdiff_t z, std::ptrdiff_t y, std::ptrdiff_t x) const {
		return _p_array[(z*_y_dim + y)*_x_dim+x];
	}

	T& operator()(std::ptrdiff_t z, std::ptrdiff_t y, std::ptrdiff_t x) {
		return _p_array[(z*_y_dim + y)*_x_dim+x];
	}

	std::ptrdiff_t get_z_dim() const {
		return _z_dim;
	}

	std::ptrdiff_t get_y_dim() const {
		return _y_dim;
	}

	std::ptrdiff_t get_x_dim() const {
		return _x_dim;
	}

private:
	T* _p_array;
	std::ptrdiff_t _z_dim;
	std::ptrdiff_t _y_dim;
	std::ptrdiff_t _x_dim;
};

#endif //_ARRAY3D_H