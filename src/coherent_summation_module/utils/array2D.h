#ifndef _ARRAY2D_H
#define _ARRAY2D_H

#include <cstddef>
#include <type_traits>

template<typename T>
class Array2D {
public:
    using value_type = T;
    using pointer_type = value_type *;
    using reference_type = value_type &;
    using creference_type = const value_type &;
    using size_type = std::ptrdiff_t;
private:
    using vref_type = typename std::conditional<std::is_const<T>::value, creference_type, reference_type>::type;
public:
    Array2D() = default;

    Array2D(pointer_type p_array, size_type y_dim, size_type x_dim)
            : _p_array(p_array), _y_dim(y_dim), _x_dim(x_dim), _y_stride(_x_dim), _x_stride(1) {}

    Array2D(pointer_type p_array,
            size_type y_dim,
            size_type x_dim,
            size_type y_stride,
            size_type x_stride)
            : _p_array(p_array), _y_dim(y_dim), _x_dim(x_dim), _y_stride(y_stride), _x_stride(x_stride) {}

    Array2D(const Array2D<T> &) = delete;

    Array2D &operator=(const Array2D<T> &) = delete;

    Array2D &operator=(Array2D<T> &&) noexcept = default;

    Array2D(Array2D<T> &&) noexcept = default;

    creference_type operator()(size_type y, size_type x) const {
        return _p_array[y * _y_stride + x * _x_stride];
    }

    vref_type operator()(size_type y, size_type x) {
        return _p_array[y * _y_stride + x * _x_stride];
    }

    pointer_type get() const {
        return _p_array;
    }

    pointer_type get(size_type y, size_type x) const {
        return _p_array + y * _y_stride + x * _x_stride;
    }

    size_type get_y_dim() const { return _y_dim; }

    size_type get_x_dim() const { return _x_dim; }

    size_type get_y_stride() const { return _y_stride; }

    size_type get_x_stride() const { return _x_stride; }

private:
    pointer_type _p_array = nullptr;
    size_type _y_dim = 0;
    size_type _x_dim = 0;
    size_type _y_stride = 0;
    size_type _x_stride = 0;
};

#endif //_ARRAY2D_H