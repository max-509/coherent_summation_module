#include "TimeArrivalNNException.h"

TimeArrivalNNException::TimeArrivalNNException(std::string msg) : msg_(std::move(msg))
{ }

char const* TimeArrivalNNException::what() const noexcept {
	return msg_.c_str();
}
