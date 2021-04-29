#ifndef _TIME_ARRIVAL_NN_EXCEPTION_H
#define _TIME_ARRIVAL_NN_EXCEPTION_H

#include <exception>
#include <string>

class TimeArrivalNNException final : public std::exception {
public:
	explicit TimeArrivalNNException(std::string msg);

	char const* what() const noexcept override;

	virtual ~TimeArrivalNNException() = default;
private:
	std::string msg_;
};

#endif //_TIME_ARRIVAL_NN_EXCEPTION_H