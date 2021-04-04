#include "TimeArrivalNNBase.h"
#include "TimeArrivalNNException.h"

#include <string>

TimeArrivalNNBase::TimeArrivalNNBase() :
        graph_(TF_NewGraph(), TF_DeleteGraph),
        status_(TF_NewStatus(), TF_DeleteStatus) {
    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(
                std::string("ERROR: Unable to create graph %s") + std::string(TF_Message(status_.get())));
    }
}