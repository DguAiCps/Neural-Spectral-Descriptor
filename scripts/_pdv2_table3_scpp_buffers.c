/* MATLAB numeric array storage addresses; read-only, never alters inputs. */
#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize count, i;
    uint64_T *output;
    if (nrhs != 1 || nlhs > 1 || !mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("nsd:buffers:input", "Expected one cell of full numeric arrays.");
    count = mxGetNumberOfElements(prhs[0]);
    plhs[0] = mxCreateNumericMatrix(count, 2, mxUINT64_CLASS, mxREAL);
    output = (uint64_T *)mxGetData(plhs[0]);
    for (i = 0; i < count; ++i) {
        const mxArray *value = mxGetCell(prhs[0], i);
        if (value == NULL || !(mxIsNumeric(value) || mxIsLogical(value) || mxIsChar(value)) || mxIsSparse(value) || mxIsComplex(value))
            mexErrMsgIdAndTxt("nsd:buffers:type", "Only full real numerical, logical or char buffers are supported.");
        output[i] = (uint64_T)(uintptr_t)mxGetData(value);
        output[i + count] = (uint64_T)(mxGetNumberOfElements(value) * mxGetElementSize(value));
    }
}
