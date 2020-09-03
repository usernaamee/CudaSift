%module pycusift
%{
    #define SWIG_FILE_WITH_INIT
    #define SWIG_PYTHON_CAST_MODE
    #include "pycusift.h"
%}

%include "numpy.i"
%include <std_pair.i>
%template(cpair) std::pair<int, float*>;

%init %{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* input_img, int imgWidth, int imgHeight)};

%typemap(out) std::pair<int, float*> sift_feature_extractor {
  int n_entries = (int) $1.first;
  int total_elements = n_entries * 130;
  $result = PyList_New(total_elements);
  for(int i = 0; i < total_elements; i++) {
    PyList_SetItem($result, i, PyFloat_FromDouble( (double) $1.second[i]));
  }
}

%include "pycusift.h"
