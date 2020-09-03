#include <cudaImage.h>
#include <cudaSift.h>
#include <utility>

std::pair<int, float*> sift_feature_extractor(float *input_img, int imgWidth, int imgHeight, int numOctaves=6, float initBlur=1.6f, float thresh=3.5f, float minScale=0.001f, bool upScale=false);
