/*
    Takes input a single channel numpy ndarray image (float32 datatype) as input
    Outputs number of detected keypoints (N) and float array with Nx130 entries
    Float array is actually a flattened 2D matrix, where first two columns are (x, y) of keypoint
    Rest of the 128 columns are feature descriptors.
*/


#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "pycusift.h"

using namespace std;

pair<int, float*> sift_feature_extractor(float *input_img, 
                                          int imgWidth,
                                          int imgHeight,
                                          int numOctaves, 
                                          float initBlur, 
                                          float thresh, 
                                          float minScale, 
                                          bool upScale)
{
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    CudaImage img;
    img.Allocate(imgHeight, imgWidth, imgHeight, false, NULL, input_img);
    img.Download();
    ExtractSift(siftData, img, numOctaves, initBlur, thresh, minScale, upScale);
    int extracted_npts = siftData.numPts;
    float * result = new float[(int)siftData.numPts * 130];
    for(int i=0; i<siftData.numPts; i++){
        result[i * 130] = siftData.h_data[i].xpos;
        result[i * 130 + 1] = siftData.h_data[i].ypos;
        for(int j=0; j<128; j++){
            result[i * 130 + 2 + j] = siftData.h_data[i].data[j];
        }
    }
    FreeSiftData(siftData);
    return make_pair(extracted_npts, result);
}

int main(){

    cv::Mat limg;
    cv::imread("data/img1.png", 0).convertTo(limg, CV_32FC1);
    sift_feature_extractor((float*)limg.data, 960, 1280);
    return 0;
}
