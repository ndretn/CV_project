#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/eigen.h>
#include <math.h>

typedef Eigen::ArrayXf ulbpArray;

const int uniform[]={1,2,3,4,5,0,6,7,8,0,0,0,9,0,10,11,12,0,0,0,0,0,0,0,13,0,0,0,14,0,15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,19,0,0,0,20,
                     0,21,22,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,0,0,0,0,0,0,0,26,0,0,0,27,
                     0,28,29,30,31,0,32,0,0,0,33,0,0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,36,37,38,0,39,0,0,0,40,0,0,0,0,0,0,0,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,43,44,0,45,0,0,0,46,0,0,0,0,0,0,0,47,48,49,0,50,0,0,0,
                     51,52,53,0,54,55,56,57,58};

class ULBP
{
public:
    /**
     * @brief getULBP get the uniform local binary patterns of an image
     * @param image the input image in grayscale
     * @return an ulbpArray with the histogram of the input image
     */
    template<typename T>
    static ulbpArray getULBP(const cv::Mat& image){
        int rows = image.rows;
        int cols = image.cols;

        ulbpArray hist = ulbpArray::Zero(59);
        for(int i = 1;i<rows-1;i++)
        {
            for(int j=1;j<cols-1;j++)
            {

                T center = image.at<T>(i,j);
                T up = image.at<T>(i-1,j);
                T down = image.at<T>(i+1,j);
                T left = image.at<T>(i,j-1);
                T right = image.at<T>(i,j+1);
                float x,y;
                float cost = cos(2*M_PI/8.0);
                T lbp = 0;
                x = 1-cost;
                y = 1-cost;
                lbp+=((image.at<T>(i-1,j-1)*(1-x)*(1-y)+left*x*(1-y)+up*(1-x)*y+center*x*y)>=center);
                //lbp+=((image.at<T>(i-1,j-1))>=center);
                lbp += (up>=center) * 2;
                x = cost;
                y = 1-cost;
                lbp+= ((up*(1-x)*(1-y)+center*x*(1-y)+image.at<T>(i-1,j+1)*(1-x)*y+right*x*y)>=center)*4;
                //lbp+=((image.at<T>(i-1,j+1))>=center)*4;
                lbp += (right>=center) * 8;
                x = cost;
                y = cost;
                lbp+= ((center*(1-x)*(1-y)+down*x*(1-y)+right*(1-x)*y+image.at<T>(i+1,j+1)*x*y)>=center) * 16;
                //lbp+=((image.at<T>(i+1,j+1))>=center)*16;
                lbp += (down>=center) * 32;
                x = 1-cost;
                y = cost;
                lbp+= ((left*(1-x)*(1-y)+image.at<T>(i+1,j-1)*x*(1-y)+center*(1-x)*y+down*x*y)>=center) *64;
                //lbp+=((image.at<T>(i+1,j-1))>=center)*64;
                lbp += (left>=center) * 128;
                hist(uniform[lbp])++;
            }
        }
        return hist;
    }
};

