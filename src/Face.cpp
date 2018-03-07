#include <Face.h>


Face::Face(const cv::Mat& rgb_image,const cv::Mat& depth_image):ulbp_rgb(16),ulbp_depth(16)
{
    rgb = rgb_image;
    depth = depth_image;
    computeUlbp();
}

//-------------------------------------PUBLIC METHODS----------------------------------------


std::vector<ulbpArray> Face::getUlbpRgb()
{
    return ulbp_rgb;
}

std::vector<ulbpArray> Face::getUlbpDepth()
{
    return ulbp_depth;
}

ulbpArray Face::getUlbpRgb(int i)
{
    return ulbp_rgb[i];
}

ulbpArray Face::getUlbpDepth(int i)
{
    return ulbp_depth[i];
}

cv::Mat& Face::getRgbImage()
{
    return rgb;
}

cv::Mat& Face::getDepthImage()
{
    return depth;
}


//-------------------------------------PRIVATE METHODS---------------------------------------
void Face::computeUlbp()
{    
    int div_number = 4;
    //Compute rgb ulbp
    int rows_for_ulbp = rgb.rows/div_number;
    int cols_for_ulbp = rgb.cols/div_number;

    cv::Mat gray;
    cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    for(int i=0;i<ulbp_rgb.size();i++)
    {
        int row = i/div_number;
        int col = i%div_number;

        //select a square from the original image
        int starting_col = col*cols_for_ulbp;
        int starting_row = row*rows_for_ulbp;
        cv::Rect ROI(starting_col, starting_row, cols_for_ulbp, rows_for_ulbp);
        cv::Mat square = gray(ROI);

        //compute ulbp for this square and save it in the vector
        ulbp_rgb[i] = ULBP::getULBP<uchar>(square);
        //normalize ulbp
        ulbp_rgb[i] /= ulbp_rgb[i].sum();
    }

    //Compute depth ulbp
    rows_for_ulbp = depth.rows/div_number;
    cols_for_ulbp = depth.cols/div_number;
    for(int i=0;i<ulbp_depth.size();i++)
    {
        int row = i/div_number;
        int col = i%div_number;

        //select a square from the original image
        int starting_col = col*cols_for_ulbp;
        int starting_row = row*rows_for_ulbp;
        cv::Rect ROI(starting_col, starting_row, cols_for_ulbp, rows_for_ulbp);
        cv::Mat square = depth(ROI);

        //compute ulbp for this square and save it in the vector
        ulbp_depth[i] = ULBP::getULBP<ushort>(square);
        //normalize ulbp
        ulbp_depth[i] /= ulbp_depth[i].sum();
    }
}
