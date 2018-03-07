#include <Ulbp.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class Face
{
public:
    /**
     * @brief Face class that contains the data of a single face, this construct the class from rgb and depth face
     *        and compute the ulbp of both
     * @param rgb_image the rgb image of the face
     * @param depth_image the depth image of the face
     */
    Face(const cv::Mat& rgb_image,const cv::Mat& depth_image);

    /**
     * @brief getUlbpRgb get the vector with the 16 ulbp calculated from the rgb image of the face
     * @return a vector with 16 ulbpArray containing the ulbps
     */
    std::vector<ulbpArray> getUlbpRgb();

    /**
     * @brief getUlbpRgb get the vector with the 16 ulbp calculated from the depth image of the face
     * @return a vector with 16 ulbpArray containing the ulbps
     */
    std::vector<ulbpArray> getUlbpDepth();

    /**
     * @brief getUlbpRgb get the vector with the 16 ulbp calculated from the rgb image of the face
     * @param i the index of the ulbp in the vector
     * @return an ulbpArray containing the ulbps
     */
    ulbpArray getUlbpRgb(int i);

    /**
     * @brief getUlbpRgb get the ulbp calculated from the depth image of the face
     * @param i the index of the ulbp in the vector
     * @return an ulbpArray containing the ulbps
     */
    ulbpArray getUlbpDepth(int i);

    /**
     * @brief getRgbImage get the rgb image of this Face
     * @return the cv::Mat containing the rgb image
     */
    cv::Mat& getRgbImage();

    /**
     * @brief getRgbImage get the depth image of this Face
     * @return the cv::Mat containing the depth image
     */
    cv::Mat& getDepthImage();


private:
    //the rgb and depth of the face of the person
    cv::Mat rgb;
    cv::Mat depth;

    //The 16 ulbp of rgb and depth
    std::vector<ulbpArray> ulbp_rgb;
    std::vector<ulbpArray> ulbp_depth;

    /**
     * @brief computeUlbp compute the Ulbp from the rgb and depth images
     */
    void computeUlbp();
};
