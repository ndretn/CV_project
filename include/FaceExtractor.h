#include <ctime>
#include <iostream>
#include <math.h>

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CRForestEstimator.h>

#include <Eigen/Geometry>



#define CLOCKS_PER_MS (CLOCKS_PER_SEC/1000)

class FaceExtractor
{
public:

    /**
     * @brief FaceExtractor create an object of the class FaceExtractor, this class can be used to extract the face of the person
     *          from a depth image and the corresponding rgb image.
     */
    FaceExtractor();

    /**
     * @brief FaceExtractor create an object of the class FaceExtractor, this class can be used to extract the face of the person
     *          from a depth image and the corresponding rgb image.
     * @param input_depth the input depth where the face will be extracted
     * @param input_rgb the input rgb image of which to extract the face
     */
    FaceExtractor(const cv::Mat& input_depth,const cv::Mat& input_rgb);

    /**
     * @brief setInputImages set the input images from where to extract the face
     * @param input_depth
     * @param input_rgb
     */
    void setInputImages(const cv::Mat& input_depth,const cv::Mat& input_rgb);

    /**
     * @brief getPersonCloud get the cloud of the person
     * @return the cloud with only the person
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPersonCloud();

    /**
     * @brief cropHead crop the head from the input depth_image using orientation information in g_means
     * @param output_head_depth the output depth_image with only the head
     * @param output_head_rgb the output rgb with only the head
     */
    void cropHead(cv::Mat& output_head_depth, cv::Mat& output_head_rgb);

    /**
     * @brief getHeadCenter get the PointXYZ estimated to be the center of the head
     * @return the PointXYZ center of the head if cropHead was called a (0,0,0) point otherwise
     */
    pcl::PointXYZ getHeadCenter();

    /**
     * @brief getHeadPose get the rotation matrix of the pose of the head
     * @return a vector<float> with the rotation matrix starting from roll pitch and yaw of the pose of the head if cropHead was called a (0,0,0,0,0,0,0,0,0) point otherwise
     * the rotation matrix is
     * R_m = [cos θ cos ψ , cos φ sin ψ + sin φ sin θ cos ψ , sin φ sin ψ - cos φ sin θ cos ψ
     *        -cos θ cos ψ , cos φ cos ψ - sin φ sin θ sin ψ , sin φ cos ψ + cos φ sin θ sin ψ
     *        sin θ , -sin φ cos θ , cos φ cos θ]
     */
    std::vector<float> getHeadPose();

private:
    cv::Mat rgb_img,threeD_img,depth_img;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,person_cloud;
    float mean_z;
    std::vector< cv::Vec<float,POSE_SIZE> > g_means;
    Eigen::Matrix<float, 3,3> PDepth,PRgb;
    Eigen::Matrix<float, 4,4> RTRgb;
    CRForestEstimator estimator;
    std::string trees_path;

    /**
     * @brief segmentate segmentate the background from the person that is at the center of the pointcloud
     * @param input_cloud the input cloud with a person in the center
     * @param output_cloud the output cloud with only the person
     */
    void segmentate(const pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud);

    /**
     * @brief transformDepthImageTo3DImage transform a depth image to a 3d image with (x,y,z) coordinates for each pixel
     * @param input_depth the input depth image
     * @param output_img the output 3d image
     * @param width the width of the output image
     * @param height the height of the output image
     */
    void transformDepthImageTo3DImage(const cv::Mat& input_depth, cv::Mat& output_img, int width, int height);

    /**
     * @brief projectDepthPointTo3D project a depth point in to a 3d point in world coordinate
     * @param input_depth the input depth image
     * @param x the input x
     * @param y the input y
     * @return a point3D with the output X,Y,Z in world coordinate
     */
    pcl::PointXYZ projectDepthPointTo3D(const cv::Mat& input_depth, const int x,const int y);

    /**
     * @brief projectPointTo2D project a 3D point into the original 2D point of the image
     * @param point the input 3D point
     * @param depth true if you want to project point into the depth image, false if you want into the rgb image
     * @param x the output x in the image
     * @param y the output y in the image
     */
    void projectPointTo2D(const pcl::PointXYZ point, bool depth, int& x, int& y);

    /**
     * @brief projectPointTo2D project a 3D point into the original 2D point of the image using P/2
     * @param point the input 3D point
     * @param x the output x in the image
     * @param y the output y in the image
     */
    void projectPointTo2DHalfP(const pcl::PointXYZ point, int& x, int& y);

    /**
     * @brief getCorrespondingRgbPoint get the corresponding rgb point from a depth point
     * @param input_depth the input depth image
     * @param x the depth x
     * @param y the depth y
     * @param x_out the rgb x
     * @param y_out the rgb y
     */
    void getCorrespondingRgbPoint(const cv::Mat &input_depth, const int x,const int y, int& x_out, int& y_out);

    /**
     * @brief loadMatrixP load the P matrix from the file in CAMERA_CONFIG_FILE_NAME
     */
    void loadMatrixP();
};
