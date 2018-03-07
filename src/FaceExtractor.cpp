#include <FaceExtractor.h>

//name of the configurations files
const std::string CAMERA_CONFIG_FILE_NAME = "../config/camera_info.yaml";
const std::string FACE_EXTRACTOR_CONFIG_FILE_NAME = "../config/face_extractor.yaml";

FaceExtractor::FaceExtractor():mean_z(0)
{
    //read configs from face_extractor.yaml
    cv::FileStorage config(FACE_EXTRACTOR_CONFIG_FILE_NAME,cv::FileStorage::READ);
    //read face_pose_estimator configs
    cv::FileNode config_forest_estimator = config["face_pose_estimator"];
    //path where to find the trees
    config_forest_estimator["path_to_trees"] >> trees_path;
    if( !estimator.loadForest(("../"+trees_path).c_str(), 10) ){

        cerr << "could not read forest!" << endl;
        exit(-1);
    }
    loadMatrixP();
}

FaceExtractor::FaceExtractor(const cv::Mat& input_depth,const cv::Mat& input_rgb):mean_z(0)
{
    //read configs from face_extractor.yaml
    cv::FileStorage config(FACE_EXTRACTOR_CONFIG_FILE_NAME,cv::FileStorage::READ);
    //read face_pose_estimator configs
    cv::FileNode config_forest_estimator = config["face_pose_estimator"];
    //path where to find the trees
    config_forest_estimator["path_to_trees"] >> trees_path;
    if( !estimator.loadForest(("../"+trees_path).c_str(), 10) ){

        cerr << "could not read forest!" << endl;
        exit(-1);
    }
    //cloud = input_cloud;
    depth_img = input_depth;
    rgb_img = input_rgb;

    loadMatrixP();
}



//----------------------------------------------------------PUBLIC METHODS--------------------------------------------------------------------------------

void FaceExtractor::setInputImages(const cv::Mat& input_depth,const cv::Mat& input_rgb)
{
    depth_img = input_depth;
    rgb_img = input_rgb;
}


std::vector<float> FaceExtractor::getHeadPose()
{   
    /*angles φ θ ψ
    * The rotation matrix is
    * R_m = [cos θ cos ψ , cos φ sin ψ + sin φ sin θ cos ψ , sin φ sin ψ - cos φ sin θ cos ψ
    *        -cos θ cos ψ , cos φ cos ψ - sin φ sin θ sin ψ , sin φ cos ψ + cos φ sin θ sin ψ
    *        sin θ , -sin φ cos θ , cos φ cos θ]
    */
    std::vector<float> head_pose(9);
    if(g_means.size() > 0)
    {
        float roll = g_means[0][3];
        float pitch = g_means[0][4];
        float yaw = g_means[0][5];
        head_pose[0] = cos(pitch)*cos(yaw);
        head_pose[1] = cos(roll)*sin(yaw)+sin(roll)*sin(pitch)*cos(yaw);
        head_pose[2] = sin(roll)*sin(yaw)-cos(roll)*sin(pitch)*cos(yaw);
        head_pose[3] = -cos(pitch)*cos(yaw);
        head_pose[4] = cos(roll)*cos(yaw)-sin(roll)*sin(pitch)*sin(yaw);
        head_pose[5] = sin(roll)*cos(yaw)+cos(roll)*sin(pitch)*sin(yaw);
        head_pose[6] = sin(pitch);
        head_pose[7] = -sin(roll)*cos(pitch);
        head_pose[8] = cos(roll)*cos(pitch);
    }
    return head_pose;
}


void FaceExtractor::cropHead(cv::Mat& output_head_depth, cv::Mat& output_head_rgb)
{
    float trees_no,max_variance,clustering_radius,ms_radius,stride,head_threshold,z_change,m,BETA,GAMMA;
    bool debug = false;

    //read configs from face_extractor.yaml
    cv::FileStorage config(FACE_EXTRACTOR_CONFIG_FILE_NAME,cv::FileStorage::READ);

    debug = ((std::string)config["show_debug_info"] == "true");

    //read face_pose_estimator configs
    cv::FileNode config_forest_estimator = config["face_pose_estimator"];
    //path where to find the trees
    config_forest_estimator["path_to_trees"] >> trees_path;
    //number of trees
    config_forest_estimator["trees_no"] >> trees_no;
    config_forest_estimator["max_variance"] >> max_variance;
    config_forest_estimator["clustering_radius"] >> clustering_radius;
    config_forest_estimator["ms_radius"] >> ms_radius;
    config_forest_estimator["stride"] >> stride;
    config_forest_estimator["head_threshold"] >> head_threshold;


    //read head_crop configs
    cv::FileNode config_head_crop = config["head_crop"];
    //number of not black pixel to find to select the current row/column as the first of the head
    config_head_crop["number_of_pixel"] >> m;
    //height of the head calulated as 100/(z-z_change) where z is the mean distance of the head from the sensor in m
    config_head_crop["face_h_changer"] >> z_change;
    //constants to redefined the crop using head pose informations
    //const float BETA = 5/8;
    //const float GAMMA = 5/8;
    config_head_crop["beta"] >> BETA;
    config_head_crop["gamma"] >> GAMMA;


    //transform the depth_image to a 3d_image with 3 channels (x,y) -> (X,Y,Z)
    transformDepthImageTo3DImage(depth_img,threeD_img,depth_img.cols,depth_img.rows);

    //find the orientation of the face
    std::vector< std::vector< Vote > > g_clusters; //full clusters of votes
    std::vector< Vote > g_votes; //all votes returned by the forest
    g_means.clear();
    g_votes.clear();
    g_clusters.clear();

    //default parameters
    //estimator.estimate(image3d,g_means,g_clusters,g_votes,5,1000,1.0,1.0,6.0,true,400);
    //parameter from head pose estimation main
    estimator.estimate(threeD_img,g_means,g_clusters,g_votes,stride,max_variance,1.0,clustering_radius,ms_radius,debug,head_threshold);

    if(debug)
        cout << "Heads found : " << g_means.size() << endl;

    if(debug && g_means.size()>0)
    {
        //the vector containing orientation information of the head (x,y,z,roll φ,pitch θ,yaw ψ)
        cout << "Estimated: " << g_means[0][0] << " " << g_means[0][1] << " " << g_means[0][2] << " " << g_means[0][3] << " " << g_means[0][4] << " " << g_means[0][5] <<endl;

        //show the point in the depth image
        pcl::PointXYZ p;
        p.x = g_means[0][0];
        p.y = g_means[0][1];
        p.z = g_means[0][2];

        // 3D to 2D projection:
        int x,y;
        projectPointTo2D(p,true,x,y);

        std::cout << "Corresponding 2D point in the image: " << x << " " << y << std::endl;

        cv::Mat rgb_img_vis = depth_img.clone();
        rgb_img_vis *= 1000;
        rgb_img_vis.convertTo(rgb_img_vis,CV_16U);
        cv::cvtColor(rgb_img_vis,rgb_img_vis,cv::COLOR_GRAY2BGR);
        // Draw a circle around the 2d point:
        cv::circle(rgb_img_vis,cv::Point(x,y),10,cv::Scalar(0,0,255), 3);

        //Show rgb image
        cv::namedWindow("Head RGB image");
        cv::imshow("Head RGB image",rgb_img_vis);

        //Apply rototranslation from depth to rgb
        Eigen::Vector4f homogeneous_point(p.x, p.y, p.z,1);

        std::cout << "Starting point " << homogeneous_point << std::endl;

        homogeneous_point = RTRgb * homogeneous_point;

        std::cout << "Arriving point " << homogeneous_point << std::endl;

        p.x = homogeneous_point[0];
        p.y = homogeneous_point[1];
        p.z = homogeneous_point[2];

        //now using the P of the rgb
        projectPointTo2D(p,false,x,y);

        std::cout << "Corresponding 2D point in the image: " << x << " " << y << std::endl;

        rgb_img_vis = rgb_img.clone();
        // Draw a circle around the 2d point:
        cv::circle(rgb_img_vis,cv::Point(x,y),10,cv::Scalar(0,0,255), 3);

        //Show rgb image
        cv::namedWindow("Head RGB image2");
        cv::imshow("Head RGB image2",rgb_img_vis);


    }

    //If a head is found it will be the only one. We now need to crop only the face from both the point cloud and the rgb image.

    //---------------------------------------------Create depth map from pointcloud---------------------------------------------------------------------------------
    //transformCloudToDepthImage(person_cloud,depth_img,cloud->width,cloud->height);

    if(debug)
    {
        //Show depth image
        cv::namedWindow("Depth image");
        cv::imshow("Depth image",depth_img*1000);
        cv::waitKey(0);
    }

    if(rgb_img.cols != depth_img.cols || rgb_img.rows != depth_img.rows)
    {
        std::cout << "RESIZE!"<< std::endl;
        cv::resize(rgb_img,rgb_img,cv::Size(depth_img.cols,depth_img.rows));
    }

    //the points where to crop
    int x_t=0,y_t=0,x_b=0,y_b=0;
    //the height of the head in relation to the distance from the sensor
    float head_height= 100/(((g_means.size()>0?g_means[0][2]:mean_z)-z_change)/1000);

    //number of not black pixel in current row/col
    int n = 0;

    //find the first row with more than m not black pixels
    for( int h = 0 ; h < depth_img.rows && n < m; ++h )
    {
        n = 0;
        for( int w = 0 ; w < depth_img.cols && n < m ; ++w )
        {
            if(depth_img.at<ushort>(h,w) > 0)
                n++;
        }
        y_t = h;
    }

    //this is the base of the head
    y_b = y_t+head_height;

    //now we crop the head from the rest of the body
    cv::Rect myROI(0, y_t, depth_img.cols, y_b-y_t);

    cv::Mat first_crop = depth_img(myROI);
    if(debug)
    {
        //Show depth image
        cv::namedWindow("First crop");
        cv::imshow("First crop",first_crop*1000);
        cv::waitKey(0);
    }
    n = 0;
    //find the first col with more than m not black pixels from left
    for( int w = 0 ; w < first_crop.cols && n < m ; ++w )
    {
        n = 0;
        for( int h = 0 ; h < first_crop.rows && n < m ; ++h )
        {
            if(first_crop.at<ushort>(h,w) > 0)
                n++;
        }
        x_t = w;
    }
    n = 0;
    //find the first col with more than m not black pixels from right
    for( int w = first_crop.cols - 1 ; w > 0 && n < m  ; --w )
    {
        n = 0;
        for( int h = 0 ; h < first_crop.rows && n < m ; ++h )
        {
            if(first_crop.at<ushort>(h,w) > 0)
                n++;
        }
        x_b = w;
    }

    if(debug)
        std::cout << "x_top= " << x_t << ", x_bot= " << x_b << std::endl;

    //redefine y_t using the head pose informations if the are any
    if(g_means.size()>0 && g_means[0][3] > 0)
        y_t = y_t+BETA*g_means[0][3]+GAMMA*g_means[0][5];
    //this is the base of the head
    y_b = y_t+head_height;

    if(debug)
        std::cout << "y_top= " << y_t << ", y_bot= " << y_b << std::endl;

    if(x_b-x_t > 0 && y_b-y_t > 0)
    {
        //now we can crop only the head
        cv::Rect myFinalROI(x_t, y_t, x_b-x_t, y_b-y_t);

        output_head_depth = depth_img(myFinalROI);

        if(debug)
        {
            std::cout << "Xt before " << x_t << " Yt before " << y_t << std::endl;
            std::cout << "Xb before " << x_b << " Yb before " << y_b << std::endl;
        }

        getCorrespondingRgbPoint(depth_img,x_t,y_t,x_t,y_t);
        getCorrespondingRgbPoint(depth_img,x_b,y_b,x_b,y_b);

        if(debug)
        {
            std::cout << "Xt after " << x_t << " Yt after " << y_t << std::endl;
            std::cout << "Xb after " << x_b << " Yb after " << y_b << std::endl;
        }

        //int offset = 10;
        //cv::Rect myFinalROIRgb(x_t-offset,y_t+offset,x_b-x_t-offset,y_b-y_t+offset);
        cv::Rect myFinalROIRgb(x_t, y_t, x_b-x_t, y_b-y_t);

        output_head_rgb = rgb_img(myFinalROIRgb);
        if(debug)
        {
            //Show depth image
            cv::namedWindow("Final crop depth");
            cv::imshow("Final crop depth",output_head_depth*1000);

            //Show depth image
            cv::namedWindow("Final crop rgb");
            cv::imshow("Final crop rgb",output_head_rgb);

            cv::waitKey(0);
        }
    }
    else
        std::cerr << "No head found!" << std::endl;

}

//---------------------------------------------------------------PRIVATE METHODS--------------------------------------------------------------------------



void FaceExtractor::transformDepthImageTo3DImage(const cv::Mat &input_depth, cv::Mat &output_img, int width, int height)
{
    float depth_focal_inverted_x = 1/PDepth(0,0);  // 1/fx
    float depth_focal_inverted_y = 1/PDepth(1,1);  // 1/fy

    int n_point = 0;

    output_img.create( height, width, CV_32FC3 );
    output_img.setTo(0);

    //get 3D from depth
    for(int y = 0; y < output_img.rows; y++)
    {
        cv::Vec3f* img3Di = output_img.ptr<cv::Vec3f>(y);
        const int16_t* depthImgi = input_depth.ptr<int16_t>(y);

        for(int x = 0; x < output_img.cols; x++){

            float d = (float)depthImgi[x];

            if ( d < 3000 && d > 0 ){

                img3Di[x][0] = d * (float(x) - PDepth(0,2)) * depth_focal_inverted_x;
                img3Di[x][1] = d * (float(y) - PDepth(1,2)) * depth_focal_inverted_y;
                img3Di[x][2] = d;

                mean_z += d;
                n_point++;
            }
            else{

                img3Di[x] = 0;
            }

        }
    }
    mean_z /= n_point;

}

pcl::PointXYZ FaceExtractor::projectDepthPointTo3D(const cv::Mat &input_depth,const int x,const int y)
{
    float depth_focal_inverted_x = 1/PDepth(0,0);  // 1/fx
    float depth_focal_inverted_y = 1/PDepth(1,1);  // 1/fy

    pcl::PointXYZ point3D;
    const int16_t* depthImgi = input_depth.ptr<int16_t>(y);

    float d = (float)depthImgi[x];

    point3D.x = d * (float(x) - PDepth(0,2)) * depth_focal_inverted_x;
    point3D.y = d * (float(y) - PDepth(1,2)) * depth_focal_inverted_y;
    point3D.z = d;

    return point3D;

}

void FaceExtractor::projectPointTo2D(const pcl::PointXYZ point, bool depth, int& x, int& y)
{
    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Eigen::Vector3f homogeneous_point(point.x, point.y, point.z);
    Eigen::Vector3f point_2d = (depth?PDepth:PRgb) * homogeneous_point;
    x = point_2d[0] / point_2d[2];
    y = point_2d[1] / point_2d[2];
}

void FaceExtractor::getCorrespondingRgbPoint(const cv::Mat &input_depth, const int x,const int y, int& x_out, int& y_out)
{

    pcl::PointXYZ point3D = projectDepthPointTo3D(input_depth,x,y);

    if(point3D.z != 0)
    {
        //Apply rototranslation from depth to rgb
        Eigen::Vector4f homogeneous_point(point3D.x, point3D.y, point3D.z,1);

        homogeneous_point = RTRgb * homogeneous_point;

        point3D.x = homogeneous_point[0];
        point3D.y = homogeneous_point[1];
        point3D.z = homogeneous_point[2];

        projectPointTo2D(point3D,false,x_out,y_out);
    }
    else
    {
        x_out = x;
        y_out = y;
    }
}

void FaceExtractor::loadMatrixP()
{
    //read P matrix from camera_info.yaml
    cv::FileStorage config_camera(CAMERA_CONFIG_FILE_NAME,cv::FileStorage::READ);

    //Load intrinsic parameter of depth camera
    cv::FileNode seq = config_camera["PDepth"];
    cv::FileNodeIterator it = seq.begin();

    for(int i = 0; i<9; i++, ++it)
    {
        PDepth(i/3,i%3)=(float)*it;
    }

    //load intrinsic parameter of rgb camera
    seq = config_camera["PRgb"];
    it = seq.begin();

    for(int i = 0; i<9; i++, ++it)
    {
        PRgb(i/3,i%3)=(float)*it;
    }

    //load Rototranslation matrix of the rgb camera
    seq = config_camera["RRgb"];
    it = seq.begin();

    for(int i = 0; i<9; i++, ++it)
    {
        RTRgb(i/3,i%3)=(float)*it;
    }

    seq = config_camera["TRgb"];
    it = seq.begin();

    for(int i = 0; i<3; i++, ++it)
    {
        RTRgb(i,3)=(float)*it;
        //put 0001 in the last row
        RTRgb(3,i)=0;
    }

    RTRgb(3,3) = 1;



}
