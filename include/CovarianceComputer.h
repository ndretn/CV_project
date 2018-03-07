#include <FaceExtractor.h>
#include <Pose.h>

#include <pcl/io/pcd_io.h>
#include <pcl/pcl_config.h>

#include <sstream>
#include <string>

#if PCL_VERSION_COMPARE(>=,1,8,0)
    #include <pcl/ml/kmeans.h>
#endif

#define CLOCKS_PER_MS (CLOCKS_PER_SEC/1000)

class CovarianceComputer
{
public:

    /**
     * @brief CovarianceComputer constructor of the class, will load from files or from images and compute all the covariance matrixes
     * @param config_file the path to the config file for this CovarianceComputer
     */
    CovarianceComputer(std::string config_file);
    /**
     * @brief getCovarianceRgb return the covariance matrix of the selected person in the selected pose for the rgb
     * @param person the index of the person to select
     * @param pose the index of the pose to select
     * @return a Eigen::MatrixXf with the covariance matrix
     */
    Eigen::MatrixXf getCovarianceRgb(int person,int pose);

    /**
     * @brief getCovarianceDepth return the covariance matrix of the selected person in the selected pose for the depth
     * @param person the index of the person to select
     * @param pose the index of the pose to select
     * @return a Eigen::MatrixXf with the covariance matrix
     */
    Eigen::MatrixXf getCovarianceDepth(int person,int pose);

    /**
     * @brief getNumberOfPersons return the number of persons of which covariance have been computed
     * @return the number of persons
     */
    int getNumberOfPersons();

    /**
     * @brief getNumberOfPoses return the number of poses every person has
     * @return the number of poses
     */
    int getNumberOfPoses();

    /**
     * @brief getImagesPerPerson return the number of photo per person that has been used
     * @return the number of photo per person
     */
    int getImagesPerPerson();

    /**
     * @brief getFirstPerson return the index of the first person
     * @return the index of the first person
     */
    int getFirstPerson();

private:

    //This is where all necessary data will be saved
    std::vector<std::vector<Pose> > persons;

    //Config parameters
    bool debug, automatic_pose, train, move_images;
    int number_of_persons,image_per_person,number_of_poses,starting_person,starting_photo;
    std::string input_path,output_path,train_path;
    FaceExtractor extractor;

    /**
     * @brief computeFromImages compute all the covariance matrixes from the input images, then save them to files
     */
    void computeFromImages();

    /**
     * @brief loadFromFile load the covariance matrixes computed previously from files
     */
    void loadFromFile();

    /**
     * @brief loadDepthImageCompressed load depth image from bin file for the biwi database
     * @param fname the name of the file to load
     * @param depthImg the cv::Mat where to put the data
     * @return true if all is gone well
     */
    bool loadDepthImageCompressed( std::string fname , cv::Mat& depthImg);

    /**
     * @brief write_binary write the matrix in a binary file
     * @param filename a const char* containing the path of the file to write
     * @param matrix the Eigen::Matrix to save in the file
     */
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix){
        std::ofstream out(filename,ios::out | ios::binary | ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }

    /**
     * @brief read_binary read the matrix from a binary file
     * @param filename a const char* containing the path of the file to read
     * @param matrix the Eigen::Matrix where to load the data of the file
     */
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename,ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }

};
