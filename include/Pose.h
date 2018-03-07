#include <Face.h>

class Pose
{
public:
    /**
     * @brief Pose class that contains the data of all the Faces in this pose of this person
     */
    Pose();

    /**
     * @brief addFace add a Face to this pose
     * @param face the face to add
     */
    void addFace(const Face face);

    /**
     * @brief addFace add a Face to this pose
     * @param face the face to add
     * @param rotation the rotation matrix of this face
     */
    void addFace(const Face face,const std::vector<float> rotation);

    /**
     * @brief getFace get the Face at the specified index
     * @param i the index
     * @return the Face at the i index
     */
    Face getFace(int i);

    /**
     * @brief removeFace remove the first Face in faces
     * @return the removed Face
     */
    Face removeFace();

    /**
     * @brief getRotation get the first rotation in rotations
     * @return the Rotation matrix
     */
    std::vector<float> getRotation();

    /**
     * @brief removeRotation remove the first rotation from rotations
     */
    void removeRotation();

    /**
     * @brief getCovarianceRgb get the covariance matrix for the rgb images
     * @return the covariance matrix computed from rgb images
     */
    Eigen::MatrixXf getCovarianceRgb();

    /**
     * @brief getCovarianceDepth get the covariance matrix for the depth images
     * @return the covariance matrix computed from depth images
     */
    Eigen::MatrixXf getCovarianceDepth();

    /**
     * @brief setCovarianceRgb set the covariance matrix to the one passed as parameter
     * @param cov the covariance matrix
     */
    void setCovarianceRgb(Eigen::MatrixXf cov);

    /**
     * @brief setCovarianceDepth set the covariance matrix to the one passed as parameter
     * @param cov the covariance matrix
     */
    void setCovarianceDepth(Eigen::MatrixXf cov);

    /**
     * @brief setCovariance set the covariance matrixes with the matrix passed as parameters
     * @param covRgb the rgb covariance matrix
     * @param covDepth the depth covariance matrix
     */
    void setCovariance(Eigen::MatrixXf covRgb,Eigen::MatrixXf covDepth);

    /**
     * @brief getNumberOfFaces get the current number of faces added in this pose
     * @return the current number of faces
     */
    int getNumberOfFaces();

    /**
     * @brief computeCovariance compute the covariance from the currently added faces
     */
    void computeCovariance();


private:
    //The differents faces of this person in this pose
    std::vector<Face> faces;
    //The rotation matrixes of the faces for the automatice pose detection
    std::vector<std::vector<float> > rotations;

    //The covariance matrix of rgb and depth
    Eigen::MatrixXf covariance_rgb;
    Eigen::MatrixXf covariance_depth;

    /**
     * @brief updateCovariance update the covariance values with the last added face
     */
    void updateCovariance();

    /**
     * @brief getMeanUlbpRgb get an array that is the mean of all the ulbp at index i of the faces
     * @param i the index of the square ulbp is calculated
     * @return an ulbpArray that is the mean of all the ulbps at index i of the faces in this pose
     */
    ulbpArray getMeanUlbpRgb(int i);

    /**
     * @brief getMeanUlbpDepth get an array that is the mean of all the ulbp at index i of the faces
     * @param i the index of the square ulbp is calculated
     * @return an ulbpArray that is the mean of all the ulbps at index i of the faces in this pose
     */
    ulbpArray getMeanUlbpDepth(int i);
};
