#include <Pose.h>

Pose::Pose():covariance_rgb(16,16),covariance_depth(16,16)
{
    covariance_rgb = Eigen::MatrixXf::Zero(16,16);
    covariance_depth = Eigen::MatrixXf::Zero(16,16);
}

//-------------------------------------PUBLIC METHODS----------------------------------------

void Pose::addFace(const Face face)
{
    faces.push_back(face);
    computeCovariance();
}

void Pose::addFace(const Face face,const std::vector<float> rotation)
{
    faces.push_back(face);
    rotations.push_back(rotation);
    computeCovariance();
}

Face Pose::getFace(int i)
{
    return faces[i];
}

Face Pose::removeFace()
{
    Face f = faces[0];
    faces.erase(faces.begin());
    computeCovariance();
    removeRotation();
    return f;
}

std::vector<float> Pose::getRotation()
{
    return rotations[0];
}

void Pose::removeRotation()
{
    rotations.erase(rotations.begin());
}

Eigen::MatrixXf Pose::getCovarianceRgb()
{
    return covariance_rgb;
}

Eigen::MatrixXf Pose::getCovarianceDepth()
{
    return covariance_depth;
}

void Pose::setCovarianceRgb(Eigen::MatrixXf cov)
{
    covariance_rgb = cov;
}

void Pose::setCovarianceDepth(Eigen::MatrixXf cov)
{
    covariance_depth = cov;
}

void Pose::setCovariance(Eigen::MatrixXf covRgb,Eigen::MatrixXf covDepth)
{
    setCovarianceRgb(covRgb);
    setCovarianceDepth(covDepth);
}

int Pose::getNumberOfFaces()
{
    return faces.size();
}

void Pose::computeCovariance()
{
    //compute the variance matrixes from the current faces
    //rgb covariance
    for(int row=0;row<covariance_rgb.rows();row++)
    {
        for(int col=0;col<covariance_rgb.cols();col++)
        {
            float c=0;

            ulbpArray x_row_mean = getMeanUlbpRgb(row);
            ulbpArray x_col_mean = getMeanUlbpRgb(col);

            for(int i=0; i<faces.size(); i++)
            {
                ulbpArray x_row = faces[i].getUlbpRgb(row);
                ulbpArray x_col = faces[i].getUlbpRgb(col);

                c += (x_row - x_row_mean).matrix().transpose() * (x_col-x_col_mean).matrix();
            }
            covariance_rgb(row,col) = c/faces.size();
        }
    }

    //depth covariance
    for(int row=0;row<covariance_depth.rows();row++)
    {
        for(int col=0;col<covariance_depth.cols();col++)
        {
            float c=0;

            ulbpArray x_row_mean = getMeanUlbpDepth(row);
            ulbpArray x_col_mean = getMeanUlbpDepth(col);

            for(int i=0; i<faces.size(); i++)
            {
                ulbpArray x_row = faces[i].getUlbpDepth(row);
                ulbpArray x_col = faces[i].getUlbpDepth(col);

                c += (x_row - x_row_mean).matrix().transpose() * (x_col-x_col_mean).matrix();
            }
            covariance_depth(row,col) = c/faces.size();
        }
    }
}

//-------------------------------------PRIVATE METHODS---------------------------------------

void Pose::updateCovariance()
{
    //update the values in the covariance matrixes with the last added face
    //rgb covariance
    for(int row=0;row<covariance_rgb.rows();row++)
    {
        for(int col=0;col<covariance_rgb.cols();col++)
        {
            float c = covariance_rgb(row,col)*(faces.size()-1);

            ulbpArray x_row = faces[faces.size()-1].getUlbpRgb(row);
            ulbpArray x_col = faces[faces.size()-1].getUlbpRgb(col);

            ulbpArray x_row_mean = getMeanUlbpRgb(row);
            ulbpArray x_col_mean = getMeanUlbpRgb(col);

            c += (x_row - x_row_mean).matrix().transpose() * (x_col-x_col_mean).matrix();

            covariance_rgb(row,col) = c/faces.size();
        }
    }

    //depth covariance
    for(int row=0;row<covariance_depth.rows();row++)
    {
        for(int col=0;col<covariance_depth.cols();col++)
        {
            float c = covariance_depth(row,col) * (faces.size()-1);

            ulbpArray x_row = faces[faces.size()-1].getUlbpDepth(row);
            ulbpArray x_col = faces[faces.size()-1].getUlbpDepth(col);

            ulbpArray x_row_mean = getMeanUlbpDepth(row);
            ulbpArray x_col_mean = getMeanUlbpDepth(col);

            c += (x_row - x_row_mean).matrix().transpose() * (x_col-x_col_mean).matrix();

            covariance_depth(row,col) = c/faces.size();
        }
    }
}

ulbpArray Pose::getMeanUlbpRgb(int i)
{
    ulbpArray mean = ulbpArray::Zero(59);

    for(int j=0; j<faces.size(); j++)
    {
        mean += faces[j].getUlbpRgb(i);
    }
    return mean/faces.size();
}

ulbpArray Pose::getMeanUlbpDepth(int i)
{
    ulbpArray mean = ulbpArray::Zero(59);

    for(int j=0; j<faces.size(); j++)
    {
        mean += faces[j].getUlbpDepth(i);
    }
    return mean/faces.size();
}
