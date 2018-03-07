#include <CovarianceComputer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/pcl_config.h>

#include <algorithm>

#include <sstream>
#include <string>


#include <svm_wrapper.h>


#if PCL_VERSION_COMPARE(>=,1,8,0)
#include <pcl/ml/kmeans.h>
#endif

#define CLOCKS_PER_MS (CLOCKS_PER_SEC/1000)

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct greater
{
  template<class T>
  bool operator()(T const &a, T const &b) const { return a > b; }
};


//name of the configurations files
const std::string TRAINING_CONFIG_FILE_NAME = "../config/training.yaml";
const std::string TEST_CONFIG_FILE_NAME = "../config/testing.yaml";


int main(int argc, char *argv[])
{
  clock_t global_time = clock();
  //Create object covariance computer which will compute all the covariance matrixes using configuration parameter from the file passed as input
  cout << "Starting the creation of the training dataset " << endl;
  CovarianceComputer cov_train(TRAINING_CONFIG_FILE_NAME);
  cout << "Starting the creation of the test dataset " << endl;
  CovarianceComputer cov_test(TEST_CONFIG_FILE_NAME);

  //value of sigma in the kernel computation formula
  float sigma = 1/2.0;


  cout << "\nSTARTING TRAINING KERNEL COMPUTATION WITH RGB IMAGES...";

  int first_person_train = cov_train.getFirstPerson();
  int first_person_test = cov_test.getFirstPerson();
  int number_of_persons = cov_train.getNumberOfPersons();
  int number_of_poses = cov_train.getNumberOfPoses();
  int number_of_photo_per_person = cov_train.getImagesPerPerson();
  int number_of_test = cov_test.getNumberOfPersons();

  clock_t start_time = clock();
  Eigen::MatrixXf kernelRgb(number_of_persons*number_of_poses,number_of_persons*number_of_poses);
  for(int i = 0;i < number_of_persons;i++){
    for(int j=0;j<number_of_poses;j++){
      Eigen::MatrixXf x1 = cov_train.getCovarianceRgb(i+first_person_train,j)*100;
      for(int k = 0;k < number_of_persons;k++){
        for(int h=0;h<number_of_poses;h++){
          Eigen::MatrixXf x2 = cov_train.getCovarianceRgb(k+first_person_train,h)*100;
          Eigen::MatrixXf sum = (x1 + x2)/2;
          int r = i*number_of_poses+j;
          int c = k*number_of_poses+h;
          kernelRgb(r,c)=exp(-sigma*(log(sum.determinant())-1/2.0*(log(x1.determinant())+log(x2.determinant()))));
        }
      }
    }
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;

  //saving the kernel in a file for debug
  std::ofstream outRgb("../config/kernelRgb.txt");
  if (outRgb.is_open())
  {
    outRgb << "KERNEL: \n" << kernelRgb << '\n';
  }
  outRgb.close();
  cout << "Kernel rgb saved in config/kernelRgb.txt \n" << endl;

  cout << "STARTING TRAINING KERNEL COMPUTATION WITH DEPTH IMAGES... ";

  start_time = clock();
  Eigen::MatrixXf kernelDepth(number_of_persons*number_of_poses,number_of_persons*number_of_poses);
  for(int i = 0;i < number_of_persons;i++){
    for(int j=0;j<number_of_poses;j++){
      Eigen::MatrixXf x1 = cov_train.getCovarianceDepth(i+first_person_train,j)*100;
      for(int k = 0;k < number_of_persons;k++){
        for(int h=0;h<number_of_poses;h++){
          Eigen::MatrixXf x2 = cov_train.getCovarianceDepth(k+first_person_train,h)*100;
          Eigen::MatrixXf sum = (x1 + x2)/2;
          int r = i*number_of_poses+j;
          int c = k*number_of_poses+h;
          kernelDepth(r,c)=exp(-sigma*(log(sum.determinant())-1/2.0*(log(x1.determinant())+log(x2.determinant()))));
        }
      }
    }
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;

  //saving the kernel in a file for debug
  std::ofstream outDepth("../config/kernelDepth.txt");
  if (outDepth.is_open())
  {
    outDepth << "KERNEL: \n" << kernelDepth << '\n';
  }
  outDepth.close();
  cout << "Kernel depth saved in config/kernelDepth.txt \n" << endl;

  std::string train_file_rgb = "../Train/trainRgb";
  std::string test_file_rgb = "../Train/testRgb";

  cout << "Creating train files for every person and training the svms with rgb images.. " << endl;

  start_time = clock();
  std::vector<svm_wrapper> svmsRgb(number_of_persons);
  for(int n=0;n<number_of_persons;n++)
  {
    std::stringstream curr_train_file;
    curr_train_file << train_file_rgb << n;
    std::ofstream out2(curr_train_file.str().c_str());
    for(int i= 0;i<number_of_persons;i++){
      if(i==n) out2 << 1 << " 0:" << i+1;
      else out2 << -1 << " 0:" << i+1;
      for(int j=0;j<number_of_persons;j++){
        out2 << " " << j+1 << ":" << kernelRgb(j,i);
      }
      out2 << "\n";
    }
    out2.close();

    svmsRgb[n].svm_training(curr_train_file.str());
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms. Saved in Train/trainRgb.. \n" << endl;

  std::string train_file_depth = "../Train/trainDepth";
  std::string test_file_depth = "../Train/testDepth";

  cout << "Creating train files for every person and training the svms with depth images.. " << endl;

  start_time = clock();
  std::vector<svm_wrapper> svmsDepth(number_of_persons);
  for(int n=0;n<number_of_persons;n++)
  {
    std::stringstream curr_train_file;
    curr_train_file << train_file_depth << n;
    std::ofstream out2(curr_train_file.str().c_str());
    for(int i= 0;i<number_of_persons;i++){
      if(i==n) out2 << 1 << " 0:" << i+1;
      else out2 << -1 << " 0:" << i+1;
      for(int j=0;j<number_of_persons;j++){
        out2 << " " << j+1 << ":" << kernelDepth(j,i);
      }
      out2 << "\n";
    }
    out2.close();
    //svmsDepth[n].set_c(10.);
    svmsDepth[n].svm_training(curr_train_file.str());
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms. Saved in Train/trainDepth.. \n" << endl;


  cout << "Creating test files for every person with rgb images.. ";

  start_time = clock();
  Eigen::MatrixXf testRgb(number_of_persons,number_of_test);
  for(int n=0;n<number_of_test;n++)
  {
    Eigen::MatrixXf covRgb;
    covRgb=cov_test.getCovarianceRgb(n+first_person_test,0)*100;
    //covRgb=cov_test.getCovarianceDepth(n,0)*100;
    for(int k = 0;k < number_of_persons;k++){
      for(int h=0;h<number_of_poses;h++){
        Eigen::MatrixXf x2 = cov_train.getCovarianceRgb(k+first_person_train,h)*100;
        //Eigen::MatrixXf x2 = cov_train.getCovarianceDepth(k,h)*100;
        Eigen::MatrixXf sum = (covRgb + x2)/2.0;
        testRgb(k,n)=exp(-sigma*(log(sum.determinant())-1/2.0*(log(x2.determinant())+log(covRgb.determinant()))));
      }
    }
  }

  for(int n=0;n<number_of_persons;n++){
    std::stringstream curr_test_file;
    curr_test_file << test_file_rgb << n;
    std::ofstream out2(curr_test_file.str().c_str());
    for(int i= 0;i<number_of_test;i++){
      out2 << 1 << " 0:1";
      for(int j=0;j<number_of_persons;j++){
        out2 << " " << j+1 << ":" << testRgb(j,i);
      }
      out2 << "\n";
    }
    out2.close();
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms. Saved in Train/testRgb" << endl;


  cout << "Creating test files for every person with depth images.. ";

  start_time = clock();
  Eigen::MatrixXf testDepth(number_of_persons,number_of_test);
  for(int n=0;n<number_of_test;n++)
  {
    Eigen::MatrixXf covDepth;
    covDepth=cov_test.getCovarianceDepth(n+first_person_test,0)*100;
    for(int k = 0;k < number_of_persons;k++){
      for(int h=0;h<number_of_poses;h++){
        Eigen::MatrixXf x2 = cov_train.getCovarianceDepth(k+first_person_train,h)*100;
        Eigen::MatrixXf sum = (covDepth + x2)/2.0;
        testDepth(k,n)=exp(-sigma*(log(sum.determinant())-1/2.0*(log(x2.determinant())+log(covDepth.determinant()))));
      }
    }
  }
  for(int n=0;n<number_of_persons;n++){
    std::stringstream curr_test_file;
    curr_test_file << test_file_depth << n;
    std::ofstream out2(curr_test_file.str().c_str());
    for(int i= 0;i<number_of_test;i++){
      out2 << 1 << " 0:1";
      for(int j=0;j<number_of_persons;j++){
        out2 << " " << j+1 << ":" << testDepth(j,i);
      }
      out2 << "\n";
    }
    out2.close();
  }
  cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms. Saved in Train/testDepth \n" << endl;

  Eigen::MatrixXf resultsRgb(2,number_of_test);
  Eigen::MatrixXf resultsDepth(2,number_of_test);
  for(int i=0;i<number_of_test;i++){
    resultsRgb(0,i)=-10;
    resultsDepth(0,i)=-10;
  }
  cout << "Testing " << test_file_rgb << " and " << test_file_depth  << " against every person svm..." << std::endl;

  start_time = clock();
  for(int n=0;n<number_of_persons;n++)
  {
    cout << " Predicting person " << n+first_person_train << endl;
    std::vector<double> distancesRgb,distancesDepth;
    std::stringstream curr_test_file_rgb;
    std::stringstream curr_test_file_depth;
    curr_test_file_rgb << test_file_rgb << n;
    curr_test_file_depth << test_file_depth << n;
    svmsRgb[n].svm_predicting(curr_test_file_rgb.str().c_str(),distancesRgb);
    svmsDepth[n].svm_predicting(curr_test_file_depth.str().c_str(),distancesDepth);

    for(int j=0;j<number_of_test;j++){

      if(distancesRgb[j]>resultsRgb(0,j)){
        resultsRgb(0,j)=distancesRgb[j];
        resultsRgb(1,j)=n+first_person_train;
      }
      if(distancesDepth[j]>resultsDepth(0,j)){
        resultsDepth(0,j)=distancesDepth[j];
        resultsDepth(1,j)=n+first_person_train;
      }

    }
  }

  cout << "\nTesting done in " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms. \n" << endl;

  //Thresholds and results
  //25 images RGB = 0.035, Depth -0.015 -> rank-1 = 100% FP-unknown = 0%
  //20 images RGB = 0.01, Depth -0.07 -> rank-1 = 100% FP-unknown = 0%
  //15 images RGB = -0.01, Depth -0.211 -> rank-1 = 96% FP-unkown = 20%
  //10 images RGB = -0.1755, Depth -0.267 -> rank-1 = 80% FP-unkown = 21.33%

  double thresholdRgb = 0, thresholdDepth = 0;

  switch(number_of_photo_per_person){
  case 25:
    thresholdRgb = 0.035;
    thresholdDepth = -0.015;
    break;
  case 20:
    thresholdRgb = -0.01;
    thresholdDepth = -0.07;
    break;
  case 15:
    thresholdRgb = -0.01;
    thresholdDepth = -0.211;
    break;
  case 10:
    thresholdRgb = -0.1755;
    thresholdDepth = -0.267;
    break;
  }


  int tp = 0, fp = 0;
  cout << "Result with " << number_of_photo_per_person << " images per person (Thresholds: Rgb = " << thresholdRgb << ", Depth = " << thresholdDepth << ") : " << endl;
  for(int n=0;n<number_of_test;n++){
    cout << "  Test " << n << ":\t RGB: " << resultsRgb(1,n) << "(" << resultsRgb(0,n) << ")" << "\t DEPTH: " << resultsDepth(1,n) << "(" << resultsDepth(0,n) << ")";
    if(resultsRgb(1,n)==resultsDepth(1,n) && resultsRgb(0,n)>thresholdRgb && resultsDepth(0,n)>thresholdDepth)
    {
      cout << "\t RESULTS: " << resultsRgb(1,n) << endl;
      if(resultsRgb(1,n) == n+first_person_test)
        tp++;
      else
        fp++;
    }
    else cout << "\t RESULTS: UNKNOWN"  << endl;
  }

  cout << "Recognition indexes: " << endl;
  cout << "  Rank-1 = " << (double)tp/number_of_persons * 100 << "%" << endl;
  cout << "  FP-unknown = " << (double)fp/(number_of_test-number_of_persons) * 100 << "%" << endl;

  cout << "Time to run the entire programm was " << (float)(clock() - global_time)/CLOCKS_PER_MS << " ms. \n" << endl;
  return 0;
}
