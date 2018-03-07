#include <svm.h>

#include <ctime>
#include <vector>

#include <iostream>
#include <sstream>
#include <string>

#define CLOCKS_PER_MS (CLOCKS_PER_SEC/1000)

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class svm_wrapper
{
public:
    /**
     * @brief svm_wrapper a wrapper class for the use of svm from libSVM, default constructor
     */
    svm_wrapper();
    /**
     * @brief svm_wrapper a wrapper class for the use of svm from libSVM
     * @param name the name of this svm
     * @param c_value the value of the parameter C
     */
    svm_wrapper(std::string name, double c_value = 1000);

    /**
     * @brief set_c set the value of the parameter C
     * @param c_value the value to set
     */
    void set_c(double c_value);

    /**
     * @brief svm_training will train this svm using the train file passed as parameter
     * @param train_file the path to the train file
     */
    void svm_training(std::string train_file);

    /**
     * @brief svm_predicting predict the labels of the input in test_file
     * @param test_file the input file with the data to test
     * @param distances return the distances from the hyperplane of every input data
     * @return the predicted labels of the input data
     */
    std::vector<double> svm_predicting(std::string test_file, std::vector<double> &distances);

private:

    std::string svm_name,model_file;
    //train
    struct svm_parameter param;
    struct svm_problem prob;
    struct svm_model *model;
    struct svm_node *x_space;

    int cross_validation;
    int nr_fold;

    char *line;
    int max_line_len;

    //predict
    struct svm_node *x;
    int max_nr_attr;


    /**
     * @brief read_problem read in a problem (in svmlight format)
     * @param filename the path of the file where the problem is saved
     */
    void read_problem(const char *filename);

    void exit_input_error(int line_num)
    {
        fprintf(stderr,"Wrong input format at line %d\n", line_num);
        exit(1);
    }

    /**
     * @brief readline read a line from the file in input
     * @param input the input file
     * @return a static char* with the line read
     */
    char* readline(FILE *input);

};
