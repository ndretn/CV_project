#include <svm_wrapper.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//-------------------------CONSTRUCTORS------------------------------------
svm_wrapper::svm_wrapper():max_nr_attr(64)
{
    param.C = 1000;
}

svm_wrapper::svm_wrapper(std::string name, double c_value): svm_name(name),max_nr_attr(64)
{
    param.C = c_value;
}


//------------------------------PUBLIC METHODS---------------------------------
void svm_wrapper::set_c(double c_value)
{
    param.C = c_value;
}

void svm_wrapper::svm_training(std::string train_file)
{
    std::cout << "Starting SVM train...";
    clock_t start_time = clock();
    //SVM construction
    // default values
    param.svm_type = C_SVC;
    param.kernel_type = PRECOMPUTED;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 10000;
    //param.C = 1000;
    param.eps = 1e-15;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 1;
    param.weight_label = (int *)malloc(sizeof(int));
    param.weight = (double *)malloc(sizeof(double));
    param.weight_label[0] = 1;
    param.weight[0] = 1;
    cross_validation = 0;


    read_problem(train_file.c_str());

    const char *error_msg;
    error_msg = svm_check_parameter(&prob,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }

    model = svm_train(&prob,&param);

    std::stringstream model_file_name;
    model_file_name << train_file << ".model";

    model_file = model_file_name.str();

    if(svm_save_model(model_file.c_str(),model))
    {
        fprintf(stderr, "can't save model to file %s\n", model_file_name.str().c_str());
        exit(1);
    }
    svm_free_and_destroy_model(&model);

    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(x_space);
    free(line);

    std::cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << std::endl;
}

std::vector<double> svm_wrapper::svm_predicting(std::string test_file, std::vector<double>& distances)
{
    std::cout << "Starting SVM predict...";
    clock_t start_time = clock();

    int correct = 0;
    int total = 0;
    double error = 0,errno;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
    std::vector<double> labels;

    FILE *input;

    input = fopen(test_file.c_str(),"r");

    model = svm_load_model(model_file.c_str());

    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;
    int j;

    max_line_len = 1024;
    line = (char *)malloc(max_line_len*sizeof(char));
    while(readline(input) != NULL)
    {
        int i = 0;
        double target_label, predict_label;
        char *idx, *val, *label, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(total+1);


        target_label = strtod(label,&endptr);

        if(endptr == label || *endptr != '\0')
            exit_input_error(total+1);

        while(1)
        {
            if(i>=max_nr_attr-1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;
            errno = 0;
            x[i].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
                exit_input_error(total+1);
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(total+1);

            ++i;
        }
        x[i].index = -1;

        double distance = 0;
        predict_label = svm_predict(model,x,&distance);
        labels.push_back(predict_label);

        distances.push_back(distance);

        //std::cout << "Label " << predict_label << ", distance from hyperplane " << distance << "\n" << std::endl;


        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }
    //std::cout << "Accuracy = "<< (double)correct/total*100 << "% (" << correct << "/" << total <<")(classification)" << std::endl;

    svm_free_and_destroy_model(&model);
    free(x);
    free(line);
    fclose(input);
    std::cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << std::endl;

    return labels;
}


//----------------------------PRIVATE METHODS-----------------------------------
void svm_wrapper::read_problem(const char *filename)
{
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;
    double errno;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob.l = 0;
    elements = 0;

    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements;
        ++prob.l;
    }
    rewind(fp);

    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct svm_node *,prob.l);
    x_space = Malloc(struct svm_node,elements);

    max_index = 0;
    j=0;
    for(i=0;i<prob.l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);

        prob.y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }

    if(param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;

    if(param.kernel_type == PRECOMPUTED)
        for(i=0;i<prob.l;i++)
        {
            if (prob.x[i][0].index != 0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
        }

    fclose(fp);
}

char* svm_wrapper::readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}
