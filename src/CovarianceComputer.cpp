#include <CovarianceComputer.h>




CovarianceComputer::CovarianceComputer(std::string config_file):debug(false), automatic_pose(false), train(true), move_images(false), number_of_persons(1),
    image_per_person(1),number_of_poses(1),starting_person(0),starting_photo(0)
{
    //-----------------------------------------------Read configs from training.yaml--------------------------------------------------
    cv::FileStorage config(config_file,cv::FileStorage::READ);

    debug = ((std::string)config["show_debug_info"] == "true");
    automatic_pose = ((std::string)config["automatic_pose_detection"] == "true");
    train = ((std::string)config["train"] == "true");
    move_images = ((std::string)config["move_used_images"] == "true");

    config["number_of_persons"] >> number_of_persons;
    //default value if this parameter is not specified in the configuration file
    if(number_of_persons==0)
        number_of_persons = 1;
    config["number_of_photo_per_person"] >> image_per_person;
    //default value if this parameter is not specified in the configuration file
    if(image_per_person==0)
        image_per_person = 1;

    config["number_of_poses"] >> number_of_poses;
    //default value if this parameter is not specified in the configuration file
    if(number_of_poses==0)
        number_of_poses = 1;


    config["starting_person"] >> starting_person;
    config["starting_photo"] >> starting_photo;

    input_path = "../" + (std::string)config["input_path"];
    output_path = "../" + (std::string)config["output_path"];
    train_path = "../" + (std::string)config["save_path"];

    //initialize the main data structure
    persons = std::vector<std::vector<Pose> >(number_of_persons);

    clock_t start_time = clock();
    if(train)
    {
        std::cout << "Reading all input images to compute covariance matrixes..." << std::endl;
        computeFromImages();
        cout << "Covariance matrixes computed in " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;
    }
    else
    {
        std::cout << "The covariance matrixes were already computed, loading them from files..." << std::endl;
        loadFromFile();
        cout << "Covariance matrixes loaded in " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;
    }
}

//----------------------------------------------------------PUBLIC METHODS--------------------------------------------------------------------------------

Eigen::MatrixXf CovarianceComputer::getCovarianceRgb(int person, int pose)
{
    assert(person-starting_person < number_of_persons && pose < number_of_poses);
    return persons[person-starting_person][pose].getCovarianceRgb();
}

Eigen::MatrixXf CovarianceComputer::getCovarianceDepth(int person, int pose)
{
    assert(person-starting_person < number_of_persons && pose < number_of_poses);
    return persons[person-starting_person][pose].getCovarianceDepth();
}

int CovarianceComputer::getNumberOfPersons()
{
    return number_of_persons;
}

int CovarianceComputer::getNumberOfPoses()
{
    return number_of_poses;
}

int CovarianceComputer::getImagesPerPerson()
{
    return image_per_person;
}

int CovarianceComputer::getFirstPerson()
{
    return starting_person;
}

//---------------------------------------------------------------PRIVATE METHODS--------------------------------------------------------------------------

void CovarianceComputer::computeFromImages()
{
    if(automatic_pose)
    {
#if PCL_VERSION_COMPARE(<,1,8,0)
        std::cerr << "Automatic pose detection require PCL 1.8 or greater, the poses will not be automaticly detected!" << std::endl;
        automatic_pose = false;
#endif
    }
    if(!automatic_pose && number_of_poses!=5 && number_of_poses!=1)
    {
        std::cerr << "Manual pose detection works only with 1 and 5 as number of poses, it will now use 1 automatically" << std::endl;
        number_of_poses = 1;
    }

    //---------------------------------------------Load input data-----------------------------------------------------------

    //This is where pose estimation is saved for the automatic pose detection
#if PCL_VERSION_COMPARE(>=,1,8,0)
    std::vector<pcl::Kmeans::Point> poses;
#endif
    cv::Mat image,depth;

    for(int i=starting_person;i<number_of_persons+starting_person;i++)
    {
        //Strings for input file names
        std::string rgb_image;
        std::string depth_image;
        std::string depth_out;
        std::string face;

        //Create for every person a vector of number_of_poses poses
        persons[i-starting_person] = std::vector<Pose>(number_of_poses);

        std::vector<bool> found_persons(1000,false);

        //if(i == 15 || i == 18 || i == 21 || i == 22 )
        //    continue;
        int tempi = i;
        if(tempi >= 15)
            tempi = i + 1;
        if(tempi >= 18)
            tempi = i + 2;
        if(tempi >= 21)
            tempi = i + 4;

        std::cout << "Reading " << image_per_person <<" images of person " << i << std::flush;

        clock_t start_time = clock();
        int found_photo = 0;
        for(int t=starting_photo;found_photo < image_per_person;t++)
        {
            std::stringstream prefix;
            if(i<10) prefix << "0" << tempi <<"/";
            else prefix << tempi << "/";

            rgb_image = input_path + prefix.str();
            depth_image = input_path + prefix.str();
            int j;
            do{
                j = rand() % 1000;
            }
            while(found_persons[j]);

            std::stringstream suffix;
            if(j<10) suffix << "frame_0000" << j << "_";
            else if(j<100) suffix << "frame_000" << j << "_";
            else suffix << "frame_00" << j << "_";

            rgb_image += suffix.str() + "rgb.png";
            depth_image += suffix.str() + "depth.bin";

            //read rgb image and point cloud
            image = cv::imread(rgb_image);
            if(!image.data)
            {
                if(debug)
                    std::cerr << "Couldn't read the rgb image " << rgb_image << ", skipping this image."<< std::endl;

                if(t>10000000)
                {
                    std::cerr << "There aren't " << image_per_person << " photo of this person!" << std::endl;
                    return;
                }

                continue;
            }

            if (!loadDepthImageCompressed(depth_image,depth)) //* load the file
            {
                if(debug)
                    std::cerr << "Couldn't read the depth file " << depth_image << ", skipping this image."<< std::endl;
                continue;
            }

            if(debug)
            {
                cout << endl;
                cout << "Processing data from the image " << j << " of person " << tempi <<endl;
            }
            else
                std::cout << ".";


            found_photo++;
            found_persons[j] = true;                

            if(move_images)
            {
                //move the used file
                std::stringstream supp;
                supp << input_path << "used/" << prefix.str();
                system(("mkdir -p "+supp.str()).c_str());
                supp << suffix.str();
                std::string used_image_rgb = supp.str() + "rgb.png";
                std::string used_image_depth = supp.str() + "depth.bin";
                if(debug)
                {
                    std::cout << "Image " << rgb_image << " will be moved to " << used_image_rgb << std::endl;
                    std::cout << "Image " << depth_image << " will be moved to " << used_image_depth << std::endl;
                }
                rename ( rgb_image.c_str() , used_image_rgb.c_str() );
                rename ( depth_image.c_str() , used_image_depth.c_str() );
            }

            clock_t start_time = clock();

            //Find and crop the head from both the rgb image and the point cloud
            extractor.setInputImages(depth,image);
            cv::Mat cropped_depth,cropped_rgb;
            extractor.cropHead(cropped_depth,cropped_rgb);                

            if(debug)
            {
                cout << "Face extraction finished in " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;
                cout << "Writing cropped rgb and depth to file in " << output_path << endl;
                depth_out = output_path + prefix.str() + suffix.str() + "depth.png";
                face = output_path + prefix.str() + suffix.str() + "face.png";
                cv::imwrite( depth_out, cropped_depth * 10);
                cv::imwrite( face, cropped_rgb );
            }

            //Create the Face object from the cropped rgb and cropped depth
            Face face(cropped_rgb,cropped_depth);

            if(automatic_pose)
            {
#if PCL_VERSION_COMPARE(>=,1,8,0)
                //add all faces to the pose 0
                persons[i-starting_person][0].addFace(face,extractor.getHeadPose());
                //save all faces pose estimation
                poses.push_back(extractor.getHeadPose());
#endif
            }
            else
            {
                //Manual pose detection using name information
                if(number_of_poses==5)
                {
                    //pose left
                    if(j==1 || j==7)
                        persons[i-starting_person][1].addFace(face);
                    //pose right
                    else if(j==2 || j==8)
                        persons[i-starting_person][2].addFace(face);
                    //pose up
                    else if(j==3 || j==9)
                        persons[i-starting_person][3].addFace(face);
                    //pose down
                    else if(j==4 || j==10)
                        persons[i-starting_person][4].addFace(face);
                    //pose normal
                    else
                        persons[i-starting_person][0].addFace(face);
                }
                else
                {
                    persons[i-starting_person][0].addFace(face);
                }
            }
        }
        std::cout << "DONE IN " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << std::endl;
    }

    if(automatic_pose)
    {
        //------------------------------------------------Automatic pose estimation----------------------------------------
        if(debug)
            cout << "Starting meanK Algorithm to automatically estimate the poses of the heads" <<endl;

        clock_t start_time = clock();
#if PCL_VERSION_COMPARE(>=,1,8,0)
        //Poses are rappresented with a 3x3 rotation matrix
        pcl::Kmeans kmeans(poses.size(),9);
        kmeans.setClusterSize(number_of_poses);

        kmeans.setInputData(poses);

        kmeans.kMeans();

        if(debug)
            cout << "Kmeans finished in " << (float)(clock() - start_time)/CLOCKS_PER_MS << " ms." << endl;
        //Centroids are the center of the poses for the number_of_poses different poses

        pcl::Kmeans::Centroids centroids = kmeans.get_centroids();

        if(debug)
            cout<<"Number of clusters: "<<centroids.size()<<endl;

        //Now we need to test every image of every person against every centroid to find the nearest one
        //and assign this face of this person to that pose
        for(int i=0;i<number_of_persons;i++)
        {
            for(int j=0;j<image_per_person;j++)
            {
                float min_dist = 100000000000;
                int nearest_centroid = 0;
                for(int c=0;c<centroids.size();c++)
                {
                    float cur_dist = kmeans.distance(persons[i][0].getRotation(),centroids[c]);
                    if(cur_dist < min_dist)
                    {
                        min_dist = cur_dist;
                        nearest_centroid = c;
                    }
                }
                //Here we remove the face from the pose 0 and we add it to the pose "nearest_centroid"
                persons[i][nearest_centroid].addFace(persons[i][0].removeFace());
                persons[i][0].removeRotation();
            }
        }
#endif
    }

    //save the covariance matrixes to files
    for(int i=starting_person;i<number_of_persons+starting_person;i++)
    {
        //Strings for input file names
        std::string covarianceRgb,covarianceDepth;

        int tempi = i;
        if(tempi >= 15)
            tempi = i + 1;
        if(tempi >= 18)
            tempi = i + 2;
        if(tempi >= 21)
            tempi = i + 4;

        for(int j=0;j<number_of_poses;j++)
        {
            std::stringstream prefix;
            if(i<10) prefix << "00" << tempi << "_";
            else prefix << "0" << tempi << "_";
            covarianceRgb = train_path + prefix.str();
            covarianceDepth = train_path + prefix.str();
            std::stringstream suffix;
            if(j<10) suffix << "00" << j << "_";
            else suffix << "0" << j << "_";

            covarianceRgb += suffix.str() + "covarianceRgb.matrix";
            covarianceDepth += suffix.str() + "covarianceDepth.matrix";

            write_binary(covarianceRgb.c_str(),persons[i-starting_person][j].getCovarianceRgb());
            write_binary(covarianceDepth.c_str(),persons[i-starting_person][j].getCovarianceDepth());
        }
    }

}

void CovarianceComputer::loadFromFile()
{
    //load covariance matrixes from files
    for(int i=starting_person;i<number_of_persons+starting_person;i++)
    {
        //Strings for input file names
        std::string covarianceRgb,covarianceDepth;

        //Create for every person a vector of number_of_poses poses
        persons[i-starting_person] = std::vector<Pose>(number_of_poses);

        int tempi = i;
        if(tempi >= 15)
            tempi = i + 1;
        if(tempi >= 18)
            tempi = i + 2;
        if(tempi >= 21)
            tempi = i + 4;

        for(int j=0;j<number_of_poses;j++)
        {
            std::stringstream prefix;
            if(tempi<10) prefix << "00" << tempi << "_";
            else prefix << "0" << tempi << "_";
            covarianceRgb = train_path + prefix.str();
            covarianceDepth = train_path + prefix.str();
            std::stringstream suffix;
            if(j<10) suffix << "00" << j << "_";
            else suffix << "0" << j << "_";

            covarianceRgb += suffix.str() + "covarianceRgb.matrix";
            covarianceDepth += suffix.str() + "covarianceDepth.matrix";

            Eigen::MatrixXf covRgb,covDepth;
            read_binary(covarianceRgb.c_str(),covRgb);
            read_binary(covarianceDepth.c_str(),covDepth);

            persons[i-starting_person][j].setCovariance(covRgb,covDepth);
        }
    }
}

bool CovarianceComputer::loadDepthImageCompressed( std::string fname , cv::Mat& depthImg){
    //now read the depth image
    FILE* pFile = fopen(fname.c_str(), "rb");
    if(!pFile){
        cerr << "could not open file " << fname << endl;
        return false;
    }

    int im_width = 0;
    int im_height = 0;
    bool success = true;

    success &= ( fread(&im_width,sizeof(int),1,pFile) == 1 ); // read width of depthmap
    success &= ( fread(&im_height,sizeof(int),1,pFile) == 1 ); // read height of depthmap

    depthImg.create(im_height, im_width, CV_16SC1 );
    depthImg.setTo(0);


    int numempty;
    int numfull;
    int p = 0;

    if(!depthImg.isContinuous())
    {
        cerr << "Image has the wrong size! (should be 640x480)" << endl;
        return false;
    }

    int16_t* data = depthImg.ptr<int16_t>(0);
    while(p < im_width*im_height ){

        success &= ( fread( &numempty,sizeof(int),1,pFile) == 1 );

        for(int i = 0; i < numempty; i++)
            data[ p + i ] = 0;

        success &= ( fread( &numfull,sizeof(int), 1, pFile) == 1 );
        success &= ( fread( &data[ p + numempty ], sizeof(int16_t), numfull, pFile) == (unsigned int) numfull );
        p += numempty+numfull;

    }

    fclose(pFile);

    return true;
}

