// Map2CSV.cpp : Defines the entry point for the console application.

#include "stdafx.h"

const bool debug = false;

using namespace cv;
using namespace std;

int roundingAccuracy = 5;
double sensitivity = 1;
const double total = 100;

const double epsilon = 1e-6;

const double err = -5;
const double scaling = 3. / 4;

const string defaultImageName = "input.jpg";
const string defaultFileName = "output.csv";

enum ERR_CODES{
    FAILURE = -1,
    INPUT_IMAGE_NOT_LOADED = -2,
    OUTPUT_IMAGE_NOT_LOADED = -3,
    OUTPUT_FILE_NOT_LOADED = -4,
    SUM_NOT_TOTAL = -5,
    SENSITIVITY_FAILURE = -6,
    ROUNDING_ACCURACY_FAILURE = -7,
    SUCCESS = 0,
    INPUT_IMAGE_NAME,
    OUTPUT_CSV_NAME,
    SENSITIVITY,
    ROUNDING_ACCURACY,
    POINTS_EXISTS
};

struct Point2D{
    int x, y;
    Point2D(int x = 0, int y = 0) : x(x), y(y) {}
};

int width = 0; //image width
int height = 0; //image height

int decimalAccuracy = static_cast<int>(pow(10, roundingAccuracy));

static bool segDel(uchar* in){ delete in; return true; }

double toDecAccur(double n){
    return floor(n*decimalAccuracy + 0.5) / decimalAccuracy;
}

void getInput(int& i, string prompt){
    cout << prompt;
    while (!(cin >> i)){
        cout << "Enter a number. ";
        cout << prompt; 
        cin.clear();
        cin.ignore(INT_MAX, '\n');
        cin.clear();
    }
    cin.clear();
    cin.ignore(INT_MAX, '\n');
    cin.clear();
}

void getInput(double& i, string prompt){
    cout << prompt;
    while (!(cin >> i)){
        cout << "Enter a (decimal) number. ";
        cout << prompt;
        cin.clear();
        cin.ignore(INT_MAX, '\n');
        cin.clear();
    }
    cin.clear();
    cin.ignore(INT_MAX, '\n');
    cin.clear();
}

void readData(vector<uchar>& dataR, vector<uchar>& dataG, vector<uchar>& dataB, Mat& img){
    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            Vec3b pixel = img.at<Vec3b>(i, j);
            dataR[i*width + j] = pixel[2];
            dataG[i*width + j] = pixel[1];
            dataB[i*width + j] = pixel[0];
        }
    }
}

void calcStd(double& stdR, double& stdG, double& stdB, 
    const vector<uchar>& dataR, const vector<uchar>& dataG, const vector<uchar>& dataB, int& MN){
    if (err > -.5) {
        stdR = stdB = stdG = err;
        return;
    }

    double avgR(0), avgG(0), avgB(0);

    for (int i = 0; i < MN; ++i){
        avgR += dataR[i];
        avgB += dataB[i];
        avgG += dataG[i];
    }
    avgR /= MN; avgB /= MN; avgG /= MN;

    for (int i = 0; i < MN; ++i){
        double diff = dataR[i] - avgR;
        stdR += diff*diff;
        diff = dataG[i] - avgG;
        stdG += diff*diff;
        diff = dataB[i] - avgB;
        stdB += diff*diff;
    }
    stdR /= MN; stdB /= MN; stdG /= MN;
    stdR = sqrt(stdR); stdB = sqrt(stdB); stdG = sqrt(stdG);
    stdR *= sensitivity; stdB *= sensitivity; stdG *= sensitivity;
}

void helperFill(int& MN, const vector<uchar>& dataR, const vector<uchar>& dataG, const vector<uchar>& dataB, 
    vector<double>& prefs1, vector<double>& prefs2, double& stdR, double& stdG, double& stdB, 
    double& sum1, double& sum2, const int& a){

    uchar* data = new uchar[MN];
    vector<int> done;

    int count(0);

    for (int j = 0; j < MN; ++j){
        if (abs(dataR[j] - dataR[a]) < stdR && abs(dataG[j] - dataG[a]) < stdG &&
            abs(dataB[j] - dataB[a]) < stdB && abs(prefs1[j] + 1) < epsilon){
            data[j] = 255;
            done.push_back(j);
            ++count;
        }
        else{
            data[j] = 0;
        }
    }

    Mat chimg(height, width, CV_8UC1, data); //greyscale image

    if (chimg.empty()) throw OUTPUT_IMAGE_NOT_LOADED;
    if (width > GetSystemMetrics(SM_CXSCREEN) || height > GetSystemMetrics(SM_CYSCREEN)){
        int scaleW(width), scaleH(height);
        while (scaleW > GetSystemMetrics(SM_CXSCREEN) || scaleH > GetSystemMetrics(SM_CYSCREEN)){
            scaleW = static_cast<int>(scaleW * scaling);
            scaleH = static_cast<int>(scaleH * scaling);
        }
        resize(chimg, chimg, Size(scaleW, scaleH));
    }

    namedWindow("Segmented Image", CV_WINDOW_AUTOSIZE);
    //imwrite("output.jpg", chimg); //output?
    imshow("Segmented Image", chimg);
    waitKey();
    destroyWindow("Segmented Image");

    cout << endl;
    double pref1, pref2;
    getInput(pref1, "How much does party one (1) prefer the white areas? ");
    getInput(pref2, "How much does party two (2) prefer the white areas? ");
    if (debug) cout << "Prefs: " << pref1 << " " << pref2 << endl;
    sum1 += pref1;
    sum2 += pref2;
    pref1 /= count;
    pref2 /= count;
    if (debug) cout << "Prefs Divided: " << pref1 << " " << pref2 << endl;

    for (auto i = done.begin(); i != done.end(); ++i){
        prefs1[*i] = pref1;
        prefs2[*i] = pref2;
    }

    delete[] data;
}

void fillPrefs(int& MN, const vector<uchar>& dataR, const vector<uchar>& dataG, const vector<uchar>& dataB,
    vector<double>& prefs1, vector<double>& prefs2, double& stdR, double& stdG, double& stdB, const vector<int>& coords){
    double sum1(0), sum2(0);
    for (int i = 0; i < MN; ++i){
        prefs1[i] = -1;
        prefs2[i] = -1;
    }
    
    if (coords.empty()){
        cout << "Resorting to using no input points..." << endl;
        for (int i = 0; i < MN; ++i){
            if (abs(prefs1[i] + 1) > epsilon) continue;
            helperFill(MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, sum1, sum2, i);
        }
    }
    else{
        for (auto i = coords.begin(); i != coords.end(); ++i){
            if (abs(prefs1[*i] + 1) > epsilon) continue;
            helperFill(MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, sum1, sum2, *i);
        }
        for (int i = 0; i < MN; ++i){
            if (abs(prefs1[i] + 1) > epsilon) continue;
            prefs1[i] = 0; prefs2[i] = 0;
        }
    }
    if (debug) cout << "Sum 1: " << sum1 << " 2: " << sum2 << endl;
    if (abs(sum1 - total) > epsilon || abs(sum2 - total) > epsilon){
        cout << "Sum 1: " << sum1 << " 2: " << sum2 << endl;
        throw SUM_NOT_TOTAL;
    }
}

void mouseClick(int event, int x, int y, int flags, void* data)
{
    auto coords = static_cast<vector<int>*>(data);
    if (event == EVENT_LBUTTONDOWN) {
        coords->push_back(y*width + x);
        cout << "(" << x << ", " << y << ") added!" << endl;
    }
    else if (event == EVENT_RBUTTONDOWN) {
        coords->push_back(y*width + x);
        cout << "(" << x << ", " << y << ") added!" << endl;
    }
    else if (event == EVENT_MBUTTONDOWN) {
        coords->push_back(y*width + x);
        cout << "(" << x << ", " << y << ") added!" << endl;
    }
}

void fillPrefs(Mat& img, int& MN, const vector<uchar>& dataR, const vector<uchar>& dataG, const vector<uchar>& dataB,
    vector<double>& prefs1, vector<double>& prefs2, double& stdR, double& stdG, double& stdB, int& argc, char* argv[]){
    vector<int> coords;
    if (argc > POINTS_EXISTS){
        vector<int> readCoords;
        for (int i = POINTS_EXISTS; i < argc; ++i){
            istringstream iss(argv[i]);
            int coord;
            if (iss >> coord){
                readCoords.push_back(coord);
            }
            else{
                cout << "Only Numerical Coordinates." << endl;
                fillPrefs(MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, coords);
                return;
            }
        }
        if (readCoords.size() % 2 == 0){
            for (unsigned int i = 0; i < readCoords.size(); i += 2){
                if (readCoords[i] >= width || readCoords[i] < 0){
                    cout << "x coordinate out of bounds. Skipping Point..." << endl; continue;
                }
                if (readCoords[i + 1] >= height || readCoords[i + 1] < 0){
                    cout << "y coordinate out of bounds. Skipping Point..." << endl; continue;
                }
                coords.push_back(readCoords[i + 1] * width + readCoords[i]);
            }
        }
        else{
            cout << "Needs an Even Number of Coordinates." << endl;
            //fillPrefs(MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, coords);
            //return;
        }
    }
    else{
        string prompt = "Use Specific Points? Other colors will be assigned 0. \nType 'y' or 'i' for yes (image input), 't' for yes (text input) and 'n' for no. (y/n/i/t)? ";
        cout << prompt;
        string s;
        cin >> s;
        while (s != "y" && s != "n" && s!="i" && s!="t"){
            cout << prompt;
            cin >> s;
        }
        if (s == "t"){
            int x(0), y(0);
            cout << "\nEnter x and y coordinate for each point. Enter -1 to stop inputting points." << endl;
            while (true){
                getInput(x, "x coordinate? ");
                if (x < 0) break;
                else if (x >= width){
                    cout << "x coordinate too large! Try Again." << endl; continue;
                }
                getInput(y, "y coordinate? ");
                if (y < 0) break;
                else if (y >= height){
                    cout << "y coordinate too large! Try Again." << endl; continue;
                }
                coords.push_back(y*width + x);
            }
        }
        else if(s == "y" || s == "i"){ //y or i
            namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
            setMouseCallback("Input Image", mouseClick, &coords);
            imshow("Input Image", img);
            waitKey();
            destroyWindow("Input Image");
        }
    }
    fillPrefs(MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, coords);
}

void calcBlocksPerPixel(int& t, int& d, int& blocksPerPixel){
    if (d >= t){
        blocksPerPixel = 1;
    }
    else if (t % d == 0){
        blocksPerPixel = t / d;
    }
    else{
        blocksPerPixel = t / d + 1;
    }
}

void calcCSV(vector<double>& csv, const vector<double>& prefs, int& blocksPerPixelW, int& blocksPerPixelH, int& w, int& h){
    int count = 0;
    for (int i = 0; i < w; ++i){
        for (int j = 0; j < h; ++j){
            double pref = 0;
            for (int a = i*blocksPerPixelW; a < (i+1)*blocksPerPixelW && a < width; ++a){
                for (int b = j*blocksPerPixelH; b < (j+1)*blocksPerPixelH && b < height; ++b){
                    pref += prefs[a*height + b];
                }
            }
            csv[count++] = pref;
        }
    }
}

void writeCSV(int& argc, char* argv[], int& w, int& h, vector<double>& csv, string prepend){
    ofstream csvFile;
    if (argc > OUTPUT_CSV_NAME){
        csvFile.open(prepend + argv[OUTPUT_CSV_NAME]);
    }
    else{
        csvFile.open(prepend + defaultFileName);
    }
    if (!csvFile.is_open()){
        throw OUTPUT_FILE_NOT_LOADED;
    }
    int count = 0;
    for (int i = 0; i < w; ++i){
        for (int j = 0; j < h - 1; ++j){
            csvFile << toDecAccur(csv[count++]) << ",";
        }
        csvFile << toDecAccur(csv[count++]) << endl;
    }
    csvFile.close();
}

int start(int argc, char* argv[]){
    Mat img;
    if (argc > INPUT_IMAGE_NAME){
        img = imread(argv[INPUT_IMAGE_NAME], CV_LOAD_IMAGE_COLOR);
    }
    else{
        img = imread(defaultImageName);
    }
    width = img.cols;
    height = img.rows;
    int MN = height*width;
    if (debug) cout << "Image dimensions width: " << width << " height: " << height << " MN: " << MN << endl;

    if (img.empty()){
        throw INPUT_IMAGE_NOT_LOADED;
    }
    if (width > GetSystemMetrics(SM_CXSCREEN) || height > GetSystemMetrics(SM_CYSCREEN)){
        int scaleW(width), scaleH(height);
        while (scaleW > GetSystemMetrics(SM_CXSCREEN) || scaleH > GetSystemMetrics(SM_CYSCREEN)){
            scaleW = static_cast<int>(scaleW * scaling);
            scaleH = static_cast<int>(scaleH * scaling);
        }
        resize(img, img, Size(scaleW, scaleH));
        string outName = "scaled_";
        outName.append(argv[INPUT_IMAGE_NAME]);
        imwrite(outName, img);
        width = img.cols;
        height = img.rows;
        int MN = height*width;
        if (debug) cout << "Scaled image dimensions width: " << width << " height: " << height << " MN: " << MN << endl;
    }

    if (argc > SENSITIVITY){
        istringstream iss(argv[SENSITIVITY]);
        if (iss >> sensitivity && sensitivity > 0){
            cout << "Using custom sensitivity of " << sensitivity << " (default is 1)." << endl;
        }
        else throw SENSITIVITY_FAILURE;
    }

    if (argc > ROUNDING_ACCURACY){
        istringstream iss(argv[ROUNDING_ACCURACY]);
        if (iss >> roundingAccuracy && roundingAccuracy >= 0){
            decimalAccuracy = static_cast<int>(pow(10, roundingAccuracy));
            cout << "Using custom rounding accuracy of " << roundingAccuracy << " (default is 5)." << endl;
        }
        else throw ROUNDING_ACCURACY_FAILURE;
    }

    int w, h;
    getInput(w, "How many blocks horizontal? ");
    getInput(h, "How many blocks vertical? ");
    if(debug) cout << "Block dimensions width: " << w << " height: " << h << endl;

    vector<uchar> dataR(MN);
    vector<uchar> dataG(MN);
    vector<uchar> dataB(MN);
    readData(dataR, dataG, dataB, img);

    double stdR(0), stdG(0), stdB(0);
    calcStd(stdR, stdG, stdB, dataR, dataG, dataB, MN);
    if (debug) cout << "stdR: " << stdR << " stdG: " << stdG << " stdB: " << stdB << endl;

    vector<double> prefs1(MN);
    vector<double> prefs2(MN);
    fillPrefs(img, MN, dataR, dataG, dataB, prefs1, prefs2, stdR, stdG, stdB, argc, argv);

    int blocksPerPixelW(0), blocksPerPixelH(0);
    calcBlocksPerPixel(width, w, blocksPerPixelW);
    calcBlocksPerPixel(height, h, blocksPerPixelH);
    if (debug) cout << "Blocks Per Pixel W: " << blocksPerPixelW << " Blocks Per Pixel H: " << blocksPerPixelH <<endl;

    vector<double> csv1(w*h);
    vector<double> csv2(w*h);
    calcCSV(csv1, prefs1, blocksPerPixelW, blocksPerPixelH, w, h);
    calcCSV(csv2, prefs2, blocksPerPixelW, blocksPerPixelH, w, h);

    writeCSV(argc, argv, w, h, csv1, "ones_");
    writeCSV(argc, argv, w, h, csv2, "twos_");

    cout << "\nFinished Successfully." << endl;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    if (debug) cout << "starting..." << endl;
    //initialize here
    try{
        return start(argc, argv);
    }
    catch (ERR_CODES e){
        switch (e){
        case INPUT_IMAGE_NOT_LOADED:
            cout << "Input Image Not Found!" << endl;
            break;
        case OUTPUT_IMAGE_NOT_LOADED:
            cout << "Output Image Unable to be calculated!" << endl;
            break;
        case OUTPUT_FILE_NOT_LOADED:
            cout << "Output File Not Opened!" << endl;
            break;
        case SUM_NOT_TOTAL:
            cout << "Preferences MUST total to " << total << endl;
            break;
        case SENSITIVITY_FAILURE:
            cout << "Sensitivity must be a number greater than 0 (not equal to)." << endl;
            break;
        case ROUNDING_ACCURACY_FAILURE:
            cout << "Rouding Accuracy must be an integer greater than or equal to 0." << endl;
            break;
        }
        return e;
    }
    catch (...){
        cout << "Unknown Error Occurred :(" << endl;
        return FAILURE;
    }
}