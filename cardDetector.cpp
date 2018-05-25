// Python-OpenCV Playing Card Detector
//
// Author: Evan Juras
// Date: 9/5/17
// Description: Python script to detect and identify playing cards
// from a PiCamera video feed.
//

// Import necessary packages
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


using namespace std;
using namespace cv;

#define WINDOW_NAME "Thresholded image"

int BKG_THRESH = 60;
int CARD_THRESH = 30;

int threshold_value = 170;
int threshold_type = THRESH_BINARY;

Scalar color = (100,100,0);

#ifdef ADAPTIVE
int block_size = 15;
int threshold_adaptive = ADAPTIVE_THRESH_GAUSSIAN_C; //ADAPTIVE_THRESH_MEAN_C
int offset = 0;
#endif

typedef OutputArray OutputArrayOfArrays;
typedef OutputArray InputOutputArray;

Mat preprocess_image(Mat image) {
    //"""Returns a grayed and adaptively thresholded camera image."""

    Mat gray, blur, thresh;

    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5,5), 0);

    #ifndef ADAPTIVE
    createTrackbar("Threshold value", WINDOW_NAME, &threshold_value, 255);
    #else
    createTrackbar("Block size", WINDOW_NAME, &block_size, 40);
    createTrackbar("Offset", WINDOW_NAME, &offset, 50);
    #endif

    //threshold(gray, thresh, threshold_value, 255, threshold_type);
    #ifndef ADAPTIVE
        threshold(gray, thresh, threshold_value, 255, threshold_type);
    #else
        adaptiveThreshold(gray, thresh, 255, threshold_adaptive, threshold_type, block_size * 2 + 1, offset - 25);
    #endif
    
    return thresh;
}

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

int argMin(std::vector<float> x) { return std::distance(x.begin(), std::min_element(x.begin(), x.end())); }
int argMax(std::vector<float> x) { return std::distance(x.begin(), std::max_element(x.begin(), x.end())); }

std::vector<float> pointsum(std::vector<cv::Point2f> pts) {
    std::vector<float> res;
    res.resize(pts.size(), 0);
    
    for(int i = 0; i < pts.size(); i++) {
        //cout << pts[i] << "\n";
        res[i] = pts[i].x + pts[i].y;
        cout << pts[i].x << ", " << pts[i].y << " : " << res[i] << "\n";
    }

    return res;
}

std::vector<float> extractPoints(std::vector<cv::Point2f> pts, char s) {
    std::vector<float> res;
    res.resize(pts.size(), 0);
    
    for(int i = 0; i < pts.size(); i++) {
        if (s == 'x') {
            res[i] = pts[i].x;
        } else {
            res[i] = pts[i].y;
        }
    }

    return res;
}

void cutflat_card(Mat image, std::vector<cv::Point> pts, int h, int w) {
    // convert to Point2f
    std::vector<cv::Point2f> pts_;
    for (int i = 0; i < pts.size(); i++) {
        pts_.push_back(Point2f((float)pts[i].x, (float)pts[i].y));
    }

    std::vector<cv::Point2f> rect = pts_;

    std::vector<float> sums = pointsum(pts_);

    float tl = argMin(sums);
    float br = argMax(sums);
    cv::Point2f topleft = pts[tl];
    cv::Point2f bottomright = pts[br];
    cv::Point2f topright;
    cv::Point2f bottomleft;

    std::vector<float> xs = extractPoints(pts_, 'x');
    std::vector<float> ys = extractPoints(pts_, 'y');

    std::vector<int> temp;
    int size = 0;
    for (int i = 0; i < sums.size(); i++) {
        if (i != tl && i != br) {
            size++;
            temp.resize(size, i);
        }
    }
    if (ys[temp[0]] > ys[temp[1]]) {
        topright = pts[temp[1]];
        bottomleft = pts[temp[0]];
    } else {
        topright = pts[temp[0]];
        bottomleft = pts[temp[1]];
    }


    cout << topleft << " | " << topright  << "\n" << bottomleft << " | " << bottomright  << "\n";

    if (w <= 0.8*h) {// If card is vertically oriented
        cout << 1 << "\n";
        rect[0] = topleft;
        rect[1] = topright;
        rect[2] = bottomright;
        rect[3] = bottomleft;
    }

    if (w >= 1.2*h) {// If card is horizontally oriented
        cout << 2 << "\n";
        rect[0] = bottomleft;
        rect[1] = topleft;
        rect[2] = topright;
        rect[3] = bottomright;
    }

    if (w > 0.8*h && w < 1.2*h) {//If card is diamond oriented
        cout << 3 << "\n";
        // If furthest left point is higher than furthest right point,
        // card is tilted to the left.
        if (ys[argMin(xs)] <= ys[argMax(xs)]) {
            // If card is titled to the left, approxPolyDP returns points
            // in this order: top right, top left, bottom left, bottom right
            rect[0] = topright; // Top left
            rect[1] = topleft; // Top right
            rect[2] = bottomleft; // Bottom right
            rect[3] = bottomright; // Bottom left
        }
        // If furthest left point is lower than furthest right point,
        // card is tilted to the right
        if (ys[argMin(xs)] > ys[argMax(xs)]) {
            // If card is titled to the right, approxPolyDP returns points
            // in this order: top left, bottom left, bottom right, top right
            rect[0] = topleft; // Top left
            rect[1] = bottomleft; // Top right
            rect[2] = bottomright; // Bottom right
            rect[3] = topright; // Bottom left
        }
    }

    for (int i = 0; i < rect.size(); i++) {
        cout << rect[i] << "\n";
    }

    int maxWidth = 200;
    int maxHeight = 300;

    // Create destination array, calculate perspective transform matrix,
    // and warp card image
    Mat warp;
    //std::vector<cv::Point> dst = [ Point(0,0),Point(0,maxWidth-1),Point(maxWidth-1,maxHeight-1), Point(0, maxHeight-1)];
    /*std::vector<cv::Point> dst;
    dst.push_back(Point(0,0));
    dst.push_back(Point(0,maxWidth-1));
    dst.push_back(Point(maxWidth-1,maxHeight-1));
    dst.push_back(Point(0, maxHeight-1));*/
    std::vector<cv::Point2f> dst = pts_;
    dst[0] = Point(0,0);
    dst[1] = Point(maxWidth-1, 0);
    dst[2] = Point(maxWidth-1,maxHeight-1);
    dst[3] = Point(0, maxHeight-1);

    //pointsum(dst);
    Mat M = getPerspectiveTransform(rect,dst);
    warpPerspective(image, warp, M, Size(maxWidth, maxHeight));
    //cvtColor(warp, warp, COLOR_BGR2GRAY);

    imshow("da vidimo", warp);
}

Mat train(Mat frame) {
    //Mat frame = preprocess_image(image);
    Mat image = preprocess_image(frame);

    Mat output;
    image.copyTo(output);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    //Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


    //sort(contours, contours, (CV_SORT_EVERY_ROW | CV_SORT_ASCENDING));

    /// Draw contours
    Mat drawing = Mat::zeros( output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
       //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       //drawContours( drawing, contours, i, Scalar(120, 100, 100), 2, 8, hierarchy, 0, Point() );
    }

    // sort contours
    std::sort(contours.begin(), contours.end(), compareContourAreas);

    // grab contours
    drawContours( drawing, contours, contours.size()-1, Scalar(200, 200, 200), 2, 8, hierarchy, 0, Point() );
    std::vector<cv::Point> card = contours[contours.size()-1];
    std::vector<cv::Point> approx;
    //std::vector<cv::Point> smallestContour = contours[0];

    for (int i = 0; i < approx.size(); i++) {
    }

    double perimeter = arcLength(card,true);
    approxPolyDP(card, approx, 0.01*perimeter,true);

   // cout << approx << "\n";

    int x, y, width, height;

    Rect boundRect = boundingRect(card);
    int h = boundRect.height;
    int w = boundRect.width;

    cout << h << "\n" << w << "\n";

    rectangle(drawing, boundRect, Scalar(200, 200, 200));

    cutflat_card(image, approx, h, w);


    return drawing;
}

// ---- INITIALIZATION ---- //////
// Define constants and initialize variables

//// Camera settings
int IM_WIDTH = 1280;
int IM_HEIGHT = 720;
int FRAME_RATE = 10;

int main(int argc, char** argv) {


    //font = cv2.FONT_HERSHEY_SIMPLEX

    namedWindow(WINDOW_NAME, 2);
    //cout << CV_WINDOW_AUTOSIZE;

    //videostream = cv2.VideoCapture(0)
    /*
    VideoCapture camera(0);

    camera.set(CV_CAP_PROP_FPS, 10);

    cout << "Framerate: " << camera.get(CV_CAP_PROP_FPS) << endl;
    */

    //time.sleep(1) // Give the camera time to warm up

    // Load the train rank and suit images
    //path = os.path.dirname(os.path.abspath(__file__))
    //train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
    //train_suits = Cards.load_suits( path + '/Card_Imgs/')


    ////// ---- MAIN LOOP ---- //////
    // The main loop repeatedly grabs frames from the video stream
    // and processes them to find and identify playing cards.

    int cam_quit = 0; // Loop control variable

    Mat image, tresh;

    

    // Begin capturing frames
    //while (cam_quit == 0) {

        // Grab frame from video stream
        //camera.read(image);
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

        // Pre-process camera image (gray, blur, and threshold it)
        //----- pre_proc = Cards.preprocess_image(image)
        Mat drawing;
        drawing = train(image);
        //tresh = preprocess_image(image);
        
        imshow(WINDOW_NAME, drawing);
        
        // Poll the keyboard. If 'q' is pressed, exit the main loop.
        //int key = waitKey(1);
        //if (key >= 0)
            //break;
            
    //}

    waitKey(0);


    // Close all windows and close the PiCamera video stream.
    //destroyAllWindows();

}

