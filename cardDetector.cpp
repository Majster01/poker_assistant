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

//#define WINDOW_NAME "Thresholded image"

typedef struct qcard_ {
    string rank;
    string suit;
    Mat rank_img;
    Mat suit_img;
    vector<Point> contour;

} qcard;

typedef struct train_image_ {
    string name;
    Mat image;
} train_image;

string img_path = "./train_files/";

int GET_SUIT = 0;
int GET_RANK = 1;

int BKG_THRESH = 60;
int CARD_THRESH = 30;


int RANK_WIDTH = 75;
int RANK_HEIGHT = 105;

int SUIT_WIDTH = 70;
int SUIT_HEIGHT = 95;

int RANK_DIFF_MAX = 2000;
int SUIT_DIFF_MAX = 700;

int CARD_MAX_AREA = 280000;
int CARD_MIN_AREA = 80000;

int threshold_value = 170;
int threshold_type = THRESH_BINARY;

//imena cifer/grbov
vector<string> test_names;
    

//morphology
#define MODE_OPENING 1
#define MODE_CLOSING 2

int structuring_element = MORPH_RECT;
int erosion_size = 2;
int dilation_size = 2;

Scalar color = (100,100,0);

#ifdef ADAPTIVE
int block_size = 15;
int threshold_adaptive = ADAPTIVE_THRESH_GAUSSIAN_C; //ADAPTIVE_THRESH_MEAN_C
int offset = 0;
#endif

typedef OutputArray OutputArrayOfArrays;
typedef OutputArray InputOutputArray;

Mat do_morphology(int mode, Mat src) {

    Mat tmp, dst;

    Mat erode_element = getStructuringElement(structuring_element, 
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
      	Point(erosion_size, erosion_size));

    Mat dilate_element = getStructuringElement(structuring_element, 
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
      	Point(dilation_size, dilation_size));

	switch (mode) {
	case MODE_OPENING:
		erode(src, tmp, erode_element);
		dilate(tmp, dst, dilate_element);
		break;
	case MODE_CLOSING:
		dilate(src, tmp, dilate_element);
		erode(tmp, dst, erode_element);
		break;
	}

    return dst;
}

Mat preprocess_image(Mat image) {
    //"""Returns a grayed and adaptively thresholded camera image."""

    Mat gray, blur, thresh;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, blur, Size(5,5), 0);


/*
    #ifndef ADAPTIVE
    createTrackbar("Threshold value", WINDOW_NAME, &threshold_value, 255);
    #else
    createTrackbar("Block size", WINDOW_NAME, &block_size, 40);
    createTrackbar("Offset", WINDOW_NAME, &offset, 50);
    #endif

    */

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
        //cout << pts[i].x << ", " << pts[i].y << " : " << res[i] << "\n";
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

Mat cutflat_card(Mat image, std::vector<cv::Point> pts, int h, int w) {
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


    //cout << topleft << " | " << topright  << "\n" << bottomleft << " | " << bottomright  << "\n";

    if (w <= 0.87*h) {// If card is vertically oriented
        //cout << 1 << "\n";
        rect[0] = topleft;
        rect[1] = topright;
        rect[2] = bottomright;
        rect[3] = bottomleft;
    }

    if (w >= 1.15*h) {// If card is horizontally oriented
        //cout << 2 << "\n";
        rect[0] = bottomleft;
        rect[1] = topleft;
        rect[2] = topright;
        rect[3] = bottomright;
    }

    if (w > 0.87*h && w < 1.15*h) {//If card is diamond oriented
        //cout << 3 << "\n";
        // If furthest left point is higher than furthest right point,
        // card is tilted to the left.
        if (ys[argMin(xs)] <= ys[argMax(xs)]) {
            rect[0] = topright; // Top left
            rect[1] = topleft; // Top right
            rect[2] = bottomleft; // Bottom right
            rect[3] = bottomright; // Bottom left
        }
        // If furthest left point is lower than furthest right point,
        // card is tilted to the right
        if (ys[argMin(xs)] > ys[argMax(xs)]) {
            rect[0] = topleft; // Top left
            rect[1] = bottomleft; // Top right
            rect[2] = bottomright; // Bottom right
            rect[3] = topright; // Bottom left
        }
    }


    int maxWidth = 200;
    int maxHeight = 300;

    // Create destination array, calculate perspective transform matrix,
    // and warp card image
    Mat warp;
    std::vector<cv::Point2f> dst = pts_;
    dst[0] = Point(0,0);
    dst[1] = Point(maxWidth-1, 0);
    dst[2] = Point(maxWidth-1,maxHeight-1);
    dst[3] = Point(0, maxHeight-1);

    //pointsum(dst);
    Mat M = getPerspectiveTransform(rect,dst);
    warpPerspective(image, warp, M, Size(maxWidth, maxHeight));

    imshow("da vidimo", warp);

    return warp;
}

vector<int> find_cards(Mat thresh_image, vector<vector<Point> > contours, vector<Vec4i> hierarchy) {
    //"""Finds all card-sized contours in a thresholded camera image.
    //Returns the number of cards, and a list of card contours sorted
    //from largest to smallest."""

    // If there are no contours, do nothing
    if (contours.size() == 0) {
        vector<int> empty(0, 0);
        return empty;

    }
    
    vector<int> cnt_is_card(contours.size(), 0);

    // Determine which of the contours are cards by applying the
    // following criteria: 1) Smaller area than the maximum card size,
    // 2), bigger area than the minimum card size, 3) have no parents,
    // and 4) have four corners

    for (int i = 0; i < contours.size(); i++) {

        int size = contourArea(contours[i]);
        double perimeter = arcLength(contours[i],true);
        std::vector<cv::Point> approx;
        approxPolyDP(contours[i], approx, 0.01*perimeter,true);

        
        if ((size < CARD_MAX_AREA) && (size > CARD_MIN_AREA)
            && (hierarchy[i][3] == -1) && (approx.size() == 4)) {
            //cout << size << "\n";
            cnt_is_card[i] = 1;
        }
    
    }

    return cnt_is_card;

}

qcard preprocess_card(Mat image, vector<Point> cardContour) {
    //Mat image = preprocess_image(frame);

    

/*

    Mat output;
    image.copyTo(output);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Find contours
    findContours( output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


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
    std::vector<cv::Point> cardContour = contours[contours.size()-1];
    //std::vector<cv::Point> smallestContour = contours[0];

    for (int i = 0; i < approx.size(); i++) {
    }

    */

    //Mat drawing = Mat::zeros( image.size(), CV_8UC3 );

    //cout << drawing << "\n";

    std::vector<cv::Point> approx;
    double perimeter = arcLength(cardContour,true);
    approxPolyDP(cardContour, approx, 0.01*perimeter,true);

    

    //cout << approx << "\n";

    int x, y, width, height;

    Rect boundRect = boundingRect(cardContour);
    //cout << boundRect << "\n";
    int h = boundRect.height;
    int w = boundRect.width;

    //rectangle(drawing, boundRect, Scalar(200, 200, 200));

    Mat card = cutflat_card(image, approx, h, w);


    int corner_width = 35;
    int corner_height = 70;
    // Grab corner of card image, zoom, and threshold
    Mat corner_gray = card(Rect(0,0,corner_width, corner_height));
    
    Mat corner_zoom;
    resize(corner_gray, corner_zoom, Size(0,0), 4, 4);

    // za tem ko zoomiramo se znebimo majnih crnih pik ob robu ce obstajajo
    corner_zoom = do_morphology(MODE_CLOSING, corner_zoom);

    Mat corner_blur;
    GaussianBlur(corner_zoom, corner_blur, Size(5,5), 0);

    threshold(corner_zoom, corner_zoom, threshold_value, 255, THRESH_BINARY_INV);

    Mat corner;
    corner_zoom.copyTo(corner);



    vector<vector<Point> > corner_contours;
    vector<Vec4i> corner_hierarchy;

    findContours( corner_zoom, corner_contours, corner_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


    //drawContours( corner_blur, corner_contours, 0, Scalar(200, 200, 200), 2, 8, corner_hierarchy, 0, Point() );

    //drawContours( corner_blur, corner_contours, 1, Scalar(200, 200, 200), 2, 8, corner_hierarchy, 0, Point() );

    // sestejemo vse tocke contourjev.. najmanjsi in najvecji so najblizji vogali
    int s = 1;
    vector<Point2f> corner_contour;
    for (int i = 0; i < corner_contours.size(); i++) {
        for (int j = 0; j < corner_contours[i].size(); j++) {
            corner_contour.push_back(corner_contours[i][j]);
        }
    }
    vector<float> corner_sums = pointsum(corner_contour);

    vector<float> corner_xs = extractPoints(corner_contour, 'x');
    vector<float> corner_ys = extractPoints(corner_contour, 'y');

    int cxi = argMin(corner_xs);
    int cyi = argMin(corner_ys);

    int cx = (int)corner_contour[cxi].x;
    int cy = (int)corner_contour[cyi].y;

    Rect crect = Rect(cx,cy,RANK_WIDTH,RANK_HEIGHT + SUIT_HEIGHT);

    Mat figure = corner(Rect(cx,cy,RANK_WIDTH,RANK_HEIGHT + SUIT_HEIGHT));

    //threshold(figure, figure, threshold_value, 255, THRESH_BINARY_INV);

    //cout << corner_contours.size() << "\n";
    //imshow("vogal blur", corner);
    //imshow("vogal blur", figure);

    //string fileName = test_names[number];
    //string path = img_path + fileName + ".img";

    //cout << path << "\n";

    
    qcard queryCard;

    //queryCard->rank = "Face down";
    //queryCard->suit = "Face down";

    Mat rank = figure(Rect(0,0,RANK_WIDTH,RANK_HEIGHT));
    Mat suit = figure(Rect(0,RANK_HEIGHT,SUIT_WIDTH,SUIT_HEIGHT));

    queryCard.rank_img = rank;
    queryCard.suit_img = suit;
    queryCard.contour = cardContour;


    //imwrite(img_path + fileName + ".jpg", train_img);
    //imshow("suit", rank);

    
    return queryCard;
    
    //return corner_gray;
}


qcard match_card(qcard card, vector<train_image> train_ranks, vector<train_image> train_suits) {
    //"""Finds best rank and suit matches for the query card. Differences
    //the query card rank and suit images with the train rank and suit images.
    //The best match is the rank or suit image that has the least difference."""

    int best_rank_match_diff = 10000;
    int best_suit_match_diff = 10000;
    string best_rank_match_name = "Unknown";
    string best_suit_match_name = "Unknown";
    string best_rank_name = "Unknown";
    string best_suit_name = "Unknown";
    int i = 0;

    // If no contours were found in query card in preprocess_card function,
    // the img size is zero, so skip the differencing process
    // (card will be left as Unknown)
    if ((card.rank_img.size() != Size(0,0)) && (card.suit_img.size() != Size(0,0))) {

        //imshow("card",card.rank_img);
        //imshow("train",train_ranks[0].image);
        //absdiff(card.rank_img, train_ranks[1].image, diff_img);

        //imshow("train",diff_img);

        
        // Difference the query card rank image from each of the train rank images,
        // and store the result with the least difference
        for (int i = 0; i < train_ranks.size(); i++) {
            cout << card.rank_img.size() << " : " << train_ranks[i].image.size() << "\n";

            Mat diff_img;
            absdiff(card.rank_img, train_ranks[i].image, diff_img);
            int rank_diff = (int)(sum(diff_img)[0]/255);


            cout << rank_diff << "\n";
            
            if (rank_diff < best_rank_match_diff) {

                //best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff;
                best_rank_name = train_ranks[i].name;
            }

            /*
            imshow("bla", diff_img);
            waitKey(0);
            */
        }


        // Same process with suit images
        for (int i = 0; i < train_suits.size(); i++) {
            cout << card.suit_img.size() << " : " << train_suits[i].image.size() << "\n";

            Mat diff_img;
            absdiff(card.suit_img, train_suits[i].image, diff_img);
            int suit_diff = (sum(diff_img)[0]/255);

            cout << suit_diff << "\n";
            
            if (suit_diff < best_suit_match_diff) {

                //best_rank_diff_img = diff_img
                best_suit_match_diff = suit_diff;
                best_suit_name = train_suits[i].name;
            }


            /*
            imshow("blah", diff_img);
            waitKey(0);
            */
        }
    }

    // Combine best rank match and best suit match to get query card's identity.
    // If the best matches have too high of a difference value, card identity
    // is still Unknown
    if (best_rank_match_diff < RANK_DIFF_MAX) {

        best_rank_match_name = best_rank_name;
        card.rank = best_rank_match_name;
    }

    if (best_suit_match_diff < SUIT_DIFF_MAX) {

        best_suit_match_name = best_suit_name;
        card.suit = best_suit_match_name;
    }

    cout << "diff: " << best_rank_match_diff << " | " << best_suit_match_diff << "\n";
    cout << "diff: " << best_rank_match_name << " | " << best_suit_match_name << "\n";

    // Return the identiy of the card and the quality of the suit and rank match
    //return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff

    return card;
}

vector<train_image> load_train_ranks() {

    vector<train_image> ranks;
    for (int i = 0; i < 13; i++) {
        train_image tmp;
        tmp.name = test_names[i];
        tmp.image = imread(img_path + test_names[i] + ".jpg",IMREAD_GRAYSCALE);
        ranks.push_back(tmp);
    }

    return ranks;
}

vector<train_image> load_suit_ranks() {

    vector<train_image> suits;
    for (int i = 13; i < 17; i++) {
        train_image tmp;
        tmp.name = test_names[i];
        tmp.image = imread(img_path + test_names[i] + ".jpg",IMREAD_GRAYSCALE);
        suits.push_back(tmp);
    }

    return suits;
}


int main(int argc, char** argv) {

    test_names.push_back("Ace"); // 0
    test_names.push_back("Two"); // 1
    test_names.push_back("Three"); // 2
    test_names.push_back("Four"); // 3
    test_names.push_back("Five"); // 4
    test_names.push_back("Six"); // 5
    test_names.push_back("Seven"); // 6
    test_names.push_back("Eight"); // 7
    test_names.push_back("Nine"); // 8
    test_names.push_back("Ten"); // 9
    test_names.push_back("Jack"); // 10
    test_names.push_back("Queen"); // 11
    test_names.push_back("King"); // 12
    test_names.push_back("Spades"); // 13
    test_names.push_back("Hearts"); // 14
    test_names.push_back("Clubs"); // 15
    test_names.push_back("Diamonds"); // 16

    //namedWindow(WINDOW_NAME, 4);
    //cout << CV_WINDOW_AUTOSIZE;

    //videostream = cv2.VideoCapture(0)
    /*
    VideoCapture camera("video.avi");

    //camera.set(CV_CAP_PROP_FPS, 10);

    cout << "Framerate: " << camera.get(CV_CAP_PROP_FPS) << endl;
   */


    // nalozimo testne slike
    vector<train_image> train_ranks = load_train_ranks();
    vector<train_image> train_suits = load_suit_ranks();

    //imshow("ace", train_ranks[0].image);
    //imshow("spade", train_suits[0].image);


    ////// ---- MAIN LOOP ---- //////

    Mat frame, thresh, drawing;

    // Begin capturing frames
    //while (true) {

        //camera.read(frame);
        frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
            //break;
        //train(image, 4);
        //preprocesiramo sliko
        thresh = preprocess_image(frame);


        // poiscemo contourje in iz njih locimo tiste ki so karte
        Mat output;
        thresh.copyTo(output);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        // iz poiskanih contourjev poiscemo karte
        vector<int> cnt_is_card = find_cards(output, contours, hierarchy);
        
        vector<qcard> cards;
        // sprehodimo se cez conturje ki so karte int jih sprocesiramo
        for (int i = 0; i < contours.size(); i++) {
            if (cnt_is_card[i] == 1) {
                //cout << contours[i] << "\n";
                cards.push_back(preprocess_card(thresh, contours[i]));

            }
        }

        for (int i = 0; i < cards.size(); i++) {
            cards[i] = match_card(cards[i], train_ranks, train_suits);
        }

        for (int i = 0; i < cards.size(); i++) {
            cout << "Card is " << cards[i].rank << " of " << cards[i].suit << "\n";
        }

        //imshow("rank", cards[0].rank_img);
        //imshow("karta", cards[0].suit_img);

        //drawing = getSuitRank(image, GET_RANK);

        /*
        */
        imshow("bla", thresh);
        
        
        //if (waitKey(30) >= 0)
        //    break;
            
    //}
        //camera.release();

    waitKey(0);
}

