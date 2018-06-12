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

// struktura s katero shranjujem podatke o zaznanih kartah
typedef struct qcard_ {
    char* rank;
    char* suit;
    Mat rank_img;
    Mat suit_img;
    vector<Point> contour;
    Point center;
    int detected;

} qcard;

// struktura v kateri hranim testne slike
typedef struct train_image_ {
    char* name;
    Mat image;
} train_image;

// pot do testnih slik
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
int SUIT_DIFF_MAX = 1000;

int CARD_MAX_AREA = 200000;
int CARD_MIN_AREA = 50000;

int threshold_value = 170;
int threshold_type = THRESH_BINARY;

Scalar borderColor = Scalar(255, 255, 255);
Scalar borderOnlineColor = Scalar(0, 255, 0);
Scalar borderOfflineColor = Scalar(0, 0, 255);

#define middle '0'
#define player1 '1'
#define player2 '2'
#define player3 '3'
#define player4 '4'
#define player5 '5'
#define player6 '6'

// deklaracija tabele z imeni cifer/grbov
char** test_names;

// deklaracija testnih slik
vector<train_image> train_ranks;
vector<train_image> train_suits;
    

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


Rect areas[7];
vector<qcard> playingCards[7];
// dimezije sredniskih kart

//igralci
int igralci[7];
int scores[7];

void print_help() {
    printf("\nPress numbers 1-6 to capture cards of players 1-6\n\
Press number 0 to capture community cards\n\
Press q twice to quit\n\
Press h to write this message\n\n\
--------------------------------------------------\n\n");
}


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
    // pretvorim sliko v sivinsko in thresholdam

    Mat gray, blur, thresh;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, blur, Size(5,5), 0);

    #ifndef ADAPTIVE
        threshold(gray, thresh, threshold_value, 255, threshold_type);
    #else
        adaptiveThreshold(gray, thresh, 255, threshold_adaptive, threshold_type, block_size * 2 + 1, offset - 25);
    #endif
    
    return thresh;
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
    // vrne ploscato sliko karte dimenzije 200 x 300
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

    // izberem krajne tocke

    if (ys[temp[0]] > ys[temp[1]]) {
        topright = pts[temp[1]];
        bottomleft = pts[temp[0]];
    } else {
        topright = pts[temp[0]];
        bottomleft = pts[temp[1]];
    }


    if (w <= 0.87*h) {// ce je karta vertikalno orientirana
    
        rect[0] = topleft;
        rect[1] = topright;
        rect[2] = bottomright;
        rect[3] = bottomleft;
    }

    if (w >= 1.15*h) {// ce je karta horizontalno orientirana
    
        rect[0] = bottomleft;
        rect[1] = topleft;
        rect[2] = topright;
        rect[3] = bottomright;
    }

    if (w > 0.87*h && w < 1.15*h) {//If card is diamond oriented
        // če je najbolj leva tocka visje od najbolj desne -> karta najgnjena levo
        
        if (ys[argMin(xs)] <= ys[argMax(xs)]) {
            rect[0] = topright; // Top left
            rect[1] = topleft; // Top right
            rect[2] = bottomleft; // Bottom right
            rect[3] = bottomright; // Bottom left
        }
        // če je najbolj leva tocka nizje od najbolj desne -> karta najgnjena desno

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

    // z getperspectiveTransform izracuman matriko za transformacijo in jo warpam
    Mat warp;
    std::vector<cv::Point2f> dst = pts_;
    dst[0] = Point(0,0);
    dst[1] = Point(maxWidth-1, 0);
    dst[2] = Point(maxWidth-1,maxHeight-1);
    dst[3] = Point(0, maxHeight-1);

    Mat M = getPerspectiveTransform(rect,dst);
    warpPerspective(image, warp, M, Size(maxWidth, maxHeight));

    return warp;
}

vector<int> find_cards(Mat thresh_image, vector<vector<Point> > contours, vector<Vec4i> hierarchy) {
    // vrne vektor(enako dolg kot contours), ki za vsak contour pove ce je karta ali ne

    if (contours.size() == 0) {
        vector<int> empty(0, 0);
        return empty;

    }
    
    vector<int> cnt_is_card(contours.size(), 0);

    // kriteriji za karto:
    //1) Ploscina karte v mejah minimalne in maximalne podane meje
    //2) nima starsev,
    //4) ima 4 kote

    for (int i = 0; i < contours.size(); i++) {

        int size = contourArea(contours[i]);
        
        double perimeter = arcLength(contours[i],true);
        std::vector<cv::Point> approx;
        approxPolyDP(contours[i], approx, 0.01*perimeter,true);

        
        if ((size < CARD_MAX_AREA) && 
            (size > CARD_MIN_AREA) && 
            (hierarchy[i][3] == -1) && 
            (approx.size() == 4)) {
            //cout << size << "\n";
            cnt_is_card[i] = 1;
        }
    
    }

    return cnt_is_card;

}

qcard preprocess_card(Mat image, vector<Point> cardContour) {
    // shranimo podatke o karti (grb, cifra, slika) v struct qcard in ga vrnemo

    qcard queryCard;

    // zaznamo robne točke iz contoure
    std::vector<cv::Point> approx;
    double perimeter = arcLength(cardContour,true);
    approxPolyDP(cardContour, approx, 0.01*perimeter,true);

    int x, y, w, h;

    Rect boundRect = boundingRect(cardContour);

    x = boundRect.x;
    y = boundRect.y;
    h = boundRect.height;
    w = boundRect.width;

    queryCard.center.x = x + (int)(w/2);
    queryCard.center.y = y + (int)(h/2);

    
    // sliko zravnamo in preprocesirano (threshold)
    Mat card = cutflat_card(image, approx, h, w);


    int corner_width = 35;
    int corner_height = 70;
    
    //zajamemo kot karte, kjer se nahajata grb in cifra
    Mat corner_gray = card(Rect(0,0,corner_width, corner_height));
    
    // zoomiramo 4x
    Mat corner_zoom;
    resize(corner_gray, corner_zoom, Size(0,0), 4, 4);

    // za tem ko zoomiramo se znebimo majnih crnih pik ob robu ce obstajajo
    corner_zoom = do_morphology(MODE_CLOSING, corner_zoom);

    Mat corner_blur;
    GaussianBlur(corner_zoom, corner_blur, Size(5,5), 0);

    //thresholdamo da dobimo obratno barvno paleto
    threshold(corner_zoom, corner_zoom, threshold_value, 255, THRESH_BINARY_INV);

    Mat corner;
    corner_zoom.copyTo(corner);

    vector<vector<Point> > corner_contours;
    vector<Vec4i> corner_hierarchy;

    // poiscemo countoure v kotu
    findContours( corner_zoom, corner_contours, corner_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

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


    //Rect crect = Rect(cx,cy,RANK_WIDTH,RANK_HEIGHT + SUIT_HEIGHT);

    // izoliramo kot
    Mat figure = corner(Rect(cx,cy,RANK_WIDTH,RANK_HEIGHT + SUIT_HEIGHT));

    // posebej izrezemo cifro in grb
    Mat rank = figure(Rect(0,0,RANK_WIDTH,RANK_HEIGHT));
    Mat suit = figure(Rect(0,RANK_HEIGHT,SUIT_WIDTH,SUIT_HEIGHT));

    // shranimo vse podatke v qcard
    queryCard.rank = (char*)malloc(10*sizeof(char));
    queryCard.rank = (char*)"";

    queryCard.suit = (char*)malloc(10*sizeof(char));
    queryCard.suit = (char*)"";
    queryCard.rank_img = rank;
    queryCard.suit_img = suit;
    queryCard.contour = cardContour;
    queryCard.detected = 0;

    
    return queryCard;
}

int checkDetectedCard(qcard card) {
    // preveri ce je karta uspesno zaznana
    if (strcmp(card.rank, "Unknown") == 0 || strcmp(card.rank, "") == 0) {
        return 0;
    }
    if (strcmp(card.suit, "Unknown") == 0 || strcmp(card.suit, "") == 0) {
        return 0;
    }
    return 1;
}

int allDetected(vector<qcard> cards) {
    // preveri ce so vse karte uspesno zaznane
    for (int i = 0; i < cards.size(); i++) {
        if (cards[i].detected == 0) {
            return 0;
        }
    }
    return 1;
}

qcard match_card(qcard card, vector<train_image> train_ranks, vector<train_image> train_suits) {

    // poisce cifro in grb, ki se najbolje ujemata s pomocjo testnih slik

    int best_rank_match_diff = 10000;
    int best_suit_match_diff = 10000;
    char* best_rank_match_name = (char*)malloc(10*sizeof(char));
    best_rank_match_name = (char*)"Unknown";
    char* best_suit_match_name = (char*)malloc(10*sizeof(char));
    best_suit_match_name = (char*)"Unknown";
    char* best_rank_name = (char*)malloc(10*sizeof(char));
    best_rank_name = (char*)"Unknown";
    char* best_suit_name = (char*)malloc(10*sizeof(char));
    best_suit_name = (char*)"Unknown";

    // slika mora biti vecja od nic
    if ((card.rank_img.size() != Size(0,0)) && (card.suit_img.size() != Size(0,0))) {
        
        // izracunamo razliko med cifro in testno cifro, ter shranimo najboljse ujemanje
        for (int i = 0; i < train_ranks.size(); i++) {

            Mat diff_img;
            absdiff(card.rank_img, train_ranks[i].image, diff_img);
            int rank_diff = (int)(sum(diff_img)[0]/255);
            
            if (rank_diff < best_rank_match_diff) {

                best_rank_match_diff = rank_diff;
                best_rank_name = train_ranks[i].name;
            }
        }


        // izracunamo razliko med gtbom in testno grbom, ter shranimo najboljse ujemanje
        for (int i = 0; i < train_suits.size(); i++) {

            Mat diff_img;
            absdiff(card.suit_img, train_suits[i].image, diff_img);
            int suit_diff = (sum(diff_img)[0]/255);
            
            if (suit_diff < best_suit_match_diff) {

                best_suit_match_diff = suit_diff;
                best_suit_name = train_suits[i].name;
            }
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

    // preverimo ce je karta pravilno zaznana in jo oznacimo
    card.detected = checkDetectedCard(card);

    //cout << "match: " << card.rank << " of " << card.suit << "\n";

    return card;
}

Mat draw_results(Mat image, qcard card) {
    // narisemo rezultat karte na zaznano karto

    int x = card.center.x;
    int y = card.center.y;

    string rank_name = card.rank;
    string suit_name = card.suit;

    putText(image,(rank_name+" of"),Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1,(0,0,0),3,LINE_AA);
    putText(image,(rank_name+" of"),Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1,(50,200,200),2,LINE_AA);

    putText(image,suit_name,Point(x-60,y+25),FONT_HERSHEY_COMPLEX,1,(0,0,0),3,LINE_AA);
    putText(image,suit_name,Point(x-60,y+25),FONT_HERSHEY_COMPLEX,1,(50,200,200),2,LINE_AA);
    
    return image;
}

vector<train_image> load_train_ranks() {
    // nalozimo testne slike cifer

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
    //nalozimo testne slike grbov

    vector<train_image> suits;
    for (int i = 13; i < 17; i++) {
        train_image tmp;
        tmp.name = test_names[i];
        tmp.image = imread(img_path + test_names[i] + ".jpg",IMREAD_GRAYSCALE);
        suits.push_back(tmp);
    }

    return suits;
}


vector<qcard> getCards(Mat frame, Rect area) {

    // poiscemo contourje in iz njih locimo tiste ki so karte
    // vrnemo vektor vseh zaznanih qcardov

    Mat output, thresh;
    frame.copyTo(output);
    frame.copyTo(thresh);
    output = output(area);
    thresh = thresh(area);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // iz poiskanih contourjev poiscemo karte
    vector<int> cnt_is_card = find_cards(output, contours, hierarchy);
    
    vector<qcard> cards;
    // sprehodimo se cez conturje ki so karte int jih sprocesiramo
    for (int i = 0; i < contours.size(); i++) {
        if (cnt_is_card[i] == 1) {
            
            cards.push_back(preprocess_card(thresh, contours[i]));
        }
    }

    
    int max = 0;
    for (int i = 0; i < cards.size(); i++) {

        while(cards[i].detected == 0 && max < 10) {

            cards[i] = match_card(cards[i], train_ranks, train_suits);

            max++;
        }
    }

    return cards;
}

Mat printMiddle(Mat frame, vector<qcard> cards) {
    // napisemo karte iz sredine

    putText(frame, "Community",Point(3500,50),FONT_HERSHEY_COMPLEX,1.5,Scalar(0,0,0),3,LINE_AA);
    putText(frame, "Community",Point(3500,50),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);

    putText(frame, "cards:",Point(3500,90),FONT_HERSHEY_COMPLEX,1.5,Scalar(0,0,0),3,LINE_AA);
    putText(frame, "cards:",Point(3500,90),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);
    for (int i = 0; i < cards.size(); i++) {
        int x = 3600;
        int y = 150 + (i * 150);


        char* fullRank = (char*)malloc(150*sizeof(char)); strcpy(fullRank, "");
        strcat(fullRank, cards[i].rank); strcat(fullRank, " of");
        putText(frame,fullRank,Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1.5,Scalar(0,0,0),3,LINE_AA);
        putText(frame,fullRank,Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);

        putText(frame,cards[i].suit,Point(x-60,y+30),FONT_HERSHEY_COMPLEX,1.5,(0,0,0),3,LINE_AA);
        putText(frame,cards[i].suit,Point(x-60,y+30),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);
    }

    return frame;
}

Mat printCards(Mat frame, vector<qcard> cards, int ys, string playerNum) {
    // napisemo karte igralcev
    int x = 150;

    putText(frame, playerNum,Point(20,ys),FONT_HERSHEY_COMPLEX,2,Scalar(0,0,0),3,LINE_AA);
    putText(frame, playerNum,Point(20,ys),FONT_HERSHEY_COMPLEX,2,Scalar(255,255,255),2,LINE_AA);
    for (int i = 0; i < cards.size(); i++) {
        int y = ys - 20 + (i * 50);

        char* fullName = (char*)malloc(150*sizeof(char)); strcpy(fullName, "");
        strcat(fullName, cards[i].rank); strcat(fullName, " of "); strcat(fullName, cards[i].suit);
        putText(frame,(fullName),Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1.5,Scalar(0,0,0),3,LINE_AA);
        putText(frame,(fullName),Point(x-60,y-10),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);
        /*
        putText(frame,cards[i].suit,Point(x-60,y+25),FONT_HERSHEY_COMPLEX,1.5,(0,0,0),3,LINE_AA);
        putText(frame,cards[i].suit,Point(x-60,y+25),FONT_HERSHEY_COMPLEX,1.5,Scalar(255,255,255),2,LINE_AA);
        */
    }

    return frame;
}

Mat drawBorders(Mat frame) {
    // narise robove igralcev in sredine

    rectangle(frame, areas[0], borderColor, 3);
    if (igralci[1] == 1) {
        rectangle(frame, areas[1], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[1], borderOfflineColor, 3);
    }
    if (igralci[2] == 1) {
        rectangle(frame, areas[2], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[2], borderOfflineColor, 3);
    }
    if (igralci[3] == 1) {
        rectangle(frame, areas[3], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[3], borderOfflineColor, 3);
    }
    if (igralci[4] == 1) {
        rectangle(frame, areas[4], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[4], borderOfflineColor, 3);
    }
    if (igralci[5] == 1) {
        rectangle(frame, areas[5], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[5], borderOfflineColor, 3);
    }
    if (igralci[6] == 1) {
        rectangle(frame, areas[6], borderOnlineColor, 3);
    } else {
        rectangle(frame, areas[6], borderOfflineColor, 3);
    }

    return frame;
}


int main(int argc, char** argv) {

    char* path = (char*)malloc(64*sizeof(char));
    if (argc > 1) {

        path = argv[1];
    } else {

        cout << "Not enough arguments\n";
        return 1;
    }

    // shranimo si vsa imena kart
    test_names = (char**)malloc(17*sizeof(char*));

    test_names[0] = (char*)malloc(10*sizeof(char));     test_names[0] = (char*)"Ace"; // 0
    test_names[1] = (char*)malloc(10*sizeof(char));     test_names[1] = (char*)"Two"; // 1
    test_names[2] = (char*)malloc(10*sizeof(char));     test_names[2] = (char*)"Three"; // 2
    test_names[3] = (char*)malloc(10*sizeof(char));     test_names[3] = (char*)"Four"; // 3
    test_names[4] = (char*)malloc(10*sizeof(char));     test_names[4] = (char*)"Five"; // 4
    test_names[5] = (char*)malloc(10*sizeof(char));     test_names[5] = (char*)"Six"; // 5
    test_names[6] = (char*)malloc(10*sizeof(char));     test_names[6] = (char*)"Seven"; // 6
    test_names[7] = (char*)malloc(10*sizeof(char));     test_names[7] = (char*)"Eight"; // 7
    test_names[8] = (char*)malloc(10*sizeof(char));     test_names[8] = (char*)"Nine"; // 8
    test_names[9] = (char*)malloc(10*sizeof(char));     test_names[9] = (char*)"Ten"; // 9
    test_names[10] = (char*)malloc(10*sizeof(char));     test_names[10] = (char*)"Jack"; // 10
    test_names[11] = (char*)malloc(10*sizeof(char));     test_names[11] = (char*)"Queen"; // 11
    test_names[12] = (char*)malloc(10*sizeof(char));     test_names[12] = (char*)"King"; // 12
    test_names[13] = (char*)malloc(10*sizeof(char));     test_names[13] = (char*)"Spades"; // 13
    test_names[14] = (char*)malloc(10*sizeof(char));     test_names[14] = (char*)"Hearts"; // 14
    test_names[15] = (char*)malloc(10*sizeof(char));     test_names[15] = (char*)"Clubs"; // 15
    test_names[16] = (char*)malloc(10*sizeof(char));     test_names[16] = (char*)"Diamonds"; // 16

    
    VideoCapture camera(path);

    cout << "Framerate: " << camera.get(CV_CAP_PROP_FPS) << endl;

    // nalozimo testne slike
    train_ranks = load_train_ranks();
    train_suits = load_suit_ranks();

    //nastavimo dimeznije igralcev
    areas[0] = Rect(700, 700, 2200, 600);

    areas[1] = Rect(500, 50, 800, 600);
    areas[2] = Rect(1500, 50, 800, 600);
    areas[3] = Rect(2500, 50, 800, 600);
    areas[4] = Rect(500, 1350, 800, 600);
    areas[5] = Rect(1500, 1350, 800, 600);
    areas[6] = Rect(2500, 1350, 800, 600);



    Mat frame, thresh, drawing;

    // nastavimo stevilo igralcev
    int stIgralcev;
    for (int i = 0; i < 7; i++) {
        igralci[i] = 0;
        scores[i] = 0;
    }

    print_help();

    cout << "vnesite št. igralcev [1 - 6]: ";
    cin >> stIgralcev;

    for (int i =0; i < stIgralcev; i++) {
        int ix;
        cout << "vnesite št. igralca, ki bo igral [1 - 6]: ";
        cin >> ix;

        igralci[ix] = 1;
    }

    int s = 0;

    while (true) {

        camera.read(frame);

        int key;
        if ((key = waitKey(1)) >= 0) {}
        
        s = 0;
        while (key == middle && s < 30) {
            
            thresh = preprocess_image(frame);
            playingCards[0] = getCards(thresh, areas[0]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[0]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[1] == 1 && key == player1 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[1] = getCards(thresh, areas[1]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[1]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[2] == 1 && key == player2 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[2] = getCards(thresh, areas[2]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[2]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[3] == 1 && key == player3 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[3] = getCards(thresh, areas[3]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[3]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[4] == 1 && key == player4 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[4] = getCards(thresh, areas[4]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[4]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[5] == 1 && key == player5 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[5] = getCards(thresh, areas[5]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[5]) == 1) {
                break;
            }

            s++;
        }
        s = 0;
        while (igralci[6] == 1 && key == player6 && s < 30) {

            thresh = preprocess_image(frame);
            playingCards[6] = getCards(thresh, areas[6]);
            frame = drawBorders(frame);

            imshow(WINDOW_NAME, frame);
            camera.read(frame);

            if (allDetected(playingCards[6]) == 1) {
                break;
            }

            s++;
        }

        if (key == 'h' || key == 'H') {

            print_help();
        }

        if (key == 'q' || key == 'Q') {

            cout << "exiting...\n";
            break;
        }
        
        frame = drawBorders(frame);

        //narisemo zaznane karte
        frame = printMiddle(frame, playingCards[0]);
        frame = printCards(frame, playingCards[1], 550, "1:");
        frame = printCards(frame, playingCards[2], 700, "2:");
        frame = printCards(frame, playingCards[3], 850, "3:");
        frame = printCards(frame, playingCards[4], 1000, "4:");
        frame = printCards(frame, playingCards[5], 1150, "5:");
        frame = printCards(frame, playingCards[6], 1300, "6:");


        resize(frame, frame, Size(1920, 1080));
        imshow(WINDOW_NAME, frame);
            
    }
    camera.release();

    waitKey(0);
}