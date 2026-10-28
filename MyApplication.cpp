#include "Utilities.h"
#include <string>

const int TABLE_ROWS = 683;
const int TABLE_COLS = 1434;
const int NUMBER_OF_IMAGES = 7;

const char* const FILE_LOCATION = "Media/Pool/";
const char* const POOL_FILES[NUMBER_OF_IMAGES] = {
	"AllBalls.jpeg",
	"Table1.jpeg",
	"Table2.jpeg",
	"Table3.jpeg",
	"Table4.jpeg",
	"Table5.jpeg",
	"Table6.jpeg"
};

void MyApplication() {
	namedWindow("Original", WINDOW_NORMAL);

	for (int image_no = 0; image_no < NUMBER_OF_IMAGES; image_no++) {
		string filename = string(FILE_LOCATION) + POOL_FILES[image_no];
		Mat image = imread(filename, IMREAD_COLOR);
		if (image.empty()) {
			cout << "Could not open or find the image: " << filename << endl;
			continue;
		}

		cout << POOL_FILES[image_no] << " - press any key for the next image" << endl;

		imshow("Original", image);
		waitKey(0);
	}

	destroyAllWindows();
}