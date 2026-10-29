#include "Utilities.h"
#include <string>

const int TABLE_ROWS = 683;
const int TABLE_COLS = 1434;
const int NUMBER_OF_IMAGES = 7;

const char* const FILE_LOCATION = "Media/Pool/";
const char* const RESULTS_LOCATION = "Media/Pool/Results/";
const char* const POOL_FILES[NUMBER_OF_IMAGES] = {
	"AllBalls.jpeg",
	"Table1.jpeg",
	"Table2.jpeg",
	"Table3.jpeg",
	"Table4.jpeg",
	"Table5.jpeg",
	"Table6.jpeg"
};

// table corners in the original image. Point2f is (x, y) = (column, row)
const Point2f IMAGE_CORNERS[4] = {
	Point2f(337.0f, 35.0f),		// top left
	Point2f(1570.0f, 21.0f),	// top right
	Point2f(1888.0f, 451.0f),	// bottom right
	Point2f(38.0f, 472.0f),		// bottom left 
};

// where those corners go in the rectified table image
const Point2f TABLE_CORNERS[4] = {
	Point2f(0.0f, 0.0f),					
	Point2f((float)(TABLE_COLS - 1), 0.0f),		
	Point2f((float)(TABLE_COLS - 1), (float)(TABLE_ROWS - 1)),	
	Point2f(0.0f, (float)(TABLE_ROWS - 1)),		
};

// rectify the table (perspective transformation)
Mat RectifyTable(const Mat& image, const Mat& perspective_matrix) {
	Mat table_image;
	warpPerspective(image, table_image, perspective_matrix, Size(TABLE_COLS, TABLE_ROWS), INTER_LINEAR);
	return table_image;
}

// table boundary used for the transformation
Mat DrawTableCorners(const Mat& image) {
	Mat display = image.clone();

	for (int corner = 0; corner < 4; corner++) {
		line(display, IMAGE_CORNERS[corner], IMAGE_CORNERS[(corner + 1) % 4], Scalar(0, 0, 255), 3);
		circle(display, IMAGE_CORNERS[corner], 8, Scalar(0, 255, 255), FILLED);
	}

	return display;
}

// save an image to the results folder
void SaveResult(const string& step_name, int image_no, const Mat& image) {
	string filename = string(RESULTS_LOCATION) + step_name + "_" + POOL_FILES[image_no];
	if (!imwrite(filename, image))
		cout << "Could not save " << filename << " (does the Results folder exist?)" << endl;
}

void MyApplication() {
	Mat perspective_matrix = getPerspectiveTransform(IMAGE_CORNERS, TABLE_CORNERS);
	cout << "Perspective transformation matrix:" << endl << perspective_matrix << endl << endl;

	namedWindow("Original", WINDOW_NORMAL);
	namedWindow("Rectified table", WINDOW_NORMAL);

	for (int image_no = 0; image_no < NUMBER_OF_IMAGES; image_no++)
	{
		string filename = string(FILE_LOCATION) + POOL_FILES[image_no];
		Mat image = imread(filename, IMREAD_COLOR);
		if (image.empty())
		{
			cout << "Could not open " << filename << endl;
			continue;
		}
		cout << POOL_FILES[image_no] << " - press any key for the next image" << endl;

		Mat corners_image = DrawTableCorners(image);
		Mat table_image = RectifyTable(image, perspective_matrix);

		imshow("Original", corners_image);
		imshow("Rectified table", table_image);
		SaveResult("Step1_Corners", image_no, corners_image);
		SaveResult("Step1_Rectified", image_no, table_image);

		waitKey(0);
	}
	cv::destroyAllWindows();
}