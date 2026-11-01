#include "Utilities.h"
#include <string>

const int TABLE_ROWS = 683;
const int TABLE_COLS = 1434;
const int NUMBER_OF_IMAGES = 7;
const int TRAINING_IMAGE_INDEX = 0;

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

const int MEDIAN_FILTER_SIZE = 5;

const Rect CLOTH_SAMPLE_REGION(50, 80, 600, 520);


const int HUE_BINS = 36;             
const int SATURATION_BINS = 32;      
const int HS_CHANNELS[] = { 0, 2 };
const int HS_BINS[] = { HUE_BINS, SATURATION_BINS };
const float HUE_RANGE[] = { 0.0f, 180.0f };
const float SATURATION_RANGE[] = { 0.0f, 256.0f };
const float* HS_RANGES[] = { HUE_RANGE, SATURATION_RANGE };

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

Mat SmoothImage(const Mat& table_image) {
	Mat smoothed_image;
	medianBlur(table_image, smoothed_image, MEDIAN_FILTER_SIZE);
	return smoothed_image;
}

// zoomed side-by-side of the same region (original | Gaussian | median)
Mat MakeSmoothingComparison(const Mat& table_image, const Mat& median_image) {
	Mat gaussian_image;
	GaussianBlur(table_image, gaussian_image, Size(MEDIAN_FILTER_SIZE, MEDIAN_FILTER_SIZE), 1.5);

	const Rect region(880, 220, 300, 150);
	const int zoom = 3;
	Mat zoomed[3];
	resize(table_image(region), zoomed[0], Size(), zoom, zoom, INTER_NEAREST);
	resize(gaussian_image(region), zoomed[1], Size(), zoom, zoom, INTER_NEAREST);
	resize(median_image(region), zoomed[2], Size(), zoom, zoom, INTER_NEAREST);

	Mat comparison;
	hconcat(zoomed, 3, comparison);
	return comparison;
}

// build the cloth model: 2D hue-saturation histogram of the cloth sample from the training image
MatND BuildClothModel(const Mat& smoothed_training_image) {
	Mat hls_image;
	cvtColor(smoothed_training_image, hls_image, COLOR_BGR2HLS);
	Mat cloth_sample = hls_image(CLOTH_SAMPLE_REGION);

	MatND cloth_histogram;
	calcHist(&cloth_sample, 1, HS_CHANNELS, Mat(), cloth_histogram, 2, HS_BINS, HS_RANGES);

	// normalise so the most common cloth colour = 255 (i.e. 1.0)
	normalize(cloth_histogram, cloth_histogram, 0.0, 255.0, NORM_MINMAX);
	return cloth_histogram;
}

// for the report: where the cloth samples were taken from
Mat DrawClothSampleRegion(const Mat& training_image) {
	Mat display = training_image.clone();
	rectangle(display, CLOTH_SAMPLE_REGION, Scalar(255, 0, 255), 3);
	return display;
}

// for the report: the 2D histogram as an image
Mat DrawClothHistogram(const MatND& cloth_histogram) {
	const int cell_size = 12;
	Mat display(HUE_BINS * cell_size, SATURATION_BINS * cell_size, CV_8UC1, Scalar(0));
	for (int hue_bin = 0; hue_bin < HUE_BINS; hue_bin++)
		for (int saturation_bin = 0; saturation_bin < SATURATION_BINS; saturation_bin++) {
			float value = cloth_histogram.at<float>(hue_bin, saturation_bin);
			Rect cell(saturation_bin * cell_size, hue_bin * cell_size, cell_size, cell_size);
			rectangle(display, cell, Scalar(value), FILLED);
		}
	return display;
}

// load one of the pool images
Mat LoadImage(int image_no) {
	string filename = string(FILE_LOCATION) + POOL_FILES[image_no];
	Mat image = imread(filename, IMREAD_COLOR);
	if (image.empty())
		cout << "Could not open " << filename << endl;
	return image;
}

// save an image to the results folder
void SaveResult(const string& step_name, int image_no, const Mat& image) {
	string image_name = POOL_FILES[image_no];
	image_name = image_name.substr(0, image_name.find('.'));      
	string filename = string(RESULTS_LOCATION) + step_name + "_" + image_name + ".png";
	if (!imwrite(filename, image))
		cout << "Could not save " << filename << " (does the Results folder exist?)" << endl;
}

void MyApplication() {
	Mat perspective_matrix = getPerspectiveTransform(IMAGE_CORNERS, TABLE_CORNERS);
	cout << "Perspective transformation matrix:" << endl << perspective_matrix << endl << endl;

	// ---- Training: build the cloth colour model from AllBalls.jpeg ONLY ----
	Mat training_image = LoadImage(TRAINING_IMAGE_INDEX);
	if (training_image.empty())
		return;
	Mat training_smoothed = SmoothImage(RectifyTable(training_image, perspective_matrix));
	MatND cloth_histogram = BuildClothModel(training_smoothed);

	Mat sample_display = DrawClothSampleRegion(training_smoothed);
	Mat histogram_display = DrawClothHistogram(cloth_histogram);
	SaveResult("Step3_ClothSample", TRAINING_IMAGE_INDEX, sample_display);
	SaveResult("Step3_ClothHistogram", TRAINING_IMAGE_INDEX, histogram_display);

	namedWindow("Cloth sample region", WINDOW_NORMAL);
	namedWindow("Cloth histogram (rows = hue, cols = saturation)", WINDOW_AUTOSIZE);
	imshow("Cloth sample region", sample_display);
	imshow("Cloth histogram (rows = hue, cols = saturation)", histogram_display);
	cout << "Cloth model built - press any key to process the images" << endl;
	waitKey(0);
	cv::destroyAllWindows();

	// ---- Process every image ----
	namedWindow("Rectified table", WINDOW_NORMAL);
	namedWindow("Smoothed", WINDOW_NORMAL);
	namedWindow("Zoom: original | Gaussian | median", WINDOW_NORMAL);

	for (int image_no = 0; image_no < NUMBER_OF_IMAGES; image_no++)
	{
		Mat image = LoadImage(image_no);
		if (image.empty())
			continue;
		cout << POOL_FILES[image_no] << " - press any key for the next image" << endl;

		// Step 1: rectify
		Mat table_image = RectifyTable(image, perspective_matrix);
		SaveResult("Step1_Corners", image_no, DrawTableCorners(image));
		SaveResult("Step1_Rectified", image_no, table_image);

		// Step 2: smooth
		Mat smoothed_image = SmoothImage(table_image);
		Mat comparison = MakeSmoothingComparison(table_image, smoothed_image);
		SaveResult("Step2_Smoothed", image_no, smoothed_image);
		SaveResult("Step2_Comparison", image_no, comparison);

		imshow("Rectified table", table_image);
		imshow("Smoothed", smoothed_image);
		imshow("Zoom: original | Gaussian | median", comparison);
		waitKey(0);
	}
	cv::destroyAllWindows();
}