#include "Utilities.h"
#include <string>
#include <cmath>

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

// Cloth sample = the whole table in the TRAINING image (AllBalls.jpeg), except:
//  - a border along the cushions, where the cushions and pockets are
//  - the area containing the balls, found from the AllBalls ground truth
//    (box around all 16 balls, expanded by 2 ball diameters to cover the stretched balls)
const int BALL_DIAMETER = 43;     // mm, from the brief
const int CUSHION_MARGIN = 60;    // mm
const int NUMBER_OF_BALLS = 16;
// AllBalls ground truth ball positions as (column, row), from the brief
const Point ALLBALLS_BALL_POSITIONS[NUMBER_OF_BALLS] = {
	Point(797, 293),  Point(908, 253),  Point(959, 250),  Point(1013, 255),
	Point(1072, 256), Point(1132, 254), Point(1184, 262), Point(1244, 256),
	Point(1307, 253), Point(901, 329),  Point(961, 330),  Point(1013, 335),
	Point(1067, 336), Point(1130, 334), Point(1187, 345), Point(1245, 342)
};


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

// which pixels of the training image are used as cloth samples (255 = sample, 0 = excluded)
Mat MakeClothSampleMask() {
	// start with the table interior, away from the cushions and pockets
	Mat mask(TABLE_ROWS, TABLE_COLS, CV_8UC1, Scalar(0));
	rectangle(mask, Rect(CUSHION_MARGIN, CUSHION_MARGIN,
		TABLE_COLS - 2 * CUSHION_MARGIN, TABLE_ROWS - 2 * CUSHION_MARGIN), Scalar(255), FILLED);

	// find the box containing all the balls (from the training ground truth)
	int min_col = TABLE_COLS, max_col = 0, min_row = TABLE_ROWS, max_row = 0;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		min_col = min(min_col, ALLBALLS_BALL_POSITIONS[ball].x);
		max_col = max(max_col, ALLBALLS_BALL_POSITIONS[ball].x);
		min_row = min(min_row, ALLBALLS_BALL_POSITIONS[ball].y);
		max_row = max(max_row, ALLBALLS_BALL_POSITIONS[ball].y);
	}

	// expand it by 2 ball diameters and remove it from the sample
	int expand = 2 * BALL_DIAMETER;
	rectangle(mask, Point(min_col - expand, min_row - expand),
		Point(max_col + expand, max_row + expand), Scalar(0), FILLED);
	return mask;
}

// smooth the 2D histogram: each bin becomes the average of itself and its 8 neighbours
// (2D version of the 1D histogram smoothing from the lectures).
// Hue is circular, so it wraps around at the ends; saturation does not wrap.
MatND SmoothHistogram(const MatND& histogram) {
	MatND smoothed = histogram.clone();
	for (int hue_bin = 0; hue_bin < HUE_BINS; hue_bin++)
		for (int saturation_bin = 0; saturation_bin < SATURATION_BINS; saturation_bin++) {
			float sum = 0.0f;
			int count = 0;
			for (int hue_offset = -1; hue_offset <= 1; hue_offset++)
				for (int saturation_offset = -1; saturation_offset <= 1; saturation_offset++) {
					int h = (hue_bin + hue_offset + HUE_BINS) % HUE_BINS;    // wrap around (circular hue)
					int s = saturation_bin + saturation_offset;
					if (s < 0 || s >= SATURATION_BINS)                       // no wrap for saturation
						continue;
					sum += histogram.at<float>(h, s);
					count++;
				}
			smoothed.at<float>(hue_bin, saturation_bin) = sum / count;
		}
	return smoothed;
}

// build the cloth model: 2D hue-saturation histogram of the cloth sample from the training image
MatND BuildClothModel(const Mat& smoothed_training_image, const Mat& sample_mask) {
	Mat hls_image;
	cvtColor(smoothed_training_image, hls_image, COLOR_BGR2HLS);

	MatND cloth_histogram;
	calcHist(&hls_image, 1, HS_CHANNELS, sample_mask, cloth_histogram, 2, HS_BINS, HS_RANGES);
	cloth_histogram = SmoothHistogram(cloth_histogram);

	// normalise so the most common cloth colour = 255 (i.e. 1.0)
	normalize(cloth_histogram, cloth_histogram, 0.0, 255.0, NORM_MINMAX);
	return cloth_histogram;
}

// for the report: sampled areas shown normally, excluded areas darkened
Mat DrawClothSampleRegion(const Mat& training_image, const Mat& sample_mask) {
	Mat display = training_image.clone();
	Mat darkened = training_image * 0.3;
	darkened.copyTo(display, sample_mask == 0);
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

// back-projection: each pixel gets the histogram value for its (hue, saturation)
// bright = looks like cloth, dark = does not (candidate ball)
Mat ComputeClothProbability(const Mat& smoothed_image, const MatND& cloth_histogram) {
	Mat hls_image;
	cvtColor(smoothed_image, hls_image, COLOR_BGR2HLS);
	Mat cloth_probability;
	calcBackProject(&hls_image, 1, HS_CHANNELS, cloth_histogram, cloth_probability, HS_RANGES, 1.0);
	return cloth_probability;
}

// threshold the cloth probability with Otsu's method -> binary mask
// INVERSE threshold: low cloth probability -> white (255) = candidate ball, cloth -> black (0)
Mat ThresholdClothProbability(const Mat& cloth_probability, double& otsu_threshold) {
	Mat ball_mask;
	otsu_threshold = threshold(cloth_probability, ball_mask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	return ball_mask;
}

// for the report: histogram of the cloth probability image (log scale so small counts are visible)
// with Otsu's threshold marked as a red line
Mat DrawProbabilityHistogram(const Mat& cloth_probability, double otsu_threshold) {
	int number_bins = 256;
	int channels[] = { 0 };
	float range[] = { 0.0f, 256.0f };
	const float* ranges[] = { range };
	MatND histogram;
	calcHist(&cloth_probability, 1, channels, Mat(), histogram, 1, &number_bins, ranges);

	const int height = 200;
	const int bar_width = 2;
	Mat display(height, number_bins * bar_width, CV_8UC3, Scalar(255, 255, 255));
	double max_count;
	minMaxLoc(histogram, nullptr, &max_count);
	double log_max = log(1.0 + max_count);
	for (int bin = 0; bin < number_bins; bin++) {
		double count = histogram.at<float>(bin);
		int bar_height = (int)(height * log(1.0 + count) / log_max);
		rectangle(display, Point(bin * bar_width, height - bar_height),
			Point(bin * bar_width + bar_width - 1, height - 1), Scalar(0, 0, 0), FILLED);
	}
	int threshold_x = (int)otsu_threshold * bar_width;
	line(display, Point(threshold_x, 0), Point(threshold_x, height - 1), Scalar(0, 0, 255), 2);
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
	Mat sample_mask = MakeClothSampleMask();
	MatND cloth_histogram = BuildClothModel(training_smoothed, sample_mask);

	Mat sample_display = DrawClothSampleRegion(training_smoothed, sample_mask);
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
	namedWindow("Cloth probability", WINDOW_NORMAL);
	namedWindow("Ball mask (white = not cloth)", WINDOW_NORMAL);
	namedWindow("Probability histogram (log scale, red = Otsu threshold)", WINDOW_AUTOSIZE);

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
		SaveResult("Step2_Smoothed", image_no, smoothed_image);
		SaveResult("Step2_Comparison", image_no, MakeSmoothingComparison(table_image, smoothed_image));

		// Step 3: cloth probability (back-projection of the cloth model)
		Mat cloth_probability = ComputeClothProbability(smoothed_image, cloth_histogram);
		SaveResult("Step3_ClothProbability", image_no, cloth_probability);

		// Step 4: threshold (Otsu) -> binary mask of candidate ball pixels
		double otsu_threshold;
		Mat ball_mask = ThresholdClothProbability(cloth_probability, otsu_threshold);
		Mat probability_histogram = DrawProbabilityHistogram(cloth_probability, otsu_threshold);
		cout << "   Otsu threshold = " << otsu_threshold << endl;
		SaveResult("Step4_Mask", image_no, ball_mask);
		SaveResult("Step4_Histogram", image_no, probability_histogram);

		imshow("Cloth probability", cloth_probability);
		imshow("Ball mask (white = not cloth)", ball_mask);
		imshow("Probability histogram (log scale, red = Otsu threshold)", probability_histogram);
		waitKey(0);
	}
	cv::destroyAllWindows();
}