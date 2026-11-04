#include "Utilities.h"
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>

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
const int MORPHOLOGY_ELEMENT_SIZE = 5;

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

// Region classification (Step 7)
const double MIN_AREA_FRACTION = 0.5;                  // a ball must be at least half visible to be found
const double MIN_REGION_WIDTH = BALL_DIAMETER / 2.0;   // even half a ball is at least a ball radius wide (rejects thin slivers)

// Splitting (Step 8): erosion with a 3x3 element removes 1mm from each edge, so a piece that survives
// d erosions is about 2d mm thick. A ball core must be at least MIN_REGION_WIDTH thick (as in Step 7).
const int MIN_CORE_DEPTH = (int)(MIN_REGION_WIDTH / 2.0);

// =====================================================================
//  Ground truth (Step 10), from the Excel file provided with the brief.
//  [image][ball] = {row, column}; ball 0 = white, 1-15 = numbered; {-1,-1} = not on the table.
//  Corrections (only entries that are provably wrong, documented in the report):
//   - Table1 balls 2, 4, 11: row > 683 is impossible (table has 683 rows) -> row and column swapped
//   - Table1 ball 3: identical to ball 7 in the Excel file -> value from the table in the brief
//   - AllBalls balls 11-15: labels did not match the rack layout (each stripe sits directly below
//     the solid of the same colour) -> labels from the brief (positions unchanged)
//   - Table2 balls 4 and 5: labels contradicted the ball colours (4 = purple, 5 = orange) -> swapped
// =====================================================================
const int GROUND_TRUTH[NUMBER_OF_IMAGES][NUMBER_OF_BALLS][2] = {
	// AllBalls.jpeg (training image)
	{ { 293,797 },{ 253,908 },{ 250,959 },{ 255,1013 },{ 256,1072 },{ 254,1132 },{ 262,1184 },{ 256,1244 },
	  { 253,1307 },{ 329,901 },{ 330,961 },{ 335,1013 },{ 336,1067 },{ 334,1130 },{ 345,1187 },{ 342,1245 } },
	// Table1.jpeg
	{ {288,29}, {309,1092}, {429,1212}, {383,1131}, {335,1050}, {285,1211}, {261,1170}, {352,1173},
	  {331,1132}, {240,1210}, {310,1170}, {407,1169}, {281,1135}, {374,1212}, {360,1091}, {331,1213} },
	// Table2.jpeg
	{ { 507,1169 },{ 282,1103 },{ 620,1224 },{ 391,1121 },{ 315,1046 },{ 310,1305 },{ 202,621 },{ 361,1173 },
	  {331,1132}, {177,725}, {185,1269}, {665,798}, {296,438}, {53,1235}, {362,1090}, {399,1205} },
	// Table3.jpeg
	{ {434,29}, {23,908}, {616,1224}, {447,1199}, {336,29}, {305,1406}, {210,621}, {260,1294},
	  {226,1252}, {177,725}, {171,1270}, {665,798}, {249,351}, {53,1233}, {169,901}, {399,1201} },
	// Table4.jpeg
	{ {88,1392}, {23,908}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {210,621}, {260,1294},
	  {226,1252}, {181,724}, {171,1270}, {665,798}, {249,351}, {58,1231}, {169,901}, {399,1201} },
	// Table5.jpeg
	{ {608,26}, {535,23}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294},
	  {226,1252}, {538,72}, {171,1270}, {665,798}, {249,351}, {-1,-1}, {169,901}, {399,1201} },
	// Table6.jpeg
	{ {629,947}, {535,23}, {477,1364}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294},
	  {226,1252}, {538,72}, {171,1270}, {-1,-1}, {249,351}, {-1,-1}, {169,901}, {399,1201} }
};

// a detection is correct if it lies on the true ball, i.e. within one ball radius of its true position
const double MATCH_DISTANCE = BALL_DIAMETER / 2.0;

// Part 2: a ball's pixels reach up to about one ball diameter from its position in the rectified image
// (balls are stretched: the learned single-ball area is about 1.9 times a flat 43mm circle)
const int SAMPLE_RADIUS = BALL_DIAMETER;

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

// box around all the balls in the training image (from the AllBalls ground truth),
// expanded by 2 ball diameters to cover the stretched balls
Rect GetTrainingBallArea() {
	int min_col = TABLE_COLS, max_col = 0, min_row = TABLE_ROWS, max_row = 0;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		min_col = min(min_col, ALLBALLS_BALL_POSITIONS[ball].x);
		max_col = max(max_col, ALLBALLS_BALL_POSITIONS[ball].x);
		min_row = min(min_row, ALLBALLS_BALL_POSITIONS[ball].y);
		max_row = max(max_row, ALLBALLS_BALL_POSITIONS[ball].y);
	}
	int expand = 2 * BALL_DIAMETER;
	return Rect(Point(min_col - expand, min_row - expand), Point(max_col + expand, max_row + expand));
}

// which pixels of the training image are used as cloth samples (255 = sample, 0 = excluded)
Mat MakeClothSampleMask() {
	// start with the table interior, away from the cushions and pockets
	Mat mask(TABLE_ROWS, TABLE_COLS, CV_8UC1, Scalar(0));
	rectangle(mask, Rect(CUSHION_MARGIN, CUSHION_MARGIN,
		TABLE_COLS - 2 * CUSHION_MARGIN, TABLE_ROWS - 2 * CUSHION_MARGIN), Scalar(255), FILLED);
	// remove the area containing the balls
	rectangle(mask, GetTrainingBallArea(), Scalar(0), FILLED);
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

// threshold at zero: a pixel is "not cloth" only if its (hue, saturation) has NO support
// in the smoothed cloth model, i.e. that colour never appeared in (or next to) the cloth sample
Mat ZeroSupportMask(const Mat& cloth_probability) {
	Mat ball_mask;
	threshold(cloth_probability, ball_mask, 0, 255, THRESH_BINARY_INV);   // > 0 -> 0 (cloth), 0 -> 255 (candidate ball)
	return ball_mask;
}

// clean the binary mask with morphology:
// opening (erosion then dilation) removes small specks and thin bridges,
// then closing (dilation then erosion) fills small holes and gaps inside the balls
Mat CleanMask(const Mat& ball_mask) {
	Mat element(MORPHOLOGY_ELEMENT_SIZE, MORPHOLOGY_ELEMENT_SIZE, CV_8U, Scalar(1));
	Mat opened_mask, cleaned_mask;
	morphologyEx(ball_mask, opened_mask, MORPH_OPEN, element);
	morphologyEx(opened_mask, cleaned_mask, MORPH_CLOSE, element);
	return cleaned_mask;
}

// for the report: what the cleaning changed
// white = kept, red = removed by cleaning, green = added by cleaning
Mat DrawMorphologyChanges(const Mat& before, const Mat& after) {
	Mat display(before.size(), CV_8UC3, Scalar(0, 0, 0));
	display.setTo(Scalar(255, 255, 255), before & after);
	display.setTo(Scalar(0, 0, 255), before & ~after);
	display.setTo(Scalar(0, 255, 0), ~before & after);
	return display;
}

// table structure (cushions, pockets, marks on the cloth) = everything that is "not cloth" in the
// cleaned mask of the TRAINING image, apart from its balls. The camera is stationary, so these are
// in the same place in every image. Dilated by the 5x5 element to allow for small lighting differences.
Mat MakeTableStructureMask(const Mat& training_cleaned_mask) {
	Mat structure_mask = training_cleaned_mask.clone();
	rectangle(structure_mask, GetTrainingBallArea(), Scalar(0), FILLED);
	Mat element(MORPHOLOGY_ELEMENT_SIZE, MORPHOLOGY_ELEMENT_SIZE, CV_8U, Scalar(1));
	dilate(structure_mask, structure_mask, element);
	return structure_mask;
}

// object mask = cleaned mask AND NOT table structure,
// then an opening (same 5x5 element) removes any thin leftover slivers of the cushion band
Mat RemoveTableStructure(const Mat& cleaned_mask, const Mat& structure_mask) {
	Mat object_mask = cleaned_mask & ~structure_mask;
	Mat element(MORPHOLOGY_ELEMENT_SIZE, MORPHOLOGY_ELEMENT_SIZE, CV_8U, Scalar(1));
	morphologyEx(object_mask, object_mask, MORPH_OPEN, element);
	return object_mask;
}

// connected components: the outer boundary (contour) of every separate white region.
// OpenCV traces object boundaries with 8-adjacency (background 4-adjacency, as in the lectures).
// Only the outermost contours are kept (holes inside regions are not needed).
vector<vector<Point>> FindRegions(const Mat& object_mask) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(object_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	return contours;
}

// for the report: each region filled with a random colour (fixed seed, so colours repeat each run)
Mat DrawRegions(const Mat& table_image, const vector<vector<Point>>& regions) {
	Mat display = Mat::zeros(table_image.size(), CV_8UC3);
	RNG random_colours(12345);
	for (int region = 0; region < (int)regions.size(); region++) {
		Scalar colour(random_colours.uniform(60, 256), random_colours.uniform(60, 256), random_colours.uniform(60, 256));
		drawContours(display, regions, region, colour, FILLED, 8);
	}
	return display;
}

// learn the area of ONE ball's region from the TRAINING image:
// regions that contain exactly one ground-truth ball position are single balls;
// the median of their areas is used (robust to one unusual region)
double LearnSingleBallArea(const vector<vector<Point>>& training_regions) {
	vector<double> single_areas;
	for (int region = 0; region < (int)training_regions.size(); region++) {
		int balls_inside = 0;
		for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
			if (pointPolygonTest(training_regions[region], ALLBALLS_BALL_POSITIONS[ball], false) >= 0)
				balls_inside++;
		if (balls_inside == 1)
			single_areas.push_back(contourArea(training_regions[region]));
	}
	if (single_areas.empty())
		return 0.0;

	sort(single_areas.begin(), single_areas.end());
	double median_area = single_areas[single_areas.size() / 2];
	cout << "Single-ball regions in training image: " << single_areas.size() << ", areas:";
	for (int i = 0; i < (int)single_areas.size(); i++)
		cout << " " << (int)single_areas[i];
	cout << endl << "Learned single ball area (median) = " << median_area << endl << endl;
	return median_area;
}

// classify a region using its features (area, minimum bounding rectangle width)
// returns 0 = noise (rejected), 1 = single ball, 2+ = estimated number of merged balls
int EstimateBallCount(const vector<Point>& region, double single_ball_area) {
	double area = contourArea(region);
	RotatedRect bounding_rectangle = minAreaRect(region);
	double width = min(bounding_rectangle.size.width, bounding_rectangle.size.height);

	if (area < MIN_AREA_FRACTION * single_ball_area || width < MIN_REGION_WIDTH)
		return 0;
	int ball_count = (int)round(area / single_ball_area);
	return max(ball_count, 1);
}

// for the report: red = rejected, green = single ball, yellow = merged balls (labelled with the estimated count)
Mat DrawRegionClassification(const Mat& table_image, const vector<vector<Point>>& regions, const vector<int>& ball_counts) {
	Mat display = table_image * 0.5;
	for (int region = 0; region < (int)regions.size(); region++) {
		Scalar colour = (ball_counts[region] == 0) ? Scalar(0, 0, 255) :
			(ball_counts[region] == 1) ? Scalar(0, 255, 0) : Scalar(0, 255, 255);
		drawContours(display, regions, region, colour, FILLED, 8);
		if (ball_counts[region] >= 2) {
			Rect box = boundingRect(regions[region]);
			putText(display, to_string(ball_counts[region]), Point(box.x, box.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
		}
	}
	return display;
}

// a ball "core" found by repeated erosion: its centre, and how many erosions it survived
struct BallCore {
	Point2f centre;
	int depth;
};

// split a merged region by repeated erosion (3x3 = 1mm per erosion).
// Whenever a piece is about to disappear completely, it is recorded as a core:
// its centre (from moments) and its depth (number of erosions survived).
// erosion_depth accumulates, for every pixel, how many erosions it survived (for the report).
vector<BallCore> FindBallCores(const vector<Point>& region, const Size& image_size, Mat& erosion_depth) {
	Mat current = Mat::zeros(image_size, CV_8UC1);
	vector<vector<Point>> region_list(1, region);
	drawContours(current, region_list, 0, Scalar(255), FILLED, 8);

	Mat element(3, 3, CV_8U, Scalar(1));
	vector<BallCore> cores;
	for (int depth = 0; countNonZero(current) > 0; depth++) {
		add(erosion_depth, Scalar(1), erosion_depth, current);   // every surviving pixel survived one more erosion

		Mat eroded;
		erode(current, eroded, element);

		// any piece with nothing left after this erosion is recorded as a core
		vector<vector<Point>> pieces;
		findContours(current, pieces, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		for (int piece = 0; piece < (int)pieces.size(); piece++) {
			Mat piece_mask = Mat::zeros(image_size, CV_8UC1);
			drawContours(piece_mask, pieces, piece, Scalar(255), FILLED, 8);
			if (countNonZero(piece_mask & eroded) == 0) {
				Moments piece_moments = moments(piece_mask, true);
				BallCore core;
				core.centre = Point2f((float)(piece_moments.m10 / piece_moments.m00),
					(float)(piece_moments.m01 / piece_moments.m00));
				core.depth = depth;
				cores.push_back(core);
			}
		}
		current = eroded;
	}
	return cores;
}

// keep only cores that are thick enough to be a ball, and suppress cores closer than one ball
// diameter to a deeper core: two balls cannot overlap, so their centres are at least 43mm apart.
// candidate_count returns how many cores there were before suppression (for the report).
vector<BallCore> SplitMergedRegion(const vector<Point>& region, const Size& image_size, Mat& erosion_depth, int& candidate_count) {
	vector<BallCore> all_cores = FindBallCores(region, image_size, erosion_depth);

	// deepest (most ball-like) cores first
	sort(all_cores.begin(), all_cores.end(),
		[](const BallCore& a, const BallCore& b) { return a.depth > b.depth; });

	vector<BallCore> ball_cores;
	candidate_count = 0;
	for (int core = 0; core < (int)all_cores.size(); core++) {
		if (all_cores[core].depth < MIN_CORE_DEPTH)
			continue;                                  // too thin to be a ball
		candidate_count++;
		bool too_close = false;
		for (int kept = 0; kept < (int)ball_cores.size(); kept++)
			if (norm(all_cores[core].centre - ball_cores[kept].centre) < BALL_DIAMETER)
				too_close = true;                      // same ball as a deeper core
		if (!too_close)
			ball_cores.push_back(all_cores[core]);
	}
	return ball_cores;
}

// a detected ball: its position in table coordinates (x = column, y = row, in mm)
struct Detection {
	Point2f position;
	bool from_split;    // true if it was found by splitting a merged group (Step 8)
};

// position of a single-ball region = its centroid from moments: (M10/M00, M01/M00)
Point2f RegionCentroid(const vector<Point>& region) {
	Moments region_moments = moments(region);
	return Point2f((float)(region_moments.m10 / region_moments.m00),
		(float)(region_moments.m01 / region_moments.m00));
}

// for the report: a ball-sized circle at every detection
// green = single-ball region, blue = split out of a merged group
Mat DrawDetections(const Mat& table_image, const vector<Detection>& detections) {
	Mat display = table_image.clone();
	for (int i = 0; i < (int)detections.size(); i++) {
		Scalar colour = detections[i].from_split ? Scalar(255, 0, 0) : Scalar(0, 255, 0);
		circle(display, detections[i].position, BALL_DIAMETER / 2, colour, 2);
		circle(display, detections[i].position, 3, colour, FILLED);
	}
	return display;
}

bool BallPresent(int image_no, int ball) {
	return GROUND_TRUTH[image_no][ball][0] >= 0;
}

// true position as (x, y) = (column, row)
Point2f GroundTruthPosition(int image_no, int ball) {
	return Point2f((float)GROUND_TRUTH[image_no][ball][1], (float)GROUND_TRUTH[image_no][ball][0]);
}

string BallName(int ball) {
	return (ball == 0) ? "W" : to_string(ball);
}

// automatic sanity check: warn about any ground-truth position outside the table
void CheckGroundTruth() {
	for (int image_no = 0; image_no < NUMBER_OF_IMAGES; image_no++)
		for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
			if (BallPresent(image_no, ball) &&
				(GROUND_TRUTH[image_no][ball][0] >= TABLE_ROWS || GROUND_TRUTH[image_no][ball][1] >= TABLE_COLS))
				cout << "WARNING: ground truth for " << POOL_FILES[image_no] << " ball " << BallName(ball)
				<< " is outside the table" << endl;
}

// for checking the ground truth: magenta circle at every true ball position, with its label
// drawn beside the ball (so the ball's colour stays visible) in outlined text
Mat DrawGroundTruth(const Mat& table_image, int image_no) {
	Mat display = table_image.clone();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		if (BallPresent(image_no, ball)) {
			Point2f position = GroundTruthPosition(image_no, ball);
			circle(display, position, BALL_DIAMETER / 2, Scalar(255, 0, 255), 2);
			Point2f label_position = position + Point2f((float)(BALL_DIAMETER / 2 + 2), 6.0f);
			putText(display, BallName(ball), label_position, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 4);
			putText(display, BallName(ball), label_position, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
		}
	return display;
}

// result of evaluating one image's detections against its ground truth
struct EvaluationResult {
	int true_positives = 0, false_positives = 0, false_negatives = 0;
	double total_error = 0.0, total_row_error = 0.0, total_col_error = 0.0, max_error = 0.0;
	vector<pair<int, int>> matches;      // (ball, detection)
	vector<bool> ball_matched;           // per ball
	vector<bool> detection_matched;      // per detection
};

double Precision(int tp, int fp) { return (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0; }
double Recall(int tp, int fn) { return (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0; }
double F1(double precision, double recall) {
	return (precision + recall > 0.0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

// one-to-one matching of detections to true balls within match_distance (closest pairs first),
// then TP / FP / FN and position errors for the matches
EvaluationResult EvaluateDetections(const vector<Detection>& detections, int image_no, double match_distance) {
	EvaluationResult result;
	result.ball_matched.assign(NUMBER_OF_BALLS, false);
	result.detection_matched.assign(detections.size(), false);

	// every (ball, detection) pair within the matching distance
	vector<pair<double, pair<int, int>>> candidate_pairs;
	int balls_present = 0;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		if (BallPresent(image_no, ball)) {
			balls_present++;
			for (int detection = 0; detection < (int)detections.size(); detection++) {
				double distance = norm(detections[detection].position - GroundTruthPosition(image_no, ball));
				if (distance <= match_distance)
					candidate_pairs.push_back(make_pair(distance, make_pair(ball, detection)));
			}
		}
	sort(candidate_pairs.begin(), candidate_pairs.end());   // closest first

	// accept a pair only if neither the ball nor the detection is already matched
	for (int i = 0; i < (int)candidate_pairs.size(); i++) {
		int ball = candidate_pairs[i].second.first;
		int detection = candidate_pairs[i].second.second;
		if (result.ball_matched[ball] || result.detection_matched[detection])
			continue;
		result.ball_matched[ball] = true;
		result.detection_matched[detection] = true;
		result.matches.push_back(make_pair(ball, detection));
		result.true_positives++;

		Point2f error = detections[detection].position - GroundTruthPosition(image_no, ball);
		result.total_error += candidate_pairs[i].first;
		result.total_row_error += error.y;      // + = detection is below the true position
		result.total_col_error += error.x;      // + = detection is to the right of the true position
		result.max_error = max(result.max_error, candidate_pairs[i].first);
	}
	result.false_positives = (int)detections.size() - result.true_positives;
	result.false_negatives = balls_present - result.true_positives;
	return result;
}

void PrintEvaluation(const EvaluationResult& result) {
	double precision = Precision(result.true_positives, result.false_positives);
	double recall = Recall(result.true_positives, result.false_negatives);
	cout << fixed << setprecision(2)
		<< "   TP = " << result.true_positives << ", FP = " << result.false_positives << ", FN = " << result.false_negatives
		<< ", precision = " << precision << ", recall = " << recall << ", F1 = " << F1(precision, recall) << endl;
	if (result.true_positives > 0)
		cout << setprecision(1)
		<< "   position error: mean = " << result.total_error / result.true_positives << " mm, max = " << result.max_error
		<< " mm, mean row error = " << result.total_row_error / result.true_positives
		<< " mm, mean column error = " << result.total_col_error / result.true_positives << " mm" << endl;
}

// for the report: green = match (circle at the true position, line to the detection),
// red cross = false positive, yellow circle = missed ball (false negative)
Mat DrawEvaluation(const Mat& table_image, const vector<Detection>& detections, int image_no, const EvaluationResult& result) {
	Mat display = table_image.clone();
	int radius = BALL_DIAMETER / 2;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		if (BallPresent(image_no, ball) && !result.ball_matched[ball])
			circle(display, GroundTruthPosition(image_no, ball), radius, Scalar(0, 255, 255), 2);
	for (int i = 0; i < (int)result.matches.size(); i++) {
		Point2f truth = GroundTruthPosition(image_no, result.matches[i].first);
		Point2f found = detections[result.matches[i].second].position;
		circle(display, truth, radius, Scalar(0, 255, 0), 2);
		line(display, truth, found, Scalar(0, 255, 0), 2);
		circle(display, found, 3, Scalar(0, 255, 0), FILLED);
	}
	for (int detection = 0; detection < (int)detections.size(); detection++)
		if (!result.detection_matched[detection])
			drawMarker(display, detections[detection].position, Scalar(0, 0, 255), MARKER_TILTED_CROSS, 20, 3);
	return display;
}

// overall results on the TEST images only (Table1-6): TP, FP, FN summed over the images first.
// AllBalls is the training image, so it is not included.
void PrintOverallResults(const vector<vector<Detection>>& all_detections, double match_distance) {
	int tp = 0, fp = 0, fn = 0;
	double total_error = 0.0, total_row_error = 0.0, total_col_error = 0.0;
	for (int image_no = 1; image_no < NUMBER_OF_IMAGES; image_no++) {
		EvaluationResult result = EvaluateDetections(all_detections[image_no], image_no, match_distance);
		tp += result.true_positives;
		fp += result.false_positives;
		fn += result.false_negatives;
		total_error += result.total_error;
		total_row_error += result.total_row_error;
		total_col_error += result.total_col_error;
	}
	double precision = Precision(tp, fp), recall = Recall(tp, fn);
	cout << fixed << setprecision(1) << "   match distance " << match_distance << " mm: "
		<< setprecision(2) << "TP = " << tp << ", FP = " << fp << ", FN = " << fn
		<< ", precision = " << precision << ", recall = " << recall << ", F1 = " << F1(precision, recall);
	if (tp > 0)
		cout << setprecision(1) << ", mean error = " << total_error / tp << " mm (row " << total_row_error / tp
		<< ", col " << total_col_error / tp << ")";
	cout << endl;
}

// Part 2 Step 2: the pixels used to describe each ball.
// A pixel belongs to a ball's sample if it is
//   1. in the object mask (not cloth, not table structure),
//   2. within SAMPLE_RADIUS of the ball's position, and
//   3. closer to this ball's position than to any other (the allocation step of k-means,
//      with the ball positions as fixed exemplars), so neighbouring balls share pixels fairly.
vector<Mat> MakeBallSampleMasks(const vector<Point2f>& positions, const Mat& object_mask) {
	vector<Mat> sample_masks;
	Rect image_rect(0, 0, object_mask.cols, object_mask.rows);
	for (int ball = 0; ball < (int)positions.size(); ball++) {
		Mat sample_mask = Mat::zeros(object_mask.size(), CV_8UC1);
		Rect search_box((int)positions[ball].x - SAMPLE_RADIUS, (int)positions[ball].y - SAMPLE_RADIUS,
			2 * SAMPLE_RADIUS + 1, 2 * SAMPLE_RADIUS + 1);
		search_box &= image_rect;
		for (int row = search_box.y; row < search_box.y + search_box.height; row++)
			for (int col = search_box.x; col < search_box.x + search_box.width; col++) {
				if (object_mask.at<uchar>(row, col) == 0)
					continue;                                              // test 1: cloth or table structure
				Point2f pixel((float)col, (float)row);
				double distance = norm(pixel - positions[ball]);
				if (distance > SAMPLE_RADIUS)
					continue;                                              // test 2: too far away
				bool nearest = true;
				for (int other = 0; other < (int)positions.size(); other++)
					if (other != ball && norm(pixel - positions[other]) < distance)
						nearest = false;                                   // test 3: closer to another ball
				if (nearest)
					sample_mask.at<uchar>(row, col) = 255;
			}
		sample_masks.push_back(sample_mask);
	}
	return sample_masks;
}

// for the report: each ball's sampled pixels in their real colours, everything else darkened,
// with a small dot at each ball position
Mat DrawBallSamples(const Mat& table_image, const vector<Point2f>& positions, const vector<Mat>& sample_masks) {
	Mat display = table_image * 0.25;
	for (int ball = 0; ball < (int)sample_masks.size(); ball++)
		table_image.copyTo(display, sample_masks[ball]);
	for (int ball = 0; ball < (int)positions.size(); ball++)
		circle(display, positions[ball], 3, Scalar(255, 255, 255), FILLED);
	return display;
}

// smallest and average sample size (number of pixels) in an image
void PrintSampleSizes(const vector<Mat>& sample_masks) {
	if (sample_masks.empty())
		return;
	int smallest = 1000000, total = 0;
	for (int ball = 0; ball < (int)sample_masks.size(); ball++) {
		int size = countNonZero(sample_masks[ball]);
		smallest = min(smallest, size);
		total += size;
	}
	cout << "   ball samples: smallest = " << smallest << " pixels, average = "
		<< total / (int)sample_masks.size() << " pixels" << endl;
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

	CheckGroundTruth();

	// ---- Training: build the cloth colour model from AllBalls.jpeg ONLY ----
	Mat training_image = LoadImage(TRAINING_IMAGE_INDEX);
	if (training_image.empty())
		return;
	Mat training_table = RectifyTable(training_image, perspective_matrix);   // unsmoothed (used for ball colours)
	Mat training_smoothed = SmoothImage(training_table);
	Mat sample_mask = MakeClothSampleMask();
	MatND cloth_histogram = BuildClothModel(training_smoothed, sample_mask);

	Mat sample_display = DrawClothSampleRegion(training_smoothed, sample_mask);
	Mat histogram_display = DrawClothHistogram(cloth_histogram);
	SaveResult("Step3_ClothSample", TRAINING_IMAGE_INDEX, sample_display);
	SaveResult("Step3_ClothHistogram", TRAINING_IMAGE_INDEX, histogram_display);

	// table structure: run the training image through Steps 3-5, then remove its balls
	Mat training_cleaned = CleanMask(ZeroSupportMask(ComputeClothProbability(training_smoothed, cloth_histogram)));
	Mat structure_mask = MakeTableStructureMask(training_cleaned);
	SaveResult("Step6_TableStructure", TRAINING_IMAGE_INDEX, structure_mask);

	// single ball area: run the training image through Step 6, then look at regions containing exactly one ball
	Mat training_object_mask = RemoveTableStructure(training_cleaned, structure_mask);
	double single_ball_area = LearnSingleBallArea(FindRegions(training_object_mask));
	if (single_ball_area <= 0.0) {
		cout << "Could not learn the single ball area from the training image" << endl;
		return;
	}

	// Part 2 Step 2: pixel samples of the 16 labelled balls in AllBalls, at their ground-truth positions
	vector<Point2f> training_positions;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		training_positions.push_back(GroundTruthPosition(TRAINING_IMAGE_INDEX, ball));
	vector<Mat> training_samples = MakeBallSampleMasks(training_positions, training_object_mask);
	Mat training_samples_display = DrawBallSamples(training_table, training_positions, training_samples);
	cout << "Training ball samples (AllBalls):" << endl;
	PrintSampleSizes(training_samples);
	SaveResult("Part2_Step2_TrainingSamples", TRAINING_IMAGE_INDEX, training_samples_display);

	namedWindow("Cloth sample region", WINDOW_NORMAL);
	namedWindow("Cloth histogram (rows = hue, cols = saturation)", WINDOW_AUTOSIZE);
	namedWindow("Table structure mask", WINDOW_NORMAL);
	namedWindow("Training ball samples (Part 2)", WINDOW_NORMAL);
	imshow("Cloth sample region", sample_display);
	imshow("Cloth histogram (rows = hue, cols = saturation)", histogram_display);
	imshow("Table structure mask", structure_mask);
	imshow("Training ball samples (Part 2)", training_samples_display);
	cout << "Cloth model built - press any key to process the images" << endl;
	waitKey(0);
	cv::destroyAllWindows();

	// ---- Process every image ----
	namedWindow("Evaluation (Step 10): green = match, red = false positive, yellow = missed", WINDOW_NORMAL);
	namedWindow("Ball samples (Part 2 Step 2)", WINDOW_NORMAL);

	vector<vector<Detection>> all_detections(NUMBER_OF_IMAGES);

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

		// Step 4: threshold -> binary mask of candidate ball pixels
		// Otsu's mask is computed for comparison only; the zero-support mask is used
		double otsu_threshold;
		Mat otsu_mask = ThresholdClothProbability(cloth_probability, otsu_threshold);
		Mat ball_mask = ZeroSupportMask(cloth_probability);
		cout << "   Otsu threshold = " << otsu_threshold << endl;
		SaveResult("Step4_OtsuMask", image_no, otsu_mask);
		SaveResult("Step4_Mask", image_no, ball_mask);
		SaveResult("Step4_Histogram", image_no, DrawProbabilityHistogram(cloth_probability, otsu_threshold));

		// Step 5: clean the mask (opening then closing)
		Mat cleaned_mask = CleanMask(ball_mask);
		Mat morphology_changes = DrawMorphologyChanges(ball_mask, cleaned_mask);
		SaveResult("Step5_CleanedMask", image_no, cleaned_mask);
		SaveResult("Step5_Changes", image_no, morphology_changes);

		// Step 6A: remove the table structure (cushions, pockets, marks)
		Mat object_mask = RemoveTableStructure(cleaned_mask, structure_mask);
		SaveResult("Step6_ObjectMask", image_no, object_mask);

		// Step 6B: connected components -> list of regions
		vector<vector<Point>> regions = FindRegions(object_mask);
		Mat regions_display = DrawRegions(table_image, regions);
		cout << "   regions found = " << regions.size() << endl;
		SaveResult("Step6_Regions", image_no, regions_display);

		// Step 7: classify each region as noise / single ball / merged balls
		vector<int> ball_counts;
		int singles = 0, groups = 0, rejected = 0, estimated_balls = 0;
		for (int region = 0; region < (int)regions.size(); region++) {
			int ball_count = EstimateBallCount(regions[region], single_ball_area);
			ball_counts.push_back(ball_count);
			if (ball_count == 0) rejected++;
			else if (ball_count == 1) singles++;
			else groups++;
			estimated_balls += ball_count;
		}
		cout << "   single balls = " << singles << ", merged groups = " << groups
			<< ", rejected = " << rejected << ", estimated balls = " << estimated_balls << endl;
		Mat classification_display = DrawRegionClassification(table_image, regions, ball_counts);
		SaveResult("Step7_Classification", image_no, classification_display);

		// Step 8: split merged groups into individual balls (repeated erosion)
		vector<BallCore> split_balls;
		Mat erosion_depth = Mat::zeros(table_image.size(), CV_8UC1);
		for (int region = 0; region < (int)regions.size(); region++)
			if (ball_counts[region] >= 2) {
				int candidate_count;
				vector<BallCore> cores = SplitMergedRegion(regions[region], table_image.size(), erosion_depth, candidate_count);
				Rect box = boundingRect(regions[region]);
				cout << "   merged group at (row " << box.y + box.height / 2 << ", col " << box.x + box.width / 2
					<< "): estimated " << ball_counts[region] << " balls, " << candidate_count
					<< " candidate cores, split into " << cores.size() << endl;
				split_balls.insert(split_balls.end(), cores.begin(), cores.end());
			}

		// Step 9: ball positions in table coordinates (mm)
		vector<Detection> detections;
		for (int region = 0; region < (int)regions.size(); region++)
			if (ball_counts[region] == 1) {
				Detection detection;
				detection.position = RegionCentroid(regions[region]);
				detection.from_split = false;
				detections.push_back(detection);
			}
		for (int core = 0; core < (int)split_balls.size(); core++) {
			Detection detection;
			detection.position = split_balls[core].centre;
			detection.from_split = true;
			detections.push_back(detection);
		}

		cout << "   balls detected = " << detections.size() << ", positions (row,col):";
		for (int i = 0; i < (int)detections.size(); i++)
			cout << " (" << (int)round(detections[i].position.y) << "," << (int)round(detections[i].position.x) << ")";
		cout << endl;

		Mat detections_display = DrawDetections(table_image, detections);
		SaveResult("Step8_ErosionDepth", image_no, erosion_depth * 10);
		SaveResult("Step9_Detections", image_no, detections_display);

		// Step 10: ground truth (drawn first, to check it visually)
		Mat ground_truth_display = DrawGroundTruth(table_image, image_no);
		SaveResult("Step10_GroundTruth", image_no, ground_truth_display);

		// Step 10: evaluation against the ground truth
		EvaluationResult evaluation = EvaluateDetections(detections, image_no, MATCH_DISTANCE);
		PrintEvaluation(evaluation);
		Mat evaluation_display = DrawEvaluation(table_image, detections, image_no, evaluation);
		SaveResult("Step10_Evaluation", image_no, evaluation_display);

		all_detections[image_no] = detections;

		// Part 2 Step 2: pixel samples of each detected ball
		vector<Point2f> detected_positions;
		for (int i = 0; i < (int)detections.size(); i++)
			detected_positions.push_back(detections[i].position);
		vector<Mat> detection_samples = MakeBallSampleMasks(detected_positions, object_mask);
		Mat samples_display = DrawBallSamples(table_image, detected_positions, detection_samples);
		PrintSampleSizes(detection_samples);
		SaveResult("Part2_Step2_Samples", image_no, samples_display);

		imshow("Evaluation (Step 10): green = match, red = false positive, yellow = missed", evaluation_display);
		imshow("Ball samples (Part 2 Step 2)", samples_display);
		waitKey(0);
	}
	cv::destroyAllWindows();

	// ---- Overall results on the test images (Table1-6) ----
	cout << endl << "OVERALL RESULTS (test images Table1-6, training image excluded):" << endl;
	PrintOverallResults(all_detections, MATCH_DISTANCE);

	// sensitivity to the matching distance (evaluation only - nothing in the method is changed)
	cout << endl << "Sensitivity to the matching distance:" << endl;
	const double distances[] = { 10.0, BALL_DIAMETER / 2.0, 32.0, (double)BALL_DIAMETER };
	for (int i = 0; i < 4; i++)
		PrintOverallResults(all_detections, distances[i]);
}