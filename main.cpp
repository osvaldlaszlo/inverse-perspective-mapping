#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <math.h>
#define DEG2RAD 0.01745329252f

// variable names can be referenced from "GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection, 1998"
// small h corresponds to big FOV
const int CAMERA_POS_Y = 0;  // d (cm)
const int CAMERA_POS_X = 0;  // l (cm)
const int CAMERA_POS_Z = 60; // h (cm)
const float FOV_H = 80.0f;   // (degree)
const float FOV_V = 50.0f;   // (degree)

void build_ipm_table
    (
    const int srcw,
    const int srch,
    const int dstw,
    const int dsth,
    const int vptx,
    const int vpty,
    int* maptable
    )
{
    const float alpha_h = 0.5f * FOV_H * DEG2RAD;
    const float alpha_v = 0.5f * FOV_V * DEG2RAD;
    const float gamma = -(float)(vptx - (srcw >> 1)) * alpha_h / (srcw >> 1); // camera pan angle
    const float theta = -(float)(vpty - (srch >> 1)) * alpha_v / (srch >> 1); // camera tilt angle

    const int front_map_start_position = dsth >> 1;
    const int front_map_end_position = front_map_start_position + dsth;
    const int side_map_mid_position = dstw >> 1;
    //scale to get better mapped image
    const int front_map_scale_factor = 4;
    const int side_map_scale_factor = 2;

    for (int y = 0; y < dstw; ++y)
        {
        for (int x = front_map_start_position; x < front_map_end_position; ++x)
            {
            int idx = y * dsth + (x - front_map_start_position);

            int deltax = front_map_scale_factor * (front_map_end_position - x - CAMERA_POS_X);
            int deltay = side_map_scale_factor * (y - side_map_mid_position - CAMERA_POS_Y);

            if (deltay == 0)
                {
                maptable[idx] = maptable[idx - dsth];
                }
            else
                {
                int u = (int)((atan(CAMERA_POS_Z * sin(atan((float)deltay / deltax)) / deltay) - (theta - alpha_v)) / (2 * alpha_v / srch));
                int v = (int)((atan((float)deltay / deltax) - (gamma - alpha_h)) / (2 * alpha_h / srcw));
                if (u >= 0 && u < srch && v >= 0 && v < srcw)
                    {
                    maptable[idx] = srcw * u + v;
                    }
                else
                    {
                    maptable[idx] = -1;
                    }
                }
            }
        }
}

void inverse_perspective_mapping
    (
    const int dstw,
    const int dsth,
    const unsigned char* src,
    const int* maptable,
    unsigned char* dst
    )
{
    // dst image (1cm/pixel)
    int idx = 0;
    for (int j = 0; j < dsth; ++j)
        {
        for (int i = 0; i < dstw; ++i)
            {
            if (maptable[idx] != -1)
                {
                dst[i * dsth + j] = src[maptable[idx]];
                }
            else
                {
                dst[i * dsth + j] = 0;
                }
            ++idx;
            }
        }
}

int main
    (
    int ac, 
    char** av
    )
{
    const int SRC_RESIZED_WIDTH = 720;
    const int SRC_RESIZED_HEIGHT = 360;
    const int DST_REMAPPED_WIDTH = 200;
    const int DST_REMAPPED_HEIGHT = 200;

    // init vanishing point at center of image
    int vanishing_point_x = SRC_RESIZED_WIDTH >> 1;
    int vanishing_point_y = SRC_RESIZED_HEIGHT >> 1;

    // build inverse perspective mapping table first
    int* ipm_table = new int[DST_REMAPPED_WIDTH * DST_REMAPPED_HEIGHT];
    build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT, 
                    DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, 
                    vanishing_point_x, vanishing_point_y, ipm_table);

    VideoCapture cap;
    cap.open(av[1]);
    if (!cap.isOpened())
        {
        cout << "failed to open video" << endl;
        return -1;
        }

    Mat im;
    Mat imresize;
    Mat grayresize;
    Mat imremapped = Mat(DST_REMAPPED_HEIGHT, DST_REMAPPED_WIDTH, CV_8UC1);
    char key = 0;

    while (key != 27) // press esc to stop
        {
        cap >> im;
        resize(im, imresize, Size(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT));
        cvtColor(imresize, grayresize, CV_BGR2GRAY);

        inverse_perspective_mapping(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, grayresize.data, ipm_table, imremapped.data);

        line(imresize, Point(vanishing_point_x + 10, vanishing_point_y), Point(vanishing_point_x - 10, vanishing_point_y), Scalar(0, 0, 255));
        line(imresize, Point(vanishing_point_x, vanishing_point_y + 10), Point(vanishing_point_x, vanishing_point_y - 10), Scalar(0, 0, 255));
        imshow("resize", imresize);
        imshow("remap", imremapped);

        key = waitKey(10);
        // adjust vanishing point position
        switch (key)
            {
            case 'a':
                vanishing_point_x -= 2;
                build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                                DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                                vanishing_point_x, vanishing_point_y, ipm_table);
                break;
            case 'w':
                vanishing_point_y -= 2;
                build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                                DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                                vanishing_point_x, vanishing_point_y, ipm_table);
                break;
            case 's':
                vanishing_point_y += 2;
                build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                                DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                                vanishing_point_x, vanishing_point_y, ipm_table);
                break;
            case 'd':
                vanishing_point_x += 2;
                build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                                DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                                vanishing_point_x, vanishing_point_y, ipm_table);
                break;
            default:
                break;
            }
        }

    delete [] ipm_table;
    return 0;
}