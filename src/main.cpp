#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <limits>
#include <opencv2/features2d.hpp>

#include "laws.hpp"

using namespace cv;
using namespace std;

void get_rowPixCosts(Mat img1, Mat img2, int y, short* cost, int w, int D)
{
    uchar *row1 = img1.ptr<uchar>(y);
    uchar *row2 = img2.ptr<uchar>(y);

    for(int x = 0; x < w; x++)
    {
        for(int d = 0; d < D; d++)
        {
            if(x - d >= 0)
            {
                cost[x * D + d] = abs(row1[x] - row2[x - d]);
            }
            else
            {
                cost[x * D + d] = SHRT_MAX;
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        cout << "No parameters passed, exiting...";
        return 0;
    }

    char* leftPath = argv[1];
    char* rightPath = argv[2];
    Mat im_left = imread(leftPath, IMREAD_GRAYSCALE);
    Mat im_right = imread(rightPath, IMREAD_GRAYSCALE);
    //enable RL aggregation
    bool RL = true;
    int width = im_left.cols;
    int height = im_left.rows;
    int NR = 5;
    int width2 = width + 2;
    int ndisp = 60;
    int nd2 = ndisp + 2;
    short *rowPixCosts = new short[width * ndisp];
    //Aggregated costs
    int *S_m = new int[width * ndisp];
    //Minimal oriented energies for each pixel
    short *Lr_min0 = new short[width2 * NR];
    short *Lr_min1 = new short[width2 * NR];
    //Oriented energies for each pixels disparity
    short *Lr_0 = new short[width2 * nd2];
    short *Lr_1 = new short[width2 * nd2];
    short *Lr_2 = new short[width2 * nd2];
    short *Lr_3 = new short[width2 * nd2];
    short *Lr_4 = new short[width2 * nd2];
    //BT costs for current pixel
    short *C;
    short L0, L1, L2, L3, L4;
    int d_p;
    int cd;
    short delta0, delta1, delta2, delta3, delta4;
    short L0_min, L1_min, L2_min, L3_min, L4_min;
    Mat disparity = Mat::zeros(height, width, CV_16S);
    Mat disparity2 = Mat::zeros(height, width, CV_16S);
    Mat disparity2_costs = Mat::zeros(height, width, CV_32S);
    Mat disparity_norm = Mat::zeros(height, width, CV_8U);
    short *disp2_row;
    int *disp2_cost;
    int y, x, x2;
    int S, d0, dmin, a1, a2, a3, a4;
    int best_disparity;
    int P1 = 10;
    int P2 = 35;
    int uniqueness = 10;
    int disp12maxdiff = 5;

    memset(Lr_min0, 0, sizeof(short) * width2 * NR);
    memset(Lr_min1, 0, sizeof(short) * width2 * NR);
    memset(Lr_0, 0, sizeof(short) * width2 * nd2);
    memset(Lr_1, 0, sizeof(short) * width2 * nd2);
    memset(Lr_2, 0, sizeof(short) * width2 * nd2);
    memset(Lr_3, 0, sizeof(short) * width2 * nd2);
    memset(Lr_4, 0, sizeof(short) * width2 * nd2);

    //preset boundary pixels (it would save few cycles in main loop)
    for( x = 1; x <= width; x++ )
    {
        Lr_0[x * nd2 + 0] = Lr_0[x * nd2 + ndisp + 1] =
        Lr_1[x * nd2 + 0] = Lr_1[x * nd2 + ndisp + 1] =
        Lr_2[x * nd2 + 0] = Lr_2[x * nd2 + ndisp + 1] =
        Lr_3[x * nd2 + 0] = Lr_3[x * nd2 + ndisp + 1] = SHRT_MAX;
    }

    //Cost aggregation
    //For pixels along the borders, Energy is equal to ist lowest BT cost
    for (y = 0; y < height; y++)
    {
        disp2_row = disparity2.ptr<short>(y);
        disp2_cost = disparity2_costs.ptr<int>(y);
        //read pixelwise costs on current row
        get_rowPixCosts(im_left, im_right, y, rowPixCosts, width, ndisp);
        //Calculate -1 row
        if (y == 0)
        {
            //find lowest BT cost
            for (x = 1; x <= width; x++)
            {
                for (d0 = 0; d0 < ndisp; d0++)
                {
                    Lr_0[x * nd2 + d0 + 1] = Lr_1[x * nd2 + d0 + 1] =
                    Lr_2[x * nd2 + d0 + 1] = Lr_3[x * nd2 + d0 + 1] = 0;
                }

                for (int i=0; i<4; i++)
                {
                    Lr_min1[x * NR + i] = 0;
                }
            }
        }

        //Aggregate costs
        for (x = 1; x < width+1; x++)
        {
            C = rowPixCosts + (x - 1) * ndisp;
            delta0 = Lr_min0[(x - 1) * NR + 0] + P2;
            delta1 = Lr_min1[(x - 1) * NR + 1] + P2;
            delta2 = Lr_min1[x * NR + 2] + P2;
            delta3 = Lr_min1[(x + 1) * NR + 3] + P2;
            L0_min = L1_min = L2_min = L3_min = SHRT_MAX;
            for (d0 = 1; d0 <= ndisp; d0++)
            {
                L0 = C[d0 - 1] + min((int)Lr_0[(x - 1) * nd2 + d0],
                                 min((int)Lr_0[(x - 1) * nd2 + d0 - 1] + P1,
                                 min((int)Lr_0[(x - 1) * nd2 + d0 + 1] + P1,
                                 (int)delta0))) - delta0;
                L1 = C[d0 - 1] + min((int)Lr_1[(x - 1) * nd2 + d0],
                                 min((int)Lr_1[(x - 1) * nd2 + d0 - 1] + P1,
                                 min((int)Lr_1[(x - 1) * nd2 + d0 + 1] + P1,
                                 (int)delta1))) - delta1;
                L2 = C[d0 - 1] + min((int)Lr_2[x * nd2 + d0],
                                 min((int)Lr_2[x * nd2 + d0 - 1] + P1,
                                 min((int)Lr_0[x * nd2 + d0 + 1] + P1,
                                 (int)delta2))) - delta2;
                L3 = C[d0 - 1] + min((int)Lr_3[(x + 1) * nd2 + d0],
                                 min((int)Lr_3[(x + 1) * nd2 + d0 - 1] + P1,
                                 min((int)Lr_3[(x + 1) * nd2 + d0 + 1] + P1,
                                 (int)delta3))) - delta3;
                cd = x*nd2 + d0;
                Lr_0[cd] = L0;
                L0_min = min(L0_min, L0);
                Lr_1[cd] = L1;
                L1_min = min(L1_min, L1);
                Lr_2[cd] = L2;
                L2_min = min(L2_min, L2);
                Lr_3[cd] = L3;
                L3_min = min(L3_min, L3);
                S_m[(x - 1) * ndisp + (d0 - 1)] = L0 + L1 + L2 + L3 + P2;
            }

            Lr_min0[x * NR] = L0_min;
            Lr_min0[x * NR + 1] = L1_min;
            Lr_min0[x * NR + 2] = L2_min;
            Lr_min0[x * NR + 3] = L3_min;
        }

        //aggregate costs in right-to-left direction
        if (RL)
        {
            for (x = 0; x < width; x++)
            {
                disparity.at<short>(y, x) = disp2_row[x] = -1;
                disp2_cost[x] = INT_MAX;
            }

            for (x = width; x >= 1; x--)
            {
                C = rowPixCosts + (x - 1) * ndisp;
                delta4 = Lr_min0[(x + 1) * NR + 4] + P2;
                L4_min = SHRT_MAX;
                S = INT_MAX;
                for (d0 = 1; d0 <= ndisp; d0++)
                {
                    L4 = C[d0 - 1] + min((int)Lr_4[(x + 1) * nd2 + d0],
                                     min((int)Lr_4[(x + 1) * nd2 + d0 - 1] + P1,
                                     min((int)Lr_4[(x + 1) * nd2 + d0 + 1] + P1,
                                     (int)delta4))) - delta4;
                    Lr_4[x * nd2 + d0] = L4;
                    L4_min = min(L4_min, L4);
                    S_m[(x - 1) * ndisp + (d0 - 1)] += L4;
                    //select best disparity
                    if (S_m[(x - 1) * ndisp + (d0 - 1)] < S)
                    {
                        S = S_m[(x - 1) * ndisp + (d0 - 1)];
                        best_disparity = d0 - 1;
                    }
                }

                Lr_min0[x * NR + 4] = L4_min;
                //check if subpix approximation is possible
                for (d0 = 0; d0 < ndisp; d0++)
                {
                    if (S_m[(x - 1) * ndisp + d0] * (100 - uniqueness) < S * 100
                       && abs(best_disparity - d0) > 1)
                    {
                        break;
                    }
                }

                if (d0 < ndisp)
                {
                    continue;
                }

                //subpix approximation
                d0 = best_disparity;
                x2 = (x - 1) - d0;
                if (disp2_cost[x2] > S)
                {
                    disp2_cost[x2] = (int)S;
                    disp2_row[x2] = (short)d0;
                }
                //fit nearby points with quadratc curve
                if (0 < d0 && d0 < ndisp - 1)
                {
                    int denom2 = max(S_m[(x - 1) * ndisp + d0 - 1] +
                                 S_m[(x - 1) * ndisp + d0+1] -
                                 2 * S_m[(x - 1) * ndisp + d0], 1);
                    d0 = d0 + ((S_m[(x - 1) * ndisp + d0 - 1] -
                       S_m[(x - 1) * ndisp + d0 + 1]) + denom2) / (denom2 * 2);
                }
                else
                {
                    d0 = d0;
                }
                disparity.at<short>(y, x - 1) = best_disparity;
            }

            //validate disparities
            for (x = 0; x < width; x++)
            {
                d0 = disparity.at<short>(y, x);
                if (d0 == -1)
                {
                    continue;
                }

                x2 = x - d0;
                if (0 <= x2 && abs(disp2_row[x2] - d0) > disp12maxdiff)
                {
                    disparity.at<short>(y, x) = -1;
                }
            }
        }

        memset(Lr_min1, 0, sizeof(short) * width2 * NR);
        swap(Lr_min0, Lr_min1);
    }

    medianBlur(disparity, disparity, 3);
    Mat texture, D_gradient, T_gradient;
    laws(im_left, texture, 1);

    //treshold texture
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            if (texture.at<uchar>(y, x) > 15 )
            {
                texture.at<uchar>(y, x) = 0;
            } else
            {
                texture.at<uchar>(y, x) = 255;
            }
        }
    }

    medianBlur(texture, texture, 3);
    //segment binary image using flood fill
    Mat segmented;
    Mat segmented_filtered = Mat(height, width, CV_8U);
    vector< vector <CvPoint> > blobs;
    blobs.clear();
    texture.convertTo(segmented, CV_32S);
    //255 and 0 are already "used"
    int seg_count = 2; 
    int min_area = 100;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            if (segmented.at<int>(y, x) != 1)
            {
                continue;
            }

            Rect rect;
            floodFill(segmented, cvPoint(x, y), Scalar(seg_count),
                      &rect, Scalar(0), Scalar(0), 4);
            vector< CvPoint> blob;
            for (int j = rect.y; j < (rect.y + rect.height); j++)
            {
                for (int i = rect.x; i < (rect.x + rect.width); i++)
                {
                    if (segmented.at<int>(j, i) != seg_count)
                    {
                        continue;
                    }

                    blob.push_back(cvPoint(i, j));
                }
            }

            if(blob.size() > min_area)
            {
                blobs.push_back(blob);
                for (int i = 0; i < blob.size(); i++)
                {
                    segmented_filtered.at<uchar>(blob[i].y, blob[i].x) = 255;
                }
            }
          /*  if(blob.size() > 0)
                cout << blob.size() << endl;*/
            seg_count += 1;

        }
    }

    Mat disparityNormalized = Mat(disparity.rows, disparity.cols, CV_8UC1);
    normalize(disparity, disparityNormalized, 0, 255, CV_MINMAX, CV_8U);
    //normalize(segmented, segmented, 0, 255, CV_MINMAX, CV_8U);
    imshow("seg", segmented_filtered);
    imshow("texture", texture);
    //imshow("image", im_left);
    imshow("disparity", disparityNormalized);
    waitKey(0);

    return 0;
}
