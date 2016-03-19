#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void laws(Mat input, Mat& output, int kernel)
{
    Mat texture_base = Mat::zeros(input.rows, input.cols, CV_32S);
    Mat texture = Mat::zeros(input.rows, input.cols, CV_8U);
    int tenergy, energy_sum;
    int n = kernel;
    //8 Laws texture masks
    int L3E3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int L3S3[9] = {-1, 2, -1, -2, 4, -2, -1, 2, -1};
    int E3E3[9] = {1, 0, -1, 0, 0, 0, -1, 0, 1};
    int E3L3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int E3S3[9] = {1, -2, 1, 0, 0, 0, -1, 2, -1};
    int S3S3[9] = {1, -2, 1, -2, 4, -2, 1, -2, 1};
    int S3L3[9] = {-1, -2, -1, 2, 4, 2, -1, -2, -1};
    int S3E3[9] = {1, 0, -1, -2, 0, 2, 1, 0, -1};

    //convolution of input image with laws masks into integral image
    for(int x=0; x<input.cols; x++)
    {   for(int y=0; y<input.rows; y++)
        {
            tenergy = 0;
            for(int i=0; i<9; i++)
            {
                if(y-1+i/3 >= 0 && y-1+i/3 < input.rows && x-1+i%3 >= 0 && x-1+i%3 < input.cols)
                {   tenergy += input.at<uchar>(y-1+i/3, x-1+i%3)*L3E3[i]
                                + input.at<uchar>(y-1+i/3, x-1+i%3)*E3L3[i];
                }
            }
            tenergy = tenergy;

            if(x>0 && y>0) energy_sum = abs(tenergy) + texture_base.at<int>(y, x-1) + texture_base.at<int>(y-1, x) - texture_base.at<int>(y-1, x-1);
            else if(y == 0 && x > 0) energy_sum = abs(tenergy) + texture_base.at<int>(y, x-1);
            else if(x == 0 && y > 0) energy_sum = abs(tenergy) + texture_base.at<int>(y-1, x);
            else if(x == 0 && y == 0) energy_sum = abs(tenergy);
            texture_base.at<int>(y, x) = energy_sum;
        }
    }
    //calculate texture by average filter
    for(int y=0; y<input.rows; y++)
    {   for(int x=0; x<input.cols; x++)
        {
            tenergy = 0;
            if(x>n && y>n && x<texture_base.cols-n && y<texture_base.rows-n)
            {   tenergy = texture_base.at<int>(y+n, x+n) - texture_base.at<int>(y+n, x-n-1) - texture_base.at<int>(y-n-1, x+n) + texture_base.at<int>(y-n-1, x-n-1);
                tenergy = tenergy/((2*n+1)*(2*n+1));
            }
            else if(x <= n && y>n && y+n<texture_base.rows)
            {   tenergy = texture_base.at<int>(y+n, x+n) - texture_base.at<int>(y-n-1, x+n);
                tenergy = tenergy/((x+n)*(2*n+1));
            }
            else if(x >= texture_base.cols-n && y > n && y+n<texture_base.rows)
            {   tenergy = texture_base.at<int>(y+n, texture_base.cols-1) - texture_base.at<int>(y+n, x-n-1) - texture_base.at<int>(y-n-1, texture_base.cols-1) + texture_base.at<int>(y-n-1, x-n-1);
                tenergy = tenergy/((texture_base.cols-x+n)*(2*n+1));
            }
            /*  else if(y <= n && x>n && x+n<t.cols)
            {   tenergy = t.at<int>(y+n, x+n) - t.at<int>(y+n, x-n-1);
                tenergy = tenergy/((y+n)*(2*n+1));
            }
            else if(y >= t.rows-n && x > n && x+n<t.cols)
            {   tenergy = t.at<int>(t.rows-1, x+n) - t.at<int>(t.rows-1, x-n-1) - t.at<int>(y-n-1, x+n) + t.at<int>(y-n-1, x-n-1);
                tenergy = tenergy/((t.rows-y+n)*(2*n+1));
            }*/
          /*      else if(x <= n && y <= n)
            {   tenergy = t.at<int>(y+n, x+n);
                tenergy = tenergy/((y+n)*(x+n));
            }
            else if(x>n && y>n && x+n>=t.cols && y+n>=t.rows)
            {   tenergy = t.at<int>(t.rows-1, t.cols-1) - t.at<int>(t.rows-1, x-n-1) - t.at<int>(y-n-1, t.cols-1) + t.at<int>(y-n-1, x-n-1);
                tenergy = tenergy/((t.rows-y+n)*(t.cols-x+n));
            }*/
            if(tenergy < 255) texture.at<uchar>(y, x) = tenergy;
            else texture.at<uchar>(y, x) = 255;
        }
    }
    output = texture;
}
