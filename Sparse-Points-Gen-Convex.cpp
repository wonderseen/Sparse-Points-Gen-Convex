/*
WonderSeen Release
A demo to gen the convex of an image with sparse points and visualize the optimized-convex result.
github: https://github.com/wonderseen/
csdn: http://blog.csdn.net/wonderseen
*/

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
    // read pre-treated image
    Mat image = imread("image/1.png", IMREAD_COLOR);
    Mat img(image.size(), CV_8UC1, Scalar(0));
    vector<Point> min_point, max_point;
    vector<vector<Point> > contours;
    Mat img1 = image.clone();

    // get the outline
    cvtColor(image,img,COLOR_BGR2GRAY);
    threshold(img,img, 1,250,CV_THRESH_BINARY);
    Mat Dilateelement = getStructuringElement(MORPH_RECT, Size(10,10));
    dilate(img,img,Dilateelement);
    int i,j;
    for (j = 0; j < img.rows; j++){
        int min_i = img.cols-1;
        int max_i = 0;
        for (i = 0; i < img.cols; i++){
            if(img.at<char>(Point(i,j)) != 0 ){
                if(i>max_i) max_i = i;
                if(i<min_i) min_i = i;
            }
        }
        if(min_i != img.cols-1 && max_i > 1 && min_i != max_i){
            if(min_point.size() == 0){
                min_point.push_back(Point(min_i,j));
                max_point.push_back(Point(max_i,j));
            }
            else if(min_i - min_point[min_point.size()-1].x < 60 && min_i-min_point[min_point.size()-1].x>-60){
                if(abs(min_i-min_point[min_point.size()-1].x) > 30){
                    int min;
                    if(min_i > (*(min_point.end()-1)).x)
                    {
                        min = min_i/3+(*(min_point.end()-1)).x/3*2;
                    }
                    else  min = min_i/3*2+(*(min_point.end()-1)).x/3;
                    min_point.push_back(Point(min,j));
                }

                else{
                    min_point.push_back(Point(min_i,j));
                }

                if(abs(max_i-max_point[max_point.size()-1].x) > 30){
                    int max;
                    if(max_i < (*(max_point.end()-1)).x)
                    {
                        max = max_i*0.02 + (*(max_point.end()-1)).x*0.98;
                    }
                    else  max = max_i*0.98 + (*(max_point.end()-1)).x*0.02;
                    max_point.push_back(Point(max,j));
                }
                else{
                    max_point.push_back(Point(max_i,j));
                }
            }
        }
    }

    // draw the outlines
    Mat showimg(img.size(), CV_8UC3, Scalar(0,0,0));
    for(i=0; i<min_point.size()-2;i++){
        cv::line( showimg, min_point[i], min_point[i+1], cv::Scalar(255), 1);
        cv::line( showimg, max_point[i], max_point[i+1], cv::Scalar(255), 1);
    }
    cv::line( showimg, min_point[i+1], max_point[i+1], cv::Scalar(0,0,255), 1);
    cv::line( showimg, min_point[0], max_point[0], cv::Scalar(0,0,255), 1);
    
    // gen the poly-contour of raw-img
    Mat showgray(image.size(), CV_8UC1, Scalar(0));
    cvtColor( showimg, showgray, COLOR_BGR2GRAY);
    threshold( showgray, showgray, 10,255,CV_THRESH_BINARY);
    vector<Vec4i> hierarcy;
    findContours(showgray, contours, hierarcy, 0, CV_CHAIN_APPROX_NONE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<vector<Point> > hull(contours.size());
    for(i=0; i < contours.size(); i++) approxPolyDP(Mat(contours[i]), contours_poly[i], 1, false);
    for(i=0; i < contours.size(); i++){
        convexHull( Mat(contours_poly[i]), hull[i], false );
        if(hull[i].size()==0)
        {
            cout << "No correct index." << endl;
            return -1;
        }
        //drawContours(img1, hull, i, Scalar(0, 255, 255), 1, 100);
    }

    // Fitting Depressions
    int y1,y2;
    int x1,x2;
    int it = 0;
    for(; it != (*hull.begin()).size(); it++)
    {
        int judge_y;
        if(it == (*hull.begin()).size()-1)
        {
            judge_y = y1-y2;
            x1 = (hull[0][it]).x;
            x2 = (*(*hull.begin()).begin()).x;
            y1 = (hull[0][it]).y;
            y2 = (*(*hull.begin()).begin()).y;
        }
        else
        {
            judge_y = hull[0][it].y-hull[0][it+1].y;
            y1 = hull[0][it].y;
            y2 = hull[0][it+1].y;
            x1 = hull[0][it].x;
            x2 = hull[0][it+1].x;
        }

        if(abs(y1-y2)<=2) continue;

        // if judge_y < 0, the line (x1, y1)-(x2,y2) belongs to the right part of the convex
        if(judge_y < 0){
            float k = (float)(x1-x2)/((float)judge_y);
            // calculate the points score 在该直线的纵向区间内查找
            for(int y = y1+5; y < y2-5; y += 2)
            {
                bool flag = false;
                bool insert_flag = false;
                int start_x = (-y1+y)*k+x1 > 0?(-y1+y)*k+x1:0;
                int score;
                for(int x=start_x+15; x>=0; x--) // 从右往左方向查找
                {
                    if(img.at<char>(Point(x,y)) != 0){
                        if(x>start_x){ // 如果发现有点在直线右侧,直接拓展凸集边界
                            insert_flag = true;
                            flag = true;
                            hull[0].insert((*hull.begin()).begin()+it+1, 1, Point(x,y));
                            break;
                        }
                        int offset_min = 20; //允许凸集最大偏差
                        int offset_max = 50; //允许凸集最大偏差
                        if(abs(x-start_x) > offset_min && abs(x-start_x) < offset_max){ // 如果发现有点在直线左侧,且在处理凹陷阈值内
                            score = 0;
                            for(i=-4; i<=4; i++)
                            {
                                for(j=-2;j<=2;j++)
                                {
                                    if(img.at<char>(start_x+i,y+j)!=0){ // 原来凸集在该行的边界位置就有像素,所以不需要更改此处
                                        flag = true;
                                        break;
                                    }
                                    if(img.at<char>(x+i,y+j)>0) score++;// 如果发现该行凸集边界是需要优化的,就计算离start_x最近的点是否是 密集点分布 还是 偶然分布
                                    if(score>5) break;
                                    else
                                        continue;
                                }
                            }
                            if(!flag || score > 5) insert_flag=true;// 如果是 该行的边界需要优化,且存在最近点的密集分布
                            else{
                                insert_flag = false;
                                break;
                            }
                            cv::circle(img1, Point(x,y), 5, cv::Scalar(125,0,125), 1);
                            //cout << it << " , " << k << ',' << start_x << ",  x-start=" << x-start_x << "  ," << x << ',' << y << endl;
                            // insert new point
                            hull[0].insert((*hull.begin()).begin()+it+1, 1, Point(x,y));
                            break;
                        }

                    }
                }
                if(flag || insert_flag) break;
            }
            cv::line(img1, Point(x1,y1), Point(x2,y2), cv::Scalar(125,0,125), 1);
            imshow("search-process",img1);
            waitKey(100);
        }

        // if judge_y > 0, the line (x1, y1)-(x2,y2) belongs to the left part of the convex
        if(judge_y > 0){
            float k = (float)(x1-x2)/((float)judge_y);
            // calculate the points score 在该直线的纵向区间内查找
            for(int y = y2-5; y < y1+5; y += 2)
            {
                cout << y2-5 << ',' << y1+5 << endl;
                bool flag = false;
                bool insert_flag = false;
                int start_x = (-y1+y)*k+x1;
                int score;
                for(int x=start_x-15; x<img.cols; x++) // 从右往左方向查找
                {
                    if(img.at<char>(Point(x,y)) != 0){
                        if(x<start_x){ // 如果发现有点在直线左侧,直接拓展凸集边界
                            insert_flag = true;
                            flag = true;
                            hull[0].insert((*hull.begin()).begin()+it+1, 1, Point(x,y));
                            break;
                        }
                        int offset_min = 20; //允许凸集最大偏差
                        int offset_max = 50; //允许凸集最大偏差
                        if(abs(x-start_x) > offset_min && abs(x-start_x) < offset_max){ // 如果发现有点在直线左侧,且在处理凹陷阈值内
                            score = 0;
                            for(i=-4; i<=4; i++)
                            {
                                for(j=-2;j<=2;j++)
                                {
                                    if(img.at<char>(start_x+i,y+j)!=0){ // 原来凸集在该行的边界位置就有像素,所以不需要更改此处
                                        flag = true;
                                        break;
                                    }
                                    if(img.at<char>(x+i,y+j)>0) score++;// 如果发现该行凸集边界是需要优化的,就计算离start_x最近的点是否是 密集点分布 还是 偶然分布
                                    if(score>5) break;
                                    else
                                        continue;
                                }
                            }
                            if(!flag || score > 5) insert_flag=true;// 如果是 该行的边界需要优化,且存在最近点的密集分布
                            else{
                                insert_flag = false;
                                break;
                            }
                            cv::circle(img1, Point(x,y), 5, cv::Scalar(125,0,125), 1);
                            //cout << it << " , " << k << ',' << start_x << ",  x-start=" << x-start_x << "  ," << x << ',' << y << endl;
                            // insert new point
                            hull[0].insert((*hull.begin()).begin()+it+1, 1, Point(x,y));
                            break;
                        }
                    }
                }
                if(flag || insert_flag) break;
            }
            cv::line(img1, Point(x1,y1), Point(x2,y2), cv::Scalar(125,0,125), 1);
            imshow("search-process",img1);
            waitKey(100);
        }
    }

    for(it=0; it != (*hull.begin()).size()-1; it++)
    {
        cv::line(img1, (*hull.begin())[it], (*hull.begin())[it+1], cv::Scalar(0,0,125), 2);
    }
    cv::line(img1, (*hull.begin())[it], (*hull.begin())[0], cv::Scalar(0,0,125), 2);
    imshow("Fitting-Result",img1);
    waitKey(20000);
    return 0;
}
