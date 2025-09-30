# coding=utf-8 m7000  Image
import cv2
import csv
import os
import numpy as np
from PIL import Image
import pandas as pd


def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def Char2Num(str1):
    Chars = {
        "A": 1, "B": 2, "C": 3, "D": 4,
        "E": 5, "F": 6, "G": 7, "H": 8,
    }
    return Chars.get(str1, None)


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    print(lower, upper)
    return edged


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def Get_RedCircle(Path1):
    img = Image.open(Path1)
    colors = []
    h, w = img.size
    for x in range(h):
        for y in range(w):
            color = img.getpixel((x, y))
            # print(color)
            if color == (255, 102, 0):
                colors.append([x, y])
    if len(colors) > 0:
        return False
    else:
        return True


Edge_Pixel = 380
# min_radius = 5
# max_radius = 35
min_radius = 15
max_radius = 80
Pixel_radius = 1900  # 3800  1900
PixelTomm = 15.6 / (Pixel_radius * 2)  # Each pixel is equivalent to 2mm. The diameter of the 24-well plate is 15.6mm.
# ImgCenter_X, ImgCenter_Y = 0, 0

lower_Brightness = 50  # 50
Circularity_Limit = 0.8

# lower_Green = np.array([50, 160, 180])  # 35, 43, 46，，60, 200, 120:::35, 60, 120
lower_Green = np.array([60, 255, 90])  # 35, 43, 46，，60, 200, 120:::35, 60, 120,20240325=>([55, 220, 220])
upper_Green = np.array([77, 255, 255])  # np.array([77, 255, 255])
# lower_Green = np.array([156, 100, 100])  # red Lower Limit
# upper_Green = np.array([180, 255, 255])  # red Upper Limit

root = R".\PixImage"
PixLocationRoot = R".\PixLocation"
CenterList = [('WellNr', 'X', 'Y', 'Radius')]
csvFileList = [('FileName', 'WellNumber', 'PickNumber')]
CenterDict = {'A01': (2182, 1920), 'A02': (2170, 1904), 'A03': (2160, 1872), 'A04': (2176, 1886), 'A05': (2174, 1868),
              'A06': (2176, 1874), 'B01': (2184, 1926), 'B02': (2182, 1908), 'B03': (2172, 1896), 'B04': (2190, 1898),
              'B05': (2192, 1890), 'B06': (2192, 1874), 'C01': (2204, 1926), 'C02': (2194, 1934), 'C03': (2188, 1908),
              'C04': (2208, 1908), 'C05': (2192, 1894), 'C06': (2206, 1884), 'D01': (2204, 1932), 'D02': (2214, 1920),
              'D03': (2202, 1934), 'D04': (2212, 1928), 'D05': (2202, 1896), 'D06': (2200, 1886)}

delete_files(PixLocationRoot)

for dir_Path, dir_names, filenames in os.walk(root):
    for filepath in filenames:
        myList = [('No', 'PixelX', 'PixelY', 'PixelArea', 'XOffset', 'YOffset', 'Circularity')]
        imagePath = os.path.join(dir_Path, filepath)

        img_RGB = cv2.imread(imagePath, 1)  # Input image located at path img/0.jpg, with image type being RGB image.

        # Find center of circle
        img_Copy = cv2.imread(imagePath, 1)
        gray_Circle = cv2.cvtColor(img_Copy, cv2.COLOR_BGR2GRAY)
        blurred_Circle = cv2.GaussianBlur(gray_Circle, (15, 15), 0)
        # circles = cv2.HoughCircles(blurred_Circle, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=1880,
        #                            maxRadius=1895)
        circles = cv2.HoughCircles(blurred_Circle, cv2.HOUGH_GRADIENT, 1, 100, param1=20, param2=30, minRadius=1880,
                                   maxRadius=1895)

        if circles is not None:
            circles_count = 0
            for i in circles[0, :]:
                circles_count = 1 + circles_count
            print("CirclesNr:", circles_count)
            circles = np.round(circles[0, :]).astype("int")
            for (x1, y1, r1) in circles:
                ImgCenter_X, ImgCenter_Y, Pixel_radius = x1, y1, r1
                # ImgCenter_Y = y1
                # Pixel_radius = r1
                break
        PixelTomm = 15.6 / (Pixel_radius * 2)
        print(ImgCenter_X, ImgCenter_Y, Pixel_radius)
        CenterList.append([str(filepath.split('.')[0].split('_')[5])[0:3], ImgCenter_X, ImgCenter_Y, Pixel_radius])

        # use Fix Center
        # WellName=str(filepath.split('.')[0].split('_')[5])[0:3]
        # ImgCenter_X = CenterDict[WellName][0]
        # ImgCenter_Y = CenterDict[WellName][1]

        # img_Blur = cv2.GaussianBlur(img_RGB, (7, 7), 0, 0)
        # cv2.medianBlur(img_RGB, 7) cv2.GaussianBlur(img_RGB, (7, 7), 0, 0)

        img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_HSV)

        mask = cv2.inRange(img_HSV, lower_Green, upper_Green)
        # mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.medianBlur(mask, 3)
        # mask = cv2.GaussianBlur(mask, (7, 7), 0, 0)
        maskAnd = cv2.bitwise_and(img_RGB, img_RGB, mask=mask)

        maskAnd = cv2.cvtColor(maskAnd, cv2.COLOR_BGR2RGB)

        # cv2.imwrite(R".\PixLocation\HSV.png", img_HSV) 
        # cv2.imwrite(R".\PixLocation\maskAnd.png", maskAnd)  

        gray = cv2.cvtColor(maskAnd, cv2.COLOR_BGR2GRAY)
        # cv2.threshold(gray, 56, 255,
        ret, binary = cv2.threshold(gray, lower_Brightness, 255,
                                    cv2.THRESH_BINARY)

        # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imwrite(R".\PixLocation\binary.png", binary)
        # cv2.imwrite(R".\PixLocation\adaptiveThreshold.png", th3)
        # break

        # C10 cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY) |cv2.THRESH_OTSU

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # eroded = cv2.erode(binary, kernel)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(binary, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img_RGB, contours, -1, (100, 100, 0), 1)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        index = 1
        for c in contours:
            (x, y), radius = cv2.minEnclosingCircle(c)

            xr, yr, wr, hr = cv2.boundingRect(c)
            ROI = img_RGB[yr: yr + hr, xr: xr + wr]
            cv2.imwrite(R".\PixLocation\ROI.png", ROI)
            center = (int(x), int(y))
            txtLocation = (int(x) + int(radius) + 3, int(y) + int(radius) + 3)
            radius = int(radius)

            # 根据面积和周长计算圆度
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) ** 0.5

            if min_radius < radius < max_radius and distance(x, y, ImgCenter_X, ImgCenter_Y) < (
                    Pixel_radius - Edge_Pixel) and \
                    Get_RedCircle(R".\PixLocation\ROI.png") and circularity > Circularity_Limit:
                img_RGB = cv2.circle(img_RGB, center, radius, (0, 102, 255), 2)
                # print(cv2.contourArea(c))
                myList.append(
                    [index, int(x), int(y), float(cv2.contourArea(c)), round((int(x) - ImgCenter_X) * PixelTomm, 3),
                     round(-(int(y) - ImgCenter_Y) * PixelTomm, 3), circularity])
                cv2.putText(img_RGB, str(index), txtLocation, cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 250), 1,
                            cv2.LINE_AA)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                index = index + 1

        FileName = str(filepath.split('.')[0].split('_')[5])[0:3]
        csvFileName = str(Char2Num(str(FileName[0:1])) + (int(FileName[1:3]) - 1) * 4)
        csvPath = os.path.join(PixLocationRoot, FileName + ".csv")

        csvFileList.append([FileName, csvFileName, index - 1])

        with open(csvPath, mode='w', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerows(myList)

        # cv2.imshow("img", img)
        img_RGB = cv2.circle(img_RGB, (ImgCenter_X, ImgCenter_Y), 2, (250, 5, 5), 2)
        img_RGB = cv2.circle(img_RGB, (ImgCenter_X, ImgCenter_Y), Pixel_radius, (250, 5, 5), 2)
        # img_RGB = cv2.circle(img_RGB, ((w // 2), h // 2), 880, (205, 50, 10), 2)
        Path = os.path.join(PixLocationRoot, FileName + ".PNG")
        cv2.imwrite(Path, img_RGB)
        if os.path.exists(R".\PixLocation\ROI.png"):
            os.remove(R".\PixLocation\ROI.png")

CenterXYPath = os.path.join(PixLocationRoot, "CenterXY.csv")
with open(CenterXYPath, mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerows(CenterList)

csvFileListPath = os.path.join(PixLocationRoot, "TotalFileCount.csv")
with open(csvFileListPath, mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerows(csvFileList)

data = pd.read_csv(csvFileListPath)
sorted_data = data.sort_values(by='WellNumber')
sorted_data.to_csv(csvFileListPath, index=False)
