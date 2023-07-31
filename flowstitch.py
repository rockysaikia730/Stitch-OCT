import cv2
import numpy as np
import tifffile as tff

#Read Tif file
filePath = '' #File Path
fileName = '__3d_2023-07-11_100304_raster_1000x300_structure_reduced.tif' #File Name
imgPath = filePath + fileName
readTiff = tff.imread(imgPath)

#Starting frame and end frame
start = 60
end = 240

#First frame
firstImg = np.array(readTiff[start,:,:],dtype=np.uint8)
firstImg = firstImg[10:-10,5:-5]
firstImg = np.where(firstImg > 100,firstImg,0)
#_, firstImg = cv2.threshold(firstImg, 100, 255, cv2.THRESH_BINARY)

#Second frame
secondImg = np.array(readTiff[start+1,:,:],dtype=np.uint8)
secondImg = secondImg[10:-10,5:-5]
secondImg = np.where(secondImg > 100,secondImg,0)
#_, secondImg = cv2.threshold(secondImg, 100, 255, cv2.THRESH_BINARY)


Loweratio = 0.7 

#Get matching features and detect shift in Y
sift = cv2.SIFT_create()
kp1,desc1 = sift.detectAndCompute(firstImg,None)
kp2,desc2 = sift.detectAndCompute(secondImg,None)

bf = cv2.BFMatcher()
rmatches = bf.knnMatch(desc1,desc2,k=2)
matches = []

#Remove noises through Lowe test
for m in rmatches:
    if(len(m) == 2 and m[0].distance < m[1].distance*Loweratio):
        matches.append(m[0])

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,2)

delxList = []
delyList = []
totalIterations = len(src_pts)
#To remove further outliers, used another thresholding 
for i in range(totalIterations):
    delx = src_pts[i][0] - dst_pts[i][0]
    dely = src_pts[i][1] - dst_pts[i][1]
    if(delx**2 < 30):
        delyList.append(dely)

ShiftInY = abs(int(np.rint(np.average(delyList))))
print(ShiftInY)

#Generalizing the name
prevImg = firstImg
curImg = secondImg
h,w = firstImg.shape

aux = np.zeros((ShiftInY,w),dtype=np.uint8)
def stitch(img1,img2):
    result = np.array(img2,dtype=np.uint8)
    #print(img2.shape)
    result = np.vstack((aux,result))
    result[:ShiftInY,:] = img1[:ShiftInY,:]
    for i in range(ShiftInY,h):
        for j in range(w):
            result[i,j] = max(result[i,j],img1[i,j])
    return result

result = prevImg
for i in range(start+2,end):
    flow = cv2.calcOpticalFlowFarneback(prevImg, curImg, None, 0.5, 10, 15, 3, 10, 1.2, 0)
    
    mask = np.zeros((h,w,3),dtype=np.uint8)
    mask[..., 1] = 255 #Max Saturation 
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    #This conversion loses information-Dont do it HSV to BGR
    #mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    lowRange = (20,255,100)
    upperRange = (90,255,255)
    mask = cv2.inRange(mask, lowRange,upperRange)
    final = cv2.bitwise_and(curImg,curImg,mask=mask)
    #cv2.imshow("hf",final)
    result = stitch(final,result)
    
    prevImg = curImg
    curImg = np.array(readTiff[i,:,:],dtype=np.uint8)
    curImg = curImg[10:-10,5:-5]
    curImg = np.where(curImg > 100,curImg,0)
    #_, curImg = cv2.threshold(curImg, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("hfjf",result)

cv2.waitKey(0)
cv2.destroyAllWindows()