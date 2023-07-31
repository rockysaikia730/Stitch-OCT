import cv2
import numpy as np
import tifffile as tff

#-------------------Parameters-------------------
filePath = ''
fileName = '__3d_2023-07-11_100304_raster_1000x300_structure_reduced.tif'

#For Real image
lowRange = (25,255,100) 
upperRange = (65,255,255)
#For Conjugate Image
conjlowRange = (115,255,100)
conjupperRange = (155,255,255)

#Starting frame and end frame
start = 25
end = 125
#Threshold Intensity for Noise
#All pixels with intensity <160 are made 0
NoiseThreshold = 160

#Lowe Ratio
#Higher the better but 0.7 is optimum
Loweratio = 0.7 
#Threshold to ensure that only keypoints with (delx)^2 < 30 are considered so 
DelXSquareThreshold = 30

#Save the file as
saveFileName = "withconj.png"
#---------------------------------------------------


#Read Tif file
imgPath = filePath + fileName
readTiff = tff.imread(imgPath)

#First frame
firstImg = np.array(readTiff[start,:,:],dtype=np.uint8)
firstImg = firstImg[10:-10,5:-5]
firstImg = np.where(firstImg > NoiseThreshold,firstImg,0)
#_, firstImg = cv2.threshold(firstImg, 100, 255, cv2.THRESH_BINARY)

#Second frame
secondImg = np.array(readTiff[start+1,:,:],dtype=np.uint8)
secondImg = secondImg[10:-10,5:-5]
secondImg = np.where(secondImg > NoiseThreshold,secondImg,0)
#_, secondImg = cv2.threshold(secondImg, 100, 255, cv2.THRESH_BINARY)

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
    if(delx**2 < DelXSquareThreshold):
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
    for i in range(ShiftInY,2*h):
        for j in range(w):
            result[i,j] = max(result[i,j],img1[i,j])
    return result

result = np.vstack((prevImg,np.zeros_like(prevImg)))
for i in range(start+2,end):
    flow = cv2.calcOpticalFlowFarneback(prevImg, curImg, None, 0.5, 10, 15, 3, 10, 1.2, 0)
    
    mask = np.zeros((h,w,3),dtype=np.uint8)
    mask[..., 1] = 255 #Max Saturation 
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    #This conversion loses information-Dont do it HSV to BGR
    colouredmask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    cv2.imshow("cmask",colouredmask)

    realmask = cv2.inRange(mask, lowRange,upperRange)
    final = cv2.bitwise_and(curImg,curImg,mask=realmask)
    #cv2.imshow("hf",final)
    Conjmask = cv2.inRange(mask, conjlowRange,conjupperRange)
    Conjfinal = cv2.bitwise_and(curImg,curImg,mask=Conjmask)
    flipped = cv2.flip(Conjfinal,0)

    final = np.vstack((final,flipped))
    #cv2.imshow("final",final)
    result = stitch(final,result)
    #cv2.imshow("res",result)
    #cv2.waitKey(0)
    
    prevImg = curImg
    curImg = np.array(readTiff[i,:,:],dtype=np.uint8)
    curImg = curImg[10:-10,5:-5]
    curImg = np.where(curImg > NoiseThreshold,curImg,0)
    #_, curImg = cv2.threshold(curImg, 100, 255, cv2.THRESH_BINARY)

#cv2.imshow("hfjf",result)
cv2.imwrite(saveFileName,result)
cv2.destroyAllWindows()