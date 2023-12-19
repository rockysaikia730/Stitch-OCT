import cv2
import numpy as np
import tifffile as tff

#-------------------------------------------------------------------------
##Change the file path and file name
filePath = '2023-12-14_133313_raster_4000x120/'#'2023-12-14_133313_raster_4000x120/'
fileName = "_StackStrRaw_masked.tif"
#-------------------------------------------------------------------------


imgPath = filePath + fileName
readTiff = tff.imread(imgPath)

firstImg = np.array(readTiff[0,:,:],dtype=np.uint8)
firstImg = firstImg[10:-10,5:-5]

secondImg = np.array(readTiff[1,:,:],dtype=np.uint8)
secondImg = secondImg[10:-10,5:-5]

Loweratio = 0.7 
DelXSquareThreshold = 30
NoiseThreshold = 50


h,w = firstImg.shape
flipShift = 10

sift = cv2.SIFT_create()
kp1,desc1 = sift.detectAndCompute(firstImg[h//2+flipShift:,:],None)
kp2,desc2 = sift.detectAndCompute(secondImg[h//2+flipShift:,:],None)

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

#sitch
aux = np.zeros((ShiftInY,w),dtype=np.uint8)
def stitch(img1,img2):
    result = np.array(img2,dtype=np.uint8)
    #print(img2.shape)
    result = np.vstack((aux,result))
    #result[:ShiftInY,:] = img1[:ShiftInY,:]
    newh,_ = img1.shape
    result[:newh,:] = np.maximum(img1,result[:newh,:])
    return result

start = 1
end = 57
result = firstImg

for i in range(start,end):
    curImg = np.array(readTiff[i,:,:],dtype=np.uint8)
    curImg = curImg[10:-10,5:-5]
    result = stitch(curImg,result)

cv2.imwrite("withoutconj.png",result)
cv2.destroyAllWindows()
