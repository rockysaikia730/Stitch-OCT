from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog,QLineEdit
import sys
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette,QPixmap, QImage,QFont
from PyQt5.QtCore import Qt, QUrl,QObject, QThread, pyqtSignal,QRect
import cv2
import tifffile as tff
import numpy as np
from time import sleep
import datetime
from PIL import Image,ImageOps
import pyautogui

globalFilename = {"fileName":"", "currentFrame":0, "maxFrame":0}
#DOF
StartDOF = 45
EndDOF = 299
ThresholdNoiseDOF = 100
Loweratio = 0.7
DelXSquareThreshold = 30

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal()

    def run(self):
        rangeValue = globalFilename["maxFrame"] - globalFilename["currentFrame"]
        self.isRunning = True
        for i in range(rangeValue):
            if(self.isRunning == False):
                break
            globalFilename["currentFrame"] +=1
            sleep(0.03)
            self.progress.emit()
        
        self.finished.emit()
    
    def stop(self):
        self.isRunning = False

class DOFStitch(QObject):
    finished = pyqtSignal()
    def run(self):
        imgPath = globalFilename["fileName"]
        self.readTiff = tff.imread(imgPath)

        firstImg = np.array(self.readTiff[StartDOF,:,:],dtype=np.uint8)
        firstImg = firstImg[10:-10,5:-5]
        firstImg = np.where(firstImg > ThresholdNoiseDOF,firstImg,0)

        secondImg = np.array(self.readTiff[StartDOF+1,:,:],dtype=np.uint8)
        secondImg = secondImg[10:-10,5:-5]
        secondImg = np.where(secondImg > ThresholdNoiseDOF,secondImg,0)
    
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

        delyList = []
        totalIterations = len(src_pts)
        #To remove further outliers, used another thresholding 
        for i in range(totalIterations):
            delx = src_pts[i][0] - dst_pts[i][0]
            dely = src_pts[i][1] - dst_pts[i][1]
            if(delx**2 < DelXSquareThreshold):
                delyList.append(dely)

        self.ShiftInY = abs(int(np.rint(np.average(delyList))))
        self.prevImg = firstImg
        self.curImg = secondImg
        self.h,self.w = firstImg.shape

        self.result = np.vstack((self.prevImg,np.zeros_like(self.prevImg)))

        self.aux = np.zeros((self.ShiftInY,self.w),dtype=np.uint8) 
        self.stitchAllFrame()

    def stitch(self,img1,img2):
        result = np.array(img2,dtype=np.uint8)
        #print(img2.shape)
        result = np.vstack((self.aux,result))
        result[:self.ShiftInY,:] = img1[:self.ShiftInY,:]
        for i in range(self.ShiftInY,2*self.h):
            for j in range(self.w):
                result[i,j] = max(result[i,j],img1[i,j])
        return result

    def stitchAllFrame(self):
        prevImg = self.prevImg
        curImg = self.curImg
        for i in range(StartDOF+2,EndDOF):
            flow = cv2.calcOpticalFlowFarneback(prevImg, curImg, None, 0.5, 10, 15, 3, 10, 1.2, 0)
            
            mask = np.zeros((self.h,self.w,3),dtype=np.uint8)
            mask[..., 1] = 255 #Max Saturation 
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            #This conversion loses information-Dont do it HSV to BGR
            #mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            lowRange = (25,255,100)
            upperRange = (65,255,255)
            realmask = cv2.inRange(mask, lowRange,upperRange)
            final = cv2.bitwise_and(curImg,curImg,mask=realmask)
            #cv2.imshow("hf",final)
            conjlowRange = (115,255,100)
            conjupperRange = (155,255,255)
            Conjmask = cv2.inRange(mask, conjlowRange,conjupperRange)
            Conjfinal = cv2.bitwise_and(curImg,curImg,mask=Conjmask)
            flipped = cv2.flip(Conjfinal,0)

            final = np.vstack((final,flipped))
            self.result = self.stitch(final,self.result)
            
            prevImg = curImg
            curImg = np.array(self.readTiff[i,:,:],dtype=np.uint8)
            curImg = curImg[10:-10,5:-5]
            curImg = np.where(curImg > 100,curImg,0)
        cv2.imwrite("DOF" + str(datetime.datetime.now()).split(".")[0] +".png",self.result)
        self.finished.emit()

DOFDisplayStart = 45
DOFDisplayEnd = 250
DOFDisplayThresholdValue = 100

DOFDisplayFrame = []

class DOFDisplay(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal()

    def run(self):
        self.readTiff = tff.imread(globalFilename["fileName"])

        self.prevImg = np.array(self.readTiff[DOFDisplayStart,:,:],dtype=np.uint8)
        self.prevImg = self.prevImg[10:-10,5:-5]
        self.prevImg = np.where(self.prevImg > DOFDisplayThresholdValue,self.prevImg,0)

        for i in range(DOFDisplayStart+1,DOFDisplayEnd):

            curImg = np.array(self.readTiff[i,:,:],dtype=np.uint8)
            curImg = curImg[10:-10,5:-5]
            curImg = np.where(curImg > DOFDisplayThresholdValue,curImg,0)  
        
            flow = cv2.calcOpticalFlowFarneback(self.prevImg, curImg, None, 0.5, 10, 15, 3, 10, 1.2, 0)
        
            mask = np.zeros((curImg.shape[0],curImg.shape[1],3),dtype=np.uint8)
            mask[..., 1] = 255 #Max Saturation 
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            colourerdmask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            #Conj image
            conjlowRange = (100,255,100)
            conjupperRange = (170,255,255)
            Conjmask = cv2.inRange(mask, conjlowRange,conjupperRange)
            Conjfinal = cv2.bitwise_and(curImg,curImg,mask=Conjmask)
            
            #Real Image
            lowRange = (20,255,100)
            upperRange = (90,255,255)
            mask = cv2.inRange(mask, lowRange,upperRange)
            final = cv2.bitwise_and(curImg,curImg,mask=mask)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (100, 100)
            fontScale = 1.4
            color = (255, 0, 0)
            thickness = 2

            #for display
            finalDisplay = cv2.cvtColor(final,cv2.COLOR_GRAY2RGB)
            finalDisplay = cv2.putText(finalDisplay, 'Real Image', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            finalDisplay =cv2.copyMakeBorder(finalDisplay, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))


            ConjfinalDisplay = cv2.cvtColor(Conjfinal,cv2.COLOR_GRAY2RGB)
            ConjfinalDisplay = cv2.putText(ConjfinalDisplay, 'Conjugate Image', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            ConjfinalDisplay =cv2.copyMakeBorder(ConjfinalDisplay, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))

            curImgDisplay = cv2.cvtColor(curImg,cv2.COLOR_GRAY2RGB)
            curImgDisplay = cv2.putText(curImgDisplay, 'Original Frame', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            curImgDisplay =cv2.copyMakeBorder(curImgDisplay, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))

            colourerdmaskImg = cv2.putText(colourerdmask, 'Mask', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            colourerdmaskImg = cv2.copyMakeBorder(colourerdmaskImg, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))
            
            Vstack1 = np.vstack((curImgDisplay,colourerdmaskImg))
            Vstack2 = np.vstack((finalDisplay,ConjfinalDisplay))
            HstackFinal = np.hstack((Vstack1,Vstack2))
            
            global DOFDisplayFrame
            img = Image.fromarray(HstackFinal.astype('uint8'), 'RGB')
            DOFDisplayFrame = img
            self.progress.emit()
            self.prevImg = curImg
            #sleep(0.03)
        #cv2.destroyAllWindows()
        self.finished.emit()


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BOELab")
        self.setGeometry(350, 100, 700, 500)
        #self.setMaximumWidth(1000)
        #self.setWindowIcon(QIcon('flag.png'))

        p =self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()
        self.show()

    def init_ui(self):
        #Displays Image
        self.Imagelabel = QLabel()

        #create open button
        openBtn = QPushButton('Open File')
        openBtn.clicked.connect(self.open_file)

        #create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        self.pauseBtn = QPushButton()
        self.pauseBtn.setEnabled(False)
        self.pauseBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseBtn.clicked.connect(self.pause_video)

        self.nextBtn = QPushButton()
        self.nextBtn.setEnabled(False)
        self.nextBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.nextBtn.clicked.connect(self.next_Image)

        self.prvBtn = QPushButton()
        self.prvBtn.setEnabled(False)
        self.prvBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.prvBtn.clicked.connect(self.prv_Image)

        self.DOFStitchBtn = QPushButton()
        self.DOFStitchBtn.setEnabled(False)
        self.DOFStitchBtn.setText("Stitch using DOF")
        self.DOFStitchBtn.clicked.connect(self.dofStitch_Image)

        self.textbox = QLineEdit(self)
        self.textbox.setText(str(StartDOF))
        self.textbox.setReadOnly(True)

        self.fromBox = QPushButton()
        #self.fromBox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.fromBox.setText("From Frame no")
        self.fromBox.setStyleSheet("background-color: lightgreen")
        self.fromBox.adjustSize()
        self.fromBox.clicked.connect(self.activateLine)

        self.LoweRatio = QPushButton()
        self.LoweRatio.setText("Lowe Ratio")
        self.LoweRatio.setStyleSheet("background-color: lightgreen")
        self.LoweRatio.adjustSize()
        self.LoweRatio.clicked.connect(self.activateLine)

        self.NoiseThresh = QPushButton()
        self.NoiseThresh.setText("Noise Threshold")
        self.NoiseThresh.setStyleSheet("background-color: lightgreen")
        self.NoiseThresh.adjustSize()
        self.NoiseThresh.clicked.connect(self.activateLine)

        self.DelxTresh = QPushButton()
        self.DelxTresh.setText("DelX Threshold")
        self.DelxTresh.setStyleSheet("background-color: lightgreen")
        self.DelxTresh.adjustSize()
        self.DelxTresh.clicked.connect(self.activateLine)

        self.textbox2 = QLineEdit(self)
        self.textbox2.setText(str(EndDOF))
        self.textbox2.setReadOnly(True)

        self.Lowetext = QLineEdit(self)
        self.Lowetext.setText(str(Loweratio))
        self.Lowetext.setReadOnly(True)

        self.Noisetext = QLineEdit(self)
        self.Noisetext.setText(str(ThresholdNoiseDOF))
        self.Noisetext.setReadOnly(True)

        self.DelXtext = QLineEdit(self)
        self.DelXtext.setText(str(DelXSquareThreshold))
        self.DelXtext.setReadOnly(True)

        self.ToBox = QPushButton()
        #self.ToBox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.ToBox.setText("To Frame no")
        self.ToBox.setStyleSheet("background-color: lightgreen")
        self.ToBox.adjustSize()
        self.ToBox.clicked.connect(self.activateLine)

        self.dofDisplayBtn = QPushButton()
        self.dofDisplayBtn.setEnabled(False)
        self.dofDisplayBtn.setText("Display DOF")
        self.dofDisplayBtn.clicked.connect(self.dofDisplay_Image)

        #create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.label.setText("Remark :\n")
        self.label.setStyleSheet("background-color: lightgreen")

        self.framelabel = QLabel()
        self.framelabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.framelabel.setStyleSheet("background-color: lightgreen")

        self.helpLabel = QLabel()
        self.helpLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.helpLabel.setStyleSheet("background-color: lightgreen")
        self.helpLabel.setFont(QFont('Times', 10))
        helptext = """Help:\n 
        1) Use open file to open any TIFF image\n
        2) Use A and D to manually scroll the video\n
        3) Press Spacebar to play and pause video\n
        4) Click on From Frame No(or To Frame No) to set the starting an ending of Stitching\n
        5) Stitched image is saved in 'DOF+TimeStamp' in png format in the same directory\n
        6) Press Display DOF to visualise the Dense optical flow tracking of the image"""
        self.helpLabel.setText(helptext)
        self.helpLabel.adjustSize()
        self.label.setMaximumWidth(400)
        self.framelabel.setMaximumWidth(400)
        self.helpLabel.setMaximumWidth(400)

        self.vboxText = QVBoxLayout()

        self.hboxText1 = QHBoxLayout()
        self.hboxText1.addWidget(self.fromBox)
        self.hboxText1.addWidget(self.textbox)

        self.hboxText2 = QHBoxLayout()
        self.hboxText2.addWidget(self.ToBox)
        self.hboxText2.addWidget(self.textbox2)

        self.hboxText3 = QHBoxLayout()
        self.hboxText3.addWidget(self.LoweRatio)
        self.hboxText3.addWidget(self.Lowetext)

        self.hboxText4 = QHBoxLayout()
        self.hboxText4.addWidget(self.NoiseThresh)
        self.hboxText4.addWidget(self.Noisetext)

        self.hboxText5 = QHBoxLayout()
        self.hboxText5.addWidget(self.DelxTresh)
        self.hboxText5.addWidget(self.DelXtext)

        HboxLayout2 = QHBoxLayout()
        HboxLayout2.addWidget(self.DOFStitchBtn)
        HboxLayout2.addWidget(self.dofDisplayBtn)

        self.vboxText.addLayout(HboxLayout2)
        self.vboxText.addLayout(self.hboxText1)
        self.vboxText.addLayout(self.hboxText2)
        self.vboxText.addLayout(self.hboxText3)
        self.vboxText.addLayout(self.hboxText4)
        self.vboxText.addLayout(self.hboxText5)
        #create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0,0,0,0)

        #set widgets to the hbox layout
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.prvBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.pauseBtn)
        hboxLayout.addWidget(self.nextBtn)
        
        
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        vbox.addLayout(hboxLayout)
        vbox.addLayout(self.vboxText)
        vbox.addWidget(self.framelabel)
        vbox.addWidget(self.label)
        vbox.addWidget(self.helpLabel)
        #create vbox layout
        vboxLayout = QHBoxLayout()
        
        vboxLayout.addWidget(self.Imagelabel)
        vboxLayout.addLayout(vbox)
        

        self.setLayout(vboxLayout)

    #Open Done
    def open_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open File")
        if self.filename != '':
            try:
                self.readTiff = tff.imread(self.filename)
                #Important Parameters
                self.frameNo = 0
                self.No_of_Frames, _,_ =  self.readTiff.shape

                image = np.array(self.readTiff[0,:,:],dtype=np.uint8)
                qimg = QImage(image,image.shape[1],image.shape[0],image.shape[1],QImage.Format_Indexed8)
                self.image = QPixmap(qimg)
                self.Imagelabel.setPixmap(self.image)

                globalFilename["fileName"] = self.filename
                globalFilename["currentFrame"] = self.frameNo
                globalFilename["maxFrame"] = self.No_of_Frames - 1
                

                self.playBtn.setEnabled(True)
                self.nextBtn.setEnabled(True)
                self.prvBtn.setEnabled(True)
                self.DOFStitchBtn.setEnabled(True)
                self.dofDisplayBtn.setEnabled(True)

                self.framelabel.setText("Frame : "+str(self.frameNo) + "/" + str(self.No_of_Frames-1))
                self.textbox2.setText(str(globalFilename["maxFrame"]))
                self.textbox.setText("0")
            except Exception as e:
                #Add Error
                self.label.setText("Remark:\nError in opening File")
                pass

    def activateLine(self):
        self.textbox.setReadOnly(False)
        self.textbox2.setReadOnly(False)
        self.DelXtext.setReadOnly(False)
        self.Noisetext.setReadOnly(False)
        self.Lowetext.setReadOnly(False)

    def next_Image(self):
        self.frameNo += 1
        if(self.No_of_Frames ==  self.frameNo):
            self.frameNo -= 1
            self.label.setText("Remark:\n" + "No more Frames")
            #Add Error Message
        else:
            image = np.array(self.readTiff[self.frameNo,:,:],dtype=np.uint8)
            qimg = QImage(image,image.shape[1],image.shape[0],image.shape[1],QImage.Format_Indexed8)
            self.image = QPixmap(qimg)
            self.Imagelabel.setPixmap(self.image)
            self.framelabel.setText("Frame : "+str(self.frameNo) + "/" + str(self.No_of_Frames-1))
        
    def prv_Image(self):
        self.frameNo -= 1
        if(self.frameNo == -1):
            self.frameNo = 0
            #Add Error Message
            self.label.setText("Remark:\n" + "No more Frames")
        else:
            image = np.array(self.readTiff[self.frameNo,:,:],dtype=np.uint8)
            qimg = QImage(image,image.shape[1],image.shape[0],image.shape[1],QImage.Format_Indexed8)
            self.image = QPixmap(qimg)
            self.Imagelabel.setPixmap(self.image)
            self.framelabel.setText("Frame : "+str(self.frameNo) + "/" + str(self.No_of_Frames-1))

    def dofDisplay_Image(self):
        global DOFDisplayStart,DOFDisplayEnd
        DOFDisplayStart = int(self.textbox.text())
        DOFDisplayEnd = int(self.textbox2.text())

        self.thread2 = QThread()
        self.worker2 = DOFDisplay()
        self.worker2.moveToThread(self.thread2)

        self.thread2.started.connect(self.worker2.run)
        self.worker2.finished.connect(self.thread2.quit)
        self.worker2.finished.connect(self.EnableDOFBtn)
        self.worker2.finished.connect(self.worker2.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
    
        self.worker2.progress.connect(self.updateDOFFrame)
        self.thread2.start()
        self.dofDisplayBtn.setEnabled(False) 
        self.label.setText("Remark:\n" + "Visualisation of Dense Optical Flow Tracking")

    def EnableDOFBtn(self):
        self.next_Image()
        self.prv_Image()
        self.dofDisplayBtn.setEnabled(True)  

    def updateDOFFrame(self):
        global DOFDisplayFrame
        pilH, pilW = DOFDisplayFrame.size
        _, sH = pyautogui.size()
        if(pilH > sH):
            DOFDisplayFrame = DOFDisplayFrame.resize((sH-100,int(pilW/pilH * (sH-100))))
        
        image = np.array(DOFDisplayFrame)
        #self.mQImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)
        qimg = QImage(image,image.shape[1],image.shape[0],image.shape[1]*3,QImage.Format_RGB888)
        self.image = QPixmap(qimg)
        self.Imagelabel.setPixmap(self.image)
        self.Imagelabel.show()

    def play_video(self):
        self.playBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        #Thread
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        global globalFilename
        globalFilename["maxFrame"] = self.No_of_Frames-1
        globalFilename["currentFrame"] = self.frameNo
        globalFilename["fileName"] = self.filename

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.MakeButtonsRight)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
    
        self.worker.progress.connect(self.next_Image)
        self.thread.start()

    def MakeButtonsRight(self):
        self.playBtn.setEnabled(True)
        self.pauseBtn.setEnabled(False)

    def pause_video(self):
        self.playBtn.setEnabled(True)
        self.pauseBtn.setEnabled(False)
        self.worker.stop()
        self.thread.quit()
        #self.thread.wait()

    def keyPressEvent(self, event):
        if self.nextBtn.isEnabled():
            #Next Image
            if event.key() == 68:
                self.next_Image()

            #Previous Image
            if event.key() == 65:
                self.prv_Image()

            #Play using spacebar
            if event.key() == 32:
                if(self.playBtn.isEnabled()):
                    self.play_video()
                else:
                    self.pause_video()

    def dofStitch_Image(self):
        global StartDOF,EndDOF
        StartDOF = int(self.textbox.text())
        EndDOF = int(self.textbox2.text())
        
        self.thread1 = QThread()
        self.worker1 = DOFStitch()
        self.worker1.moveToThread(self.thread1)

        self.thread1.started.connect(self.worker1.run)
        self.worker1.finished.connect(self.thread1.quit)
        self.worker1.finished.connect(self.ResetText)
        self.worker1.finished.connect(self.worker1.deleteLater)
        self.thread1.finished.connect(self.thread1.deleteLater)

        self.thread1.start()
        self.DOFStitchBtn.setText("DOF Stitching in Progress")
        self.DOFStitchBtn.blockSignals(True)

    def ResetText(self):
        self.DOFStitchBtn.setText("Stitch using DOF")
        self.DOFStitchBtn.blockSignals(False)
        self.label.setText("Remark:\n" + "Dense Optical Flow Stitch Completed")

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: ")

app = QApplication(sys.argv)
app.setWindowIcon(QIcon('logo.png'))
window = Window()

sys.exit(app.exec_())