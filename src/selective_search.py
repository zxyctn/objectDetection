import cv2 as cv 

def selectiveSearch(img, quality, noOfRects):
    cv.setUseOptimized(True)
    cv.setNumThreads(4)
 
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(img)
 
    # check whether to use Fast Selective Search or Quality
    if (quality == 'f'):
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
 
    # run selective search segmentation on input image
    rects = ss.process()

    return rects[:noOfRects]