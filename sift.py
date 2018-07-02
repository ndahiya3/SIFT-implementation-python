import cv2
import numpy as np
import random as rand


# CS - 6475 Computational Photography, Summer 2016
# Project - 2
# Scale Invariant Image Transform (SIFT) implementation
# Distinctive Image Features from Scale-Invariants Keypoints.
# by David G. Lowe; IJCV 2004:
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

# Code implemented by Navdeep Dahiya; ndahiya3@gatech.edu; 902409566

def extractSIFTFeatures(gray, octaves, scales, sigma, sigmaN, k):
    # Extract SIFT features from input gray image
    # Octaves is number of octaves to build (optimal 4)
    # scales is number of scale levels for each octave (optimal 5)
    # sigma is initial scale (optimal 1.6)
    # nominal sigma (0.5)
    # k constant factor to multiply each scale (optimal is sqrt(2))
    r1, c1 = gray.shape

    # Build the gaussian octave
    gaussianOctaves = []
    gaussianOctaves.append(buildFirstGaussianOctave(gray, scales, sigma, sigmaN, k))

    for i in xrange(1, octaves):
        baseImage = gaussianOctaves[i - 1][2]
        row, col = baseImage.shape
        gaussianOctaves.append(buildGaussianOctave(baseImage[0:row:2, 0:col:2], scales, sigma, 0, k))

    # Build the DOG octaves from gaussian octaves
    dogOctaves = []
    for o in xrange(0, octaves):
        dogOctaves.append(buildDOGOctave(gaussianOctaves[o], scales))

    # Debug; Show all gaussian octaves
    # for i in xrange(0, octaves):
    #     print "Gaussian Octave %d size", i
    #     print gaussianOctaves[i][0].shape
    #     for j in xrange(0, scales-1):
    #         cv2.imshow('DOG octaves', dogOctaves[i][j])
    #         cv2.waitKey(0)

    # Get keypoints from each octave
    keypoints = []
    r = 10.0
    threshold = 0.03
    for o in xrange(0, octaves):
        keypoints.append(getKeyPoints(dogOctaves[o], threshold, r))

    # Resize keypoints to base octave(0) co-ordinate system
    # r1, c1 is the base octave image size
    O = []
    S = []
    for o in xrange(0, octaves):
        O.append(o - 1)
    for s in xrange(0, scales):
        S.append(np.power(k, s) * sigma)

    baseKeyPoints = []
    for o in xrange(0, octaves):  # For each octaves' Key points
        kp = keypoints[o]
        numKP = len(kp)
        mag, ori = createGradientMagnitudeandOri(gaussianOctaves[o], S)
        Y, X = gaussianOctaves[o][0].shape
        p = np.power(2.0, O[o])
        for i in xrange(0, numKP):
            currKp = kp[i]
            x = np.multiply(currKp[0], p)  # Adjust to bring to base image co-ordinate system
            y = np.multiply(currKp[1], p)
            s = currKp[2]
            if x < 0 or x > c1 - 1 or y < 0 or y > r1 - 1 or s < 0 or s > scales - 1:
                print "point coord out of range"
            sig = S[int(np.round(s))]
            # Generate Orientation histogram to assign orientation to each feature
            # Can also create new keypoints if orientation histogram has mulitple peaks > 0.8*maxPeak
            newKps = generateOrientationHistogram(mag[int(np.round(s))], ori[int(np.round(s))], sig,
                                                  int(np.round(currKp[0])), int(np.round(currKp[1])))
            for pts in xrange(0, len(newKps)):
                newKp = []
                newKp.append(x)
                newKp.append(y)
                newKp.append(p)
                newKp.append(newKps[pts])
                baseKeyPoints.append(newKp)

    return baseKeyPoints


def buildFirstGaussianOctave(gray, scales, sigma, sigmaN, k):
    # First octave creation is slightly different then the rest
    # This implementation is based on first doubling the input image
    # pre-smoothed with sigmaN
    gaussianOctave = []
    dblGray = bilinearInterpolation(gray)
    # dblGray is now doubled in size and hence means that is now
    # already pre-smoothed by 2*sigmaN
    for i in xrange(0, scales):
        desiredSigma = sigma * np.power(k, i)
        currSigma = np.sqrt(desiredSigma * desiredSigma - 2.0 * sigmaN * sigmaN)
        gaussianOctave.append(cv2.GaussianBlur(dblGray, (0, 0), currSigma))

    return gaussianOctave


def buildGaussianOctave(baseImage, scales, sigma, sigmaN, k):
    # baseImage b/w [0..1] already smoothed by sigmaN
    # Other images needs smoothing of sqrt(sigma^2 - sigmaN^2)
    # where sigma is desired smoothing
    # In practise base image is re-sampled from the previous octave's
    # image smoothed by 2*sigma. So base image is already at the desired
    # smoothness level
    # In future could make this more general and not use this assumption

    gaussianOctave = []
    gaussianOctave.append(baseImage)

    for i in xrange(1, scales):
        desiredSigma = np.power(k, i) * sigma
        currSigma = np.sqrt(desiredSigma * desiredSigma - sigmaN * sigmaN)
        gaussianOctave.append(cv2.GaussianBlur(baseImage, (0, 0), currSigma))
    return gaussianOctave


def buildDOGOctave(gaussianOctave, scales):
    # Given the gaussianOctave array with scales number of smoothed
    # images, create the corresponding Difference of Gaussian octave
    dogOctave = []

    for i in xrange(1, scales):
        dogOctave.append(np.subtract(gaussianOctave[i], gaussianOctave[i - 1]))

    return dogOctave


def createGradientMagnitudeandOri(gaussOctave, Scales):
    # Precompute gradient and orientation at each scale of
    # base gaussian octave
    row, col = gaussOctave[0].shape
    magnitudes = []
    orientations = []
    eps = 1e-10
    for k in xrange(0, len(gaussOctave)):
        mag = np.zeros((row, col), gaussOctave[0].dtype)
        ori = np.zeros((row, col), gaussOctave[0].dtype)
        for j in xrange(1, row - 1):
            for i in xrange(1, col - 1):
                dx = gaussOctave[k][j, i + 1] - gaussOctave[k][j, i - 1]
                dy = gaussOctave[k][j + 1, i] - gaussOctave[k][j - 1, i]
                mag[j, i] = np.sqrt(dx * dx + dy * dy)
                ori[j, i] = np.arctan(dy / (dx + eps))
        sigma = Scales[k]
        mag = cv2.GaussianBlur(mag, (0, 0), 1.5 * sigma)
        magnitudes.append(mag)
        orientations.append(ori)
    return magnitudes, orientations


def generateOrientationHistogram(mag, ori, sig, x, y):
    # Generate Orientation histogram from smoothed magnitude and
    # orientation images
    # Detect peak and crete new keypoints if histogram has more than
    # one peak > 0.8*max_peak

    wsize = int(2 * 1.5 * sig)
    nbins = 36
    hist = np.zeros((36, 1), mag.dtype)
    rows, cols = mag.shape
    for j in xrange(-wsize, wsize):
        for i in xrange(-wsize, wsize):
            r = y + j
            c = x + i
            if r >= 0 and r < rows and c >= 0 and c < cols:
                deg = ori[r, c] * 180.0 / 3.14159
                hist[int(deg / 10)] = hist[int(deg / 10)] + mag[r, c]
    # Find peak
    peak_loc = 0
    peak_val = hist[0]
    for k in xrange(0, nbins):
        if hist[k] > peak_val:
            peak_val = hist[k]
            peak_loc = k
    # print "peak deg: %f", peak_loc*10 + 5
    orientations = []
    orientations.append(peak_loc * 10 + 5)

    # Find other peaks greater than 0.8 * peak
    for k in xrange(0, nbins):
        if hist[k] >= 0.8 * peak_val:
            if k != peak_loc:
                orientations.append(k * 10 + 5)
    return orientations


def getKeyPoints(dogOctaves, threshold, r):
    # Given the Difference of Gaussian array calculate
    # local minima/maxima by comparing to 26 neighbors
    # at pixel level. Later these will be refined to
    # sub-pixel min/max
    # Apply edge rejection and contrast rejection thresholds
    # as well to reject unstable keypoints

    # The first and last dog's don't have enough neighbors
    # so we ignore them and only calculate min/max at pixels
    # with full set of 26 neighbors

    scales = len(dogOctaves)
    r, c = dogOctaves[0].shape
    keypoints = []
    max_iter = 5
    cnt1 = 0
    for DOG in xrange(1, scales - 1):
        currDOG = dogOctaves[DOG]
        prevDOG = dogOctaves[DOG - 1]
        nextDOG = dogOctaves[DOG + 1]
        cnt = 0
        for j in xrange(1, r - 1):
            for i in xrange(1, c - 1):
                pix = currDOG[j, i]
                prevNeighborhood = prevDOG[j - 1:j + 2, i - 1:i + 2]
                currNeighborhood = currDOG[j - 1:j + 2, i - 1:i + 2]
                nextNeighborhood = nextDOG[j - 1:j + 2, i - 1:i + 2]

                fullNeighborhood = np.zeros((3, 3, 3), currNeighborhood.dtype)
                fullNeighborhood[:, :, 0] = prevNeighborhood[:, :]
                fullNeighborhood[:, :, 1] = currNeighborhood[:, :]
                fullNeighborhood[:, :, 2] = nextNeighborhood[:, :]

                minMax = localExtrema2(fullNeighborhood)
                if minMax == 0:
                    continue
                # If we get here we have a local extrema
                cnt += 1
                ptX = i
                ptY = j
                ptZ = DOG

                neighborHood = np.zeros((3, 3, 3), currNeighborhood.dtype)
                success = 0
                for iter in xrange(0, max_iter):
                    neighborHood[:, :, 0] = prevDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    neighborHood[:, :, 1] = currDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    neighborHood[:, :, 2] = nextDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    xHat, D_xHat, H, fail = getInterpolatedMaxima(fullNeighborhood)
                    if fail == 0:
                        break
                    if np.abs(xHat[0]) <= 0.5 and np.abs(xHat[1]) <= 0.5 and np.abs(xHat[2]) <= 0.5:
                        success = 1
                        break  # We are done successfully

                    if xHat[0] > 0.5:
                        ptX += 1
                    elif xHat[0] < -0.5:
                        ptX -= 1
                    if xHat[1] > 0.5:
                        ptY += 1
                    elif xHat[1] < -0.5:
                        ptY -= 1
                    if xHat[2] > 0.5:
                        ptZ += 1
                    elif xHat[2] < -0.5:
                        ptZ -= 1

                    if ptY < 1 or ptY > r - 2:
                        break  # unsuccessful
                    if ptX < 1 or ptX > c - 2:
                        break  # unsuccessful
                    if ptZ < 1 or ptZ > scales - 2:
                        break  # unsuccessful

                if success == 1:
                    if np.abs(D_xHat) < threshold: # Contrast Threshold
                        continue
                    score = np.square(H[0, 0] + H[1, 1]) / (H[0, 0] * H[1, 1] - np.square(H[0, 1])) # Edge threshold
                    if score > (np.square(r + 1) / r):
                        continue
                    kp = []
                    kp.append(ptX + xHat[0])
                    kp.append(ptY + xHat[1])
                    kp.append(ptZ + xHat[2])
                    keypoints.append(kp)
                    cnt1 += 1
    #print cnt1
    return keypoints


def getInterpolatedMaxima(fullNeighborhood):
    # Get hessian of DOG at sample point (H)
    # Get derivative of DOG at sample point (D)
    # Solve linear system x_hat = -H^-1 * D
    # fullNeighborhood is 3x3x3 centered around
    # sample point
    # also returns value of D(xhat)
    # Also returns Hessian at current scale

    H, H1 = getHessianofDOG(fullNeighborhood)
    D = getDerivativeDOG(fullNeighborhood)
    minus_D = np.multiply(-1.0, D)
    xHat = np.zeros((3, 1), H.dtype)
    D_xhat = 0

    try:
        xHat = np.linalg.solve(H, minus_D)
        pix = fullNeighborhood[1, 1, 1]
        D_xhat = pix + 0.5 * (D[0] * xHat[0] + D[1] * xHat[1] + D[2] * xHat[2])
        return xHat, D_xhat, H1, 1
    except:
        np.linalg.LinAlgError
        return xHat, D_xhat, H1, 0


def getHessianofDOG(neighborhood):
    # DOG is 3 dimensional function of (x,y,sigma)
    # Input is 3 dimensional neighborhood centered at
    # the sample point where we are calculating Hessian
    i = 1
    j = 1
    sigma = 1
    D2_x2 = neighborhood[j, i + 1, sigma] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j, i - 1, sigma]
    D2_y2 = neighborhood[j + 1, i, sigma] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j - 1, i, sigma]
    D2_sigma2 = neighborhood[j, i, sigma + 1] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j, i, sigma - 1]
    D2_x_y = neighborhood[j + 1, i + 1, sigma] - neighborhood[j - 1, i + 1, sigma] - neighborhood[j + 1, i - 1, sigma] + \
             neighborhood[j - 1, i - 1, sigma]
    D2_x_y /= 4.0
    D2_x_sigma = neighborhood[j, i + 1, sigma + 1] - neighborhood[j, i + 1, sigma - 1] - neighborhood[
        j, i - 1, sigma + 1] + neighborhood[j, i - 1, sigma - 1]
    D2_x_sigma /= 4.0
    D2_y_sigma = neighborhood[j + 1, i, sigma + 1] - neighborhood[j + 1, i, sigma - 1] - neighborhood[
        j - 1, i, sigma + 1] + neighborhood[j - 1, i, sigma - 1]
    D2_y_sigma /= 4.0

    Hessian = np.zeros((3, 3), neighborhood.dtype)
    Hessian[0, 0] = D2_x2
    Hessian[0, 1] = D2_x_y
    Hessian[0, 2] = D2_x_sigma

    Hessian[1, 0] = D2_x_y
    Hessian[1, 1] = D2_y2
    Hessian[1, 2] = D2_y_sigma

    Hessian[2, 0] = D2_x_sigma
    Hessian[2, 1] = D2_y_sigma
    Hessian[2, 2] = D2_sigma2

    Hessian_CurrentScale = np.zeros((2, 2), Hessian.dtype)
    Hessian_CurrentScale[0, 0] = D2_x2
    Hessian_CurrentScale[0, 1] = D2_x_y
    Hessian_CurrentScale[1, 0] = D2_x_y
    Hessian_CurrentScale[1, 1] = D2_y2

    return Hessian, Hessian_CurrentScale


def getDerivativeDOG(neighborhood):
    # DOG is 3-d function (matrix) centered around sample point
    # We need Derivative of DOG wrt x,y,sigma at sample point
    # Using central difference
    i = 1
    j = 1
    sigma = 1
    Dx = (neighborhood[j, i + 1, sigma] - neighborhood[j, i - 1, sigma]) / 2.0
    Dy = (neighborhood[j + 1, i, sigma] - neighborhood[j - 1, i, sigma]) / 2.0
    Dsigma = (neighborhood[j, i, sigma + 1] - neighborhood[j, i, sigma - 1]) / 2.0
    D = np.zeros((3, 1), neighborhood.dtype)
    D[0] = Dx
    D[1] = Dy
    D[2] = Dsigma

    return D


def localExtrema1(n):
    isExtrema = 1
    pix = n[1, 1, 1]
    if pix >= 0:
        for i in xrange(0, 3):
            for j in xrange(0, 3):
                for k in xrange(0, 3):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if pix < n[i, j, k]:
                        isExtrema = 0
    else:
        for i in xrange(0, 3):
            for j in xrange(0, 3):
                for k in xrange(0, 3):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if pix > n[i, j, k]:
                        isExtrema = 0
    return isExtrema


def localExtrema(pix, neighborhood):
    lessThan = 0
    greaterThan = 0
    minMax = 1
    for k in xrange(0, 3):
        for l in xrange(0, 3):
            if greaterThan == 1 and lessThan == 1:
                minMax = 0
                break
            if pix >= neighborhood[k, l]:
                greaterThan = 1
            else:
                lessThan = 1

    return minMax


def localExtrema2(n):
    # This function is used in implementation
    # Given 3x3x3 neighborhood of (X,Y,Ssigma)
    # Compare central pixel to 26 neighbors and
    # detect if this is extrema either greater than
    # all neighbors or less than all neighbors
    pix = n[1, 1, 1]
    lessThan = 0
    greaterThan = 0
    isExtrema = 1
    numEq = 0

    for i in xrange(0, 3):
        if isExtrema == 0:
            break
        for j in xrange(0, 3):
            if isExtrema == 0:
                break
            for k in xrange(0, 3):
                if lessThan == 1 and greaterThan == 1:
                    isExtrema = 0
                    break
                if i == 1 and j == 1 and k == 1:
                    continue
                if pix >= n[i, j, k]:
                    greaterThan = 1
                elif pix <= n[i, j, k]:
                    lessThan = 1
                else:
                    numEq += 1

    if numEq == 26:
        print "All same"
        isExtrema = 0

    return isExtrema


def bilinearInterpolation(gray):
    # Double the input image with bilinear interpolation in both dimensions
    # Input image is assumed floating point b/w [0..1]
    r, c = gray.shape
    r1 = 2 * r
    c1 = 2 * c
    dest = np.zeros((r1, c1), gray.dtype)
    expanded = np.zeros((r + 2, c + 2), gray.dtype)
    expanded[1:r + 1, 1:c + 1] = gray[:, :]

    for j in xrange(1, r1 - 1):
        for i in xrange(1, c1 - 1):
            j1 = j / 2.0
            i1 = i / 2.0
            delY = j1 - int(j1)
            delX = i1 - int(i1)

            temp1 = (1.0 - delX) * expanded[int(j1), int(i1)] + delX * expanded[int(j1), int(i1) + 1]
            temp2 = (1.0 - delX) * expanded[int(j1) + 1, int(i1)] + delX * expanded[int(j1) + 1, int(i1) + 1]
            dest[j, i] = (1.0 - delY) * temp1 + delY * temp2

    return dest


# Test the routines
fileName = input('Please enter filename to process: ')
img = cv2.imread(fileName,cv2.IMREAD_COLOR)

gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.array(gray1.shape, np.float32)
gray = np.divide(gray1, 255.0)

gray = cv2.GaussianBlur(gray, (0, 0), 0.5)

k = np.sqrt(2.0)
keypoints = extractSIFTFeatures(gray, 4, 5, 1.6, 0.5, k)

for i in xrange(0, len(keypoints)):
    kp = keypoints[i]
    x = kp[0]
    y = kp[1]
    s = kp[2]
    # cv2.circle(img,(int(np.round(x)),int(np.round(y))),int(5*s),(0,0,255))
    cv2.ellipse(img, (int(np.round(x)), int(np.round(y))), (int(6 * s), int(4 * s)), kp[3], 0, 360,
                (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)))
print len(keypoints)
cv2.imshow('SIFT Keypoints', img)
cv2.waitKey(0)

cv2.imwrite('result.jpg', img)

print "Saving result in.. result.jpg"
print "Done!"
