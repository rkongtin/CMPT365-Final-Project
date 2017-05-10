from Tkinter import *
from tkFileDialog import askopenfilename
import numpy as np
import cv2
import imageio
import math


#mean absolute difference(MAD)
def MAD(i, j, x, y, cur, prev):
    N = 16
    summation = 0
    total = np.float32(0)
    for k in range(0, N):
        for l in range(0, N):
            summation += np.abs(np.int16(cur[x + k, y + l]) - np.int16(prev[x + i + k, y + j + l]))
    total = (1/np.float32(pow(N, 2))) * summation
    return total

last = False
#sequential search for the motion vector
def hierarchical_search(p, x, y, cur, prev):
    global last
    u, v = (0, 0)
    min_MAD = np.float32(999999)
    offset = int(math.ceil(np.float32(p)/2))
    while last != True:
        for i in range(-p/2, p, offset):
            for j in range(-p/2, p, offset):
                cur_MAD = MAD(i, j, x, y, cur, prev)
                if (cur_MAD < min_MAD):
                    min_MAD = cur_MAD
                    u = i
                    v = j
        if offset == 1:
            last = True
        a, b = hierarchical_search(offset, x + u, y + v, cur, prev)
        u += a
        v += b
    return u, v


#read the images for the animation
character = []
for i in range(1, 9):
    character.append(cv2.imread("Character-red-shirt%d.jpg" % i))

#make the background of the images black
lower = np.array([235, 235, 235], dtype = np.uint8)
upper = np.array([255, 255, 255], dtype = np.uint8)
for i in range(8):
    mask = cv2.inRange(character[i], lower, upper)
    character[i] = cv2.bitwise_not(character[i], character[i], mask)

#get the size of the images for the animation
row, col, channel = character[0].shape


#ask to open the video file
filename = askopenfilename()

#read the images for the video and process them
vid = imageio.get_reader(filename, 'ffmpeg')
no_of_frames = vid._meta['nframes'];
count = 0
frame = vid.get_data(count)
height, width, channels = frame.shape
prev = np.zeros((height, width), np.uint8)
u, v = (0, 0)
MV_u_prev = np.zeros((height/16, width/16), np.float32)
MV_v_prev = np.zeros((height/16, width/16), np.float32)
MVD_u = np.zeros((height/16, width/16), np.float32)
MVD_v = np.zeros((height/16, width/16), np.float32)
f = open('output.mrg', 'w+')
while(count < no_of_frames):
    frame = vid.get_data(count)
    count += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #determine where to place the image for the animation on the video
    merged_part = frame[50:row+50, 0:col]
    
    #make a mask of the sprite image and the inverse of the mask
    bgr2gray = cv2.cvtColor(character[count%8], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(bgr2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    #make the background and foreground
    background = cv2.bitwise_and(merged_part, merged_part, mask = mask_inv)
    foreground = cv2.bitwise_or(character[count%8], character[count%8], mask = mask)
    
    #merge the background and foreground
    merging = cv2.add(background, foreground)
    frame[50:row+50, 0:col] = merging
    
    
    # --- Compression ---
    #transform from RGB to YCbCr
    img_YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    rows, cols, channels = img_YCbCr.shape
    Y = np.zeros((rows, cols), np.float32)
    cb = np.zeros((rows, cols), np.float32)
    cr = np.zeros((rows, cols), np.float32)
    Y[:rows, :cols] = img_YCbCr[:,:,0]
    cb[:rows, :cols] = img_YCbCr[:,:,1]
    cr[:rows, :cols] = img_YCbCr[:,:,2]
    
    #encode I-frames and P-frames
    MV_u = np.zeros((rows/16, cols/16), np.float32)
    MV_v = np.zeros((rows/16, cols/16), np.float32)
    if (count != 1 and count % 4 != 0):
        #find motion vectors on P-frames
        for x in range(0, height - 16, 16):
            for y in range(0, width - 16, 16):
                last = False
                u, v = hierarchical_search(15, x, y, Y, prev)
                MV_u[x/16, y/16] = u
                MV_v[x/16, y/16] = v
        
        #calculate the difference, MVD, between preceding and current motion vector
        for x in range(0, height - 16, 16):
            for y in range(0, width - 16, 16):
                MVD_u[x/16][y/16] = MV_u_prev[x/16][y/16] - MV_u[x/16][y/16]
                MVD_v[x/16][y/16] = MV_v_prev[x/16][y/16] - MV_v[x/16][y/16]
        MV_u_prev = MV_u
        MV_v_prev = MV_v
        
        #do 4:2:0 chroma subsampling
        cb_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
        cr_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
        for i in np.arange(0, (rows/2)):
            for j in np.arange(0, (cols/2)):
                cb_chroma[i][j] = (cb[(2*i)][(2*j)] + cb[(2*i) + 1][(2*j)] + cb[(2*i)][(2*j) + 1] + cb[(2*i) + 1][(2*j) + 1])/4
                cr_chroma[i][j] = (cr[(2*i)][(2*j)] + cr[(2*i) + 1][(2*j)] + cr[(2*i)][(2*j) + 1] + cr[(2*i) + 1][(2*j) + 1])/4
        
        #scan the output to make it into 1D array
        MVD_u_out = MVD_u.ravel()
        MVD_v_out = MVD_v.ravel()
        output_MVD_u = ["%d" % x for x in MVD_u_out]
        output_MVD_v = ["%d" % x for x in MVD_v_out]
        size_MVD_u = len(output_MVD_u)
        size_MVD_v = len(output_MVD_v)
        
        #save it into *.mrg file format
        for i in np.arange(0, size_MVD_u - 1):
            f.write(output_MVD_u[i])
            f.write(" ")
        f.write(output_MVD_u[size_MVD_u - 1])
        f.write("\n")
        for i in np.arange(0, size_MVD_v - 1):
            f.write(output_MVD_v[i])
            f.write(" ")
        f.write(output_MVD_v[size_MVD_v - 1])
        f.write("\n")
        f.write("P")
        f.write("\n")
    else:
        #do 4:2:0 chroma subsampling
        cb_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
        cr_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
        for i in np.arange(0, (rows/2)):
            for j in np.arange(0, (cols/2)):
                cb_chroma[i][j] = (cb[(2*i)][(2*j)] + cb[(2*i) + 1][(2*j)] + cb[(2*i)][(2*j) + 1] + cb[(2*i) + 1][(2*j) + 1])/4
                cr_chroma[i][j] = (cr[(2*i)][(2*j)] + cr[(2*i) + 1][(2*j)] + cr[(2*i)][(2*j) + 1] + cr[(2*i) + 1][(2*j) + 1])/4
        
        #do 2D DCT transformation
        y_dct = cv2.dct(Y)
        cb_dct = cv2.dct(cb_chroma)
        cr_dct = cv2.dct(cr_chroma)
        
        #do quantization
        y_quantized = np.zeros((rows, cols), np.float32)
        cb_quantized = np.zeros(((rows/2), (cols/2)), np.float32)
        cr_quantized = np.zeros(((rows/2), (cols/2)), np.float32)
        a = 0
        b = 0
        for i in np.arange(0, rows):
            for j in np.arange(0, cols):
                if (i < ((rows/2)) and j < ((cols/2))):
                    if (a == 0 and b == 0):
                        y_quantized[i][j] = round(y_dct[i][j]/8)
                        cb_quantized[i][j] = round(cb_dct[i][j]/8)
                        cr_quantized[i][j] = round(cr_dct[i][j]/8)
                    else:
                        y_quantized[i][j] = math.floor(y_dct[i][j]/8)
                        cb_quantized[i][j] = math.floor(cb_dct[i][j]/8)
                        cr_quantized[i][j] = math.floor(cr_dct[i][j]/8)
                else:
                    if (a == 0 and b == 0):
                        y_quantized[i][j] = round(y_dct[i][j]/8)
                    else:
                        y_quantized[i][j] = math.floor(y_dct[i][j]/8)
                b += 1
                if (b == 8):
                    b = 0
            a += 1
            if (a == 8):
                a = 0
        
        #scan the output matrix of quantization to make it into 1D array
        y_out = y_quantized.ravel()
        cb_out = cb_quantized.ravel()
        cr_out = cr_quantized.ravel()
        output_y = ["%d" % x for x in y_out]
        output_cb = ["%d" % x for x in cb_out]
        output_cr = ["%d" % x for x in cr_out]
        size_y = len(output_y)
        size_cb = len(output_cb)
        size_cr = len(output_cr)
        
        #save it into *.mrg file format
        for i in np.arange(0, size_y - 1):
            f.write(output_y[i])
            f.write(" ")
        f.write(output_y[size_y - 1])
        f.write("\n")
        for i in np.arange(0, size_cb - 1):
            f.write(output_cb[i])
            f.write(" ")
        f.write(output_cb[size_cb - 1])
        f.write("\n")
        for i in np.arange(0, size_cr - 1):
            f.write(output_cr[i])
            f.write(" ")
        f.write(output_cr[size_cr - 1])
        f.write("\n")
        f.write("I")
        f.write("\n")
    prev = Y
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
f.close()

