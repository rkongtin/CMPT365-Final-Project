from Tkinter import *
from tkFileDialog import askopenfilename
import numpy as np
import cv2
import imageio


#ask to open the *.mrg file
filename = askopenfilename()

# --- Decompression ---
#open and read the *.mrg file
with open(filename, 'r') as f:
    data = f.readlines()
f.close()

prev = np.zeros((360, 640), np.uint8)
u, v = (0, 0)
MVD_u_prev = np.zeros((22, 40), np.float32)
MVD_v_prev = np.zeros((22, 40), np.float32)
MV_u = np.zeros((22, 40), np.float32)
MV_v = np.zeros((22, 40), np.float32)
writer = imageio.get_writer('output.mp4', fps = 25)
for i in range(len(data) - 1, -1, -1):
    #check if it is I-frame or P-frame
    if ('I' in data[i]):
        #convert strings to floats in each line
        line1 = data[i - 3].split()
        line2 = data[i - 2].split()
        line3 = data[i - 1].split()
        i -= 3
        y_out = np.array(map(float, line1))
        cb_out = np.array(map(float, line2))
        cr_out = np.array(map(float, line3))
        y_quantized = np.reshape(y_out, (-1, 640))
        cb_quantized = np.reshape(cb_out, (-1, 320))
        cr_quantized = np.reshape(cr_out, (-1, 320))
        
        #do inverse quantization
        r, c = cb_quantized.shape
        y_dct = np.zeros(((2*r), (2*c)), np.float32)
        cb_dct = np.zeros((r, c), np.float32)
        cr_dct = np.zeros((r, c), np.float32)
        a = 0
        b = 0
        for i in np.arange(0, (2*r)):
            for j in np.arange(0, (2*c)):
                if (i < r and j < c):
                    y_dct[i][j] = y_quantized[i][j] * 8
                    cb_dct[i][j] = cb_quantized[i][j] * 8
                    cr_dct[i][j] = cr_quantized[i][j] * 8
                else:
                    y_dct[i][j] = y_quantized[i][j] * 8
                b += 1
                if (b == 8):
                    b = 0
            a += 1
            if (a == 8):
                a = 0
        
        #do 2D IDCT transformation
        y = cv2.idct(y_dct)
        cb_chroma = cv2.idct(cb_dct)
        cr_chroma = cv2.idct(cr_dct)
        
        #reverse 4:2:0 chroma subsampling
        cb = np.zeros(((2*r), (2*c)), np.float32)
        cr = np.zeros(((2*r), (2*c)), np.float32)
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                cb[(2*i)][(2*j)] = cb_chroma[i][j]
                cb[(2*i) + 1][(2*j)] = cb_chroma[i][j]
                cb[(2*i)][(2*j) + 1] = cb_chroma[i][j]
                cb[(2*i) + 1][(2*j) + 1] = cb_chroma[i][j]
                cr[(2*i)][(2*j)] = cr_chroma[i][j]
                cr[(2*i) + 1][(2*j)] = cr_chroma[i][j]
                cr[(2*i)][(2*j) + 1] = cr_chroma[i][j]
                cr[(2*i) + 1][(2*j) + 1] = cr_chroma[i][j]
        
        #transform from YCbCr to RGB
        img_YCbCr = np.zeros(((2*r), (2*c), 3), np.uint8)
        img_YCbCr[:,:,0] = y
        img_YCbCr[:,:,1] = cb
        img_YCbCr[:,:,2] = cr
        image = cv2.cvtColor(img_YCbCr, cv2.COLOR_YCR_CB2RGB)
        writer.append_data(image)
        
    elif ('P' in data[i]):
        #convert strings to integers in each line
        line1 = data[i - 2].split()
        line2 = data[i - 1].split()
        i -= 2
        MVD_u_out = np.array(map(float, line1))
        MVD_v_out = np.array(map(float, line2))
        MVD_u = np.reshape(MVD_u_out, (-1, 40))
        MVD_v = np.reshape(MVD_v_out, (-1, 40))
        
        #calculate the difference, MVD, between preceding and current motion vector
        r, c, ch = prev.shape
        for x in range(0, r - 16, 16):
            for y in range(0, c, 16):
                MV_u[x/16, y/16] = MVD_u_prev[x/16, y/16] - MVD_u[x/16, y/16]
                MV_v[x/16, y/16] = MVD_v_prev[x/16, y/16] - MVD_v[x/16, y/16]
        MVD_u_prev = MV_u
        MVD_v_prev = MV_v
        
        #apply the motion vectors to the previous frame to get current frame
        Y = np.zeros((r, c), np.float32)
        cb = np.zeros((r, c), np.float32)
        cr = np.zeros((r, c), np.float32)
        for x in range(0, r - 16, 16):
            for y in range(0, c - 16, 16):
                u = int(MV_u[x/16, y/16])
                v = int(MV_v[x/16, y/16])
                for a in range(x, x+16):
                    for b in range(y, y+16):
                        Y[a, b] = prev[a + u, b + v, 0]
                        cb[a, b] = prev[a + u, b + v, 1]
                        cr[a, b] = prev[a + u, b + v, 2]
        
        #transform from YCbCr to RGB
        img_YCbCr = np.zeros((r, c, 3), np.uint8)
        img_YCbCr[:,:,0] = Y
        img_YCbCr[:,:,1] = cb
        img_YCbCr[:,:,2] = cr
        image = cv2.cvtColor(img_YCbCr, cv2.COLOR_YCR_CB2RGB)
        writer.append_data(image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    prev = img_YCbCr

writer.close()
