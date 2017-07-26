# -*- coding: utf-8 -*-

# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



'''
Generate training and test images.

'''


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys
import colorsys
import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common_cn

BGS_DIR = './bgs_1'

FONT_DIR = './fonts_cn'
FONT_HEIGHT = 48  # Pixel size to which the chars are resized

#D_OUTPUT_DIR = './syndata_cn/detect'
R_OUTPUT_DIR = 'zj_500_5-900'  #'./syndata_8000'
WORD_TXT = 'zj_500_5-900.txt'
#WORD_TIMES = 4
#OUTPUT_SHAPE = (32, 128)

#dlable = 'D_label.txt'
train_lable = 'zj_500_5-900_train.txt'
test_lable = 'zj_500_5-900_test.txt'
def make_char_ims(font_path, output_height,font_color): 
    b = random.randint(30,50)
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)+b

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new('RGB', (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, font_color,font=font)
        scale = float(output_height) / (height-b)
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        #im.save('text.jpg')
        yield c, numpy.array(im).astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    
    text_color = 1.
    
    return text_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    
    out_of_bounds_scale = True
    out_of_bounds_trans = True

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    while out_of_bounds_scale:
        scale = random.uniform((min_scale + max_scale) * 0.5 -
                               (max_scale - min_scale) * 0.5 * scale_variation,
                               (min_scale + max_scale) * 0.5 +
                               (max_scale - min_scale) * 0.5 * scale_variation)
        if scale > max_scale or scale < min_scale:
            continue
        out_of_bounds_scale = False
        
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape[0], from_shape[1]
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(numpy.dot(M, corners), axis=1) -
                              numpy.min(numpy.dot(M, corners), axis=1))
    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    while out_of_bounds_trans:
        trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
        trans = ((2.0 * trans) ** 5.0) / 2.0
        if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
            continue
        out_of_bounds_trans = False

    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    T = trans + center_to - numpy.dot(M, center_from)
    #M = numpy.eye(2)
    #M = numpy.hstack([M, numpy.zeros([2, 1])])
    M = numpy.hstack([M, T])

    return M


def generate_code():
    
    code = CHARS
        
    return code


def generate_text(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2), 3)

    text_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1], :] = char_im
        x += char_im.shape[1] + spacing

    text = numpy.ones(out_shape) * text_color * text_mask
    #cv2.imwrite('text.jpg', text)
    #print code
    return text, code


def generate_bg(images_dir):
    while True:
        filenames = os.listdir(images_dir)
        loadlist = []
        for fn in filenames:
            fullfilename = os.path.join(images_dir,fn)
            bg = cv2.imread(fullfilename, cv2.CV_LOAD_IMAGE_COLOR)
            bg = bg / 255.
            yield bg

def get_dominant_color(image):
    cv2.imwrite('1.jpg',image)
    image = Image.open('1.jpg').convert('RGBA')
    max_score = None
    dominant_color = None
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        if a == 0:
            continue
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)       
        y = (y - 16.0) / (235 - 16) 
           
        if y > 0.9:
            dominant_color = (r, g, b)
            continue

        score = (saturation + 0.1) * count
        
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

            
    os.remove(os.path.join('1.jpg'))
    
    return dominant_color
	
def colorRGB(img_color):
    a = img_color[0]
    b = img_color[1]
    c = img_color[2]
    ran = random.randint(0,50)
    ran1 = random.randint(0,50)
    ran2 = random.randint(0,50)
    if a > 127 :
        a = a - 127 - ran
        a = max(a,0)
    else:
        a = a + 127 + ran
        a = min(a,255)
    if b > 127 :
        b = b - 127 - ran1
        b = max(b,0)
    else:
        b = b + 127 + ran1
        b = min(b,255)
    if c > 127 :
        c = c - 127 - ran2
        c = max(c,0)
    else:
        c = c + 127 + ran2
        c = min(c,255)
    font_color = (a,b,c)
    print font_color
    return font_color
	
def generate_im(num_bg_images):
    bg = next(bgs)
    
    img_bg = bg * 255.
    
    bg_color = get_dominant_color(img_bg)
    try:
        font_color = colorRGB(bg_color)
    except:
        print 'the bgs is damage'
        font_color = (0,0,0)
        pass
    (a,b,c) = bg_color
    fonts, font_char_ims= load_fonts(FONT_DIR, font_color)
    char_ims = font_char_ims[random.choice(fonts)]
    text, code = generate_text(FONT_HEIGHT, char_ims)
    cv2.imwrite('text.png', text*255)
    #print text,code
    M = make_affine_transform(
            from_shape=text.shape,
            to_shape=bg.shape,
            min_scale=0.5,
            max_scale=0.5,
            rotation_variation=0.1,
            scale_variation=0.2,
            translation_variation=0.2)
    ht, wt = text.shape[0], text.shape[1]
    #print ht,wt
    corners_bf = numpy.matrix([[0, wt, 0, wt],
                               [0, 0, ht, ht]])
    text = cv2.warpAffine(text, M, (bg.shape[1], bg.shape[0]))

    corners_af = numpy.dot(M[:2, :2], corners_bf) + M[:2, -1]
    tl = numpy.min(corners_af, axis=1).T
    br = numpy.max(corners_af, axis=1).T

    box = numpy.hstack([tl, br])
    if a > 127:
        out = bg - text
    else:
        out = text + bg

    out = cv2.resize(out, (bg.shape[1], bg.shape[0]))
    out = numpy.clip(out, 0., 1.)
    #print (out*255)
    cv2.imwrite('box.png', out*255)
    return out, code, box


def load_fonts(folder_path,font_color):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT,font_color))
    return fonts, font_char_ims


def generate_ims():
    '''
    Generate number plate images.

    :return:
        Iterable of number plate images.

    '''
    variation = 1.0

    while True:
        yield generate_im(num_bg_images)


if __name__ == '__main__':
    
    #if not os.path.exists(D_OUTPUT_DIR):
    #    os.mkdir(D_OUTPUT_DIR)
    if not os.path.exists(R_OUTPUT_DIR):
        os.mkdir(R_OUTPUT_DIR)
    
    #Dfile = open(D_OUTPUT_DIR+os.sep+dlable, 'w')
    Train_file = open(train_lable, 'w')
    Test_file = open(test_lable, 'w')
    #Wfile = open('word1.txt', 'r')
    Wfile = open(WORD_TXT, 'r')
    fname = BGS_DIR 
    filenames = os.listdir(BGS_DIR)
    for fn in filenames:
        fullfilename = os.path.join(BGS_DIR,fn)
        bg = cv2.imread(fullfilename, cv2.CV_LOAD_IMAGE_COLOR)
        imgH = 500
        h, w = bg.shape[:2]
        ratio = w / float(h)
        imgW = int(ratio * imgH)
        res=cv2.resize(bg,(imgW,imgH),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(fullfilename, res)

    num_bg_images = len(os.listdir(BGS_DIR))
    bgs = generate_bg(BGS_DIR)
    cnt = 0
    for line in Wfile:
        CHARS = line.strip('\n').decode('utf-8')
        im_gen = itertools.islice(generate_ims(), num_bg_images)
        for img_idx, (im, c, bx) in enumerate(im_gen):
            im = im * 255.

            rimage ='R_{:08d}.png'.format(cnt)
            print rimage
            crop = im[int(bx[:, 1]):int(bx[:, 3]), int(bx[:, 0]):int(bx[:, 2]), ]
            '''          
            imgH = 32
            h, w = crop.shape[:2]
            ratio = w / float(h)
            imgW = int(ratio * imgH)
            res=cv2.resize(crop,(imgW,imgH),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(R_OUTPUT_DIR+os.sep+rimage.encode('utf-8'), res)'''
            '''if img_idx % 5 == 0:
                Test_file.write(rimage.encode('utf-8') + ' ' + c.encode('utf-8') + '\n')
                Test_file.flush()	'''		
            res=cv2.resize(crop,(100,32),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(R_OUTPUT_DIR+os.sep+rimage.encode('utf-8'), res)				
            Train_file.write(rimage.encode('utf-8') + ' ' + c.encode('utf-8') + '\n')
            Train_file.flush()
            cnt += 1
            '''if img_idx>(WORD_TIMES-1):
                break'''
    #Dfile.close()
    Train_file.close()
    Test_file.close()
    Wfile.close()
