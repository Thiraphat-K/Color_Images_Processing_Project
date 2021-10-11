from csv import DictWriter
from numpy.lib.function_base import percentile
from skimage import color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import urllib.request
import os
import collections
import pandas as pd
import asyncio
from until import constant
from openpyxl import load_workbook

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(img):
    image = np.asarray(bytearray(img.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    imageRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return imageRGB

def center_crop(img, dim):
	height, width = img.shape[0], img.shape[1]
    # process crop width and height for max available dimension
	# crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1] # check width
	# crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] # check height
	mid_x, mid_y = int(width/2), int(height/2)
    # cw2, ch2 = int(crop_width/2), int(crop_height/2) # crop x,y stable
	crop_img = img[mid_y-dim[0]:mid_y+dim[1], mid_x-dim[2]:mid_x+dim[3]]
	return crop_img

def get_colors(image, number_of_colors):
    modi_img = center_crop(image, (270, 175, 165, 186))
    modified_image = modi_img.reshape(modi_img.shape[0]*modi_img.shape[1], modi_img.shape[2]) #reshape 2D matrix to 3D matrix
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts_df = Counter(labels)
        # sort to ensure correct color percentage 
    counts = collections.OrderedDict(sorted(counts_df.items()))
    center_colors = clf.cluster_centers_
        # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts]
    # rgb_colors = [ordered_colors[i] for i in counts]
    return list(map(lambda h, c: {"Color":h , "Count":c }, hex_colors , counts.values())), modi_img

def saveData(date,color_data):
    try:
        color_data[-1]["Date"] = date
        count_colors = 0
        data = []
        wb = load_workbook('./data/temperature_2018.xlsx')
        ws = wb.worksheets[11]
        for i in range(len(color_data)):
            remove_color = False
            for c in constant.rm_color:
                if color_data[i]["Color"].find(c) != -1: remove_color = True
            if not remove_color: data.append(list(color_data[i].values()))
        for n in data:count_colors += n[1]
        if data[-1][2] == date :
            for d in data:
                percent = f'{(d[1]/count_colors)*100:.0f}' #find percentage
                d[1] = int(percent)
                ws.append(d)
                wb.save('./data/temperature_2018.xlsx')
            print(f'{date} success!!')
    except: print(f'{date} failed~~')

def plotGraph(image,colors, counts):
    f, ax = plt.subplots(1, 2, figsize = (8, 6))
    ax[0].imshow(image)
    ax[1].pie(counts, labels = counts, colors = colors)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    # plt.show()

async def get_data():
    for y in constant.YEAR:
        for m in constant.MONTH:
            for d in constant.DAY:
                for t in constant.TIME:
                    try:
                        url_img = f"http://tiwrmdev.hii.or.th/ContourImg/{y}/{m}/{d}/hatempY{y}M{m}D{d}T{t}.png"
                        response = urllib.request.urlopen(url_img)
                        # print("downloading : %s success!!" % url_img)
                        image = get_image(response)
                        colors, modi_img = get_colors(image, 8)
                        colors = deleteItem(colors)
                        # print(f"Color : {getValueFromKey(colors,'Color')} | Count : {getValueFromKey(colors,'Count')}")
                        # print("sum =",sum(getValueFromKey(colors,'Count')))
                        date = f'{y}-{m}-{d}-{t}'
                        saveData(date,colors)
                    except:
                        date_error = f'{y}-{m}-{d}-{t}'
                        print(f'error: {date_error}')
    # plotGraph(modi_img,getValueFromKey(colors,'Color'),getValueFromKey(colors,'Count'))

def getValueFromKey(array , key): return [i[key] for i in array if key in i]

def deleteItem(colors_l):
    colors_l = sorted(colors_l, key=lambda k:k['Count'])
    del_I = lambda c : c ['Color'] in ['#fefefe']
    for i in range(len(colors_l)):
        if del_I(colors_l[i]):
            colors_l.pop(i)
    return colors_l

async def main():
    await get_data()

if __name__ == "__main__":
    asyncio.run(main())