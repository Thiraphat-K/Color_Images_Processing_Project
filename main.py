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

def color_of_temp(date,color_data):
    try:
        color_data[-1]["Date"] = date
        count_colors = 0
        data = []
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
        data_c = combine_hex_color(data)
        data.clear()
        data.append(date)
        data.append(data_c)
        return data
    except: print(f'calculate_Temp failed~~')

def combine_hex_color(data):
    try:
        data_count = [data[i][1] for i in range(len(data))]
        total = sum(data_count)
        red = int(sum([int(data[i][0][1:3], 16)*data[i][1] for i in range(len(data))])/total)
        green = int(sum([int(data[i][0][3:5], 16)*data[i][1] for i in range(len(data))])/total)
        blue = int(sum([int(data[i][0][5:7], 16)*data[i][1] for i in range(len(data))])/total)
        pad = lambda c : c if len(c)==2 else '0'+c
        combine_color = '#'+pad(hex(red)[2:])+pad(hex(green)[2:])+pad(hex(blue)[2:])
        return combine_color
    except: print(f'error: combine_hex_color()')

def temp_compare(color_data):
    try:
        data = []
        data.append(color_data)
        red = int(sum([int(color_data[1][1:3], 16)]))
        green = int(sum([int(color_data[1][3:5], 16)]))
        blue = int(sum([int(color_data[1][5:7], 16)]))
        lst_temp = list(constant.temp_color)
        if 0 <= red <= 18 and 0 <= green <= 18 and 0 < blue <= 146+18: color_data.append(constant.temp_color[lst_temp[0]])
        elif 0 <= red <= 18 and 0 <= green <= 18 and 164 < blue <= 182+19: color_data.append(constant.temp_color[lst_temp[1]])
        elif 0 <= red <= 18 and 0 <= green <= 18 and 201 < blue <= 219+18: color_data.append(constant.temp_color[lst_temp[2]])
        elif 0 <= red <= 18 and 0 <= green <= 18 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[3]])
        elif 0 <= red <= 18 and 18 < green <= 36+19 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[4]])
        elif 0 <= red <= 18 and 55 < green <= 73+18 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[5]])
        elif 0 <= red <= 18 and 91 < green <= 109+19 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[6]])
        elif 0 <= red <= 18 and 128 < green <= 146+18 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[7]])
        elif 0 <= red <= 18 and 164 < green <= 182+19 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[8]])
        elif 0 <= red <= 18 and 201 < green <= 219+18 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[9]])
        elif 0 <= red <= 18 and 237 < green <= 255 and 237 < blue <= 255: color_data.append(constant.temp_color[lst_temp[10]])
        elif 18 < red <= 36+18 and 237 < green <= 255 and 237 >= blue > 219-18: color_data.append(constant.temp_color[lst_temp[11]])
        elif 54 < red <= 73+18 and 237 < green <= 255 and 201 >= blue > 182-18: color_data.append(constant.temp_color[lst_temp[12]])
        elif 91 < red <= 109+19 and 237 < green <= 255 and 164 >= blue > 146-19: color_data.append(constant.temp_color[lst_temp[13]])
        elif 128 < red <= 146+18 and 237 < green <= 255 and 127 >= blue > 109-18: color_data.append(constant.temp_color[lst_temp[14]])
        elif 164 < red <= 182+19 and 237 < green <= 255 and 91 >= blue > 73-19: color_data.append(constant.temp_color[lst_temp[15]])
        elif 201 < red <= 219+18 and 237 < green <= 255 and 54 >= blue > 36-18: color_data.append(constant.temp_color[lst_temp[16]])
        elif 237 < red <= 255 and 237 < green <= 255 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[17]])
        elif 237 < red <= 255 and 237 >= green > 219-18 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[18]])
        elif 237 < red <= 255 and 201 >= green > 182-18 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[19]])
        elif 237 < red <= 255 and 164 >= green > 146-19 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[20]])
        elif 237 < red <= 255 and 127 >= green > 109-18 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[21]])
        elif 237 < red <= 255 and 91 >= green > 73-19 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[22]])
        elif 237 < red <= 255 and 54 >= green > 36-18 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[23]])
        elif 237 < red <= 255 and 18 >= green >= 0 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[24]])
        elif 237 >= red > 255-18 and 18 >= green >= 0 and 18 >= blue >= 0: color_data.append(constant.temp_color[lst_temp[25]])
        elif len(color_data)==2: color_data.append('')
        rgb = f'{red}, {green}, {blue}'
        color_data.append(rgb)
        return color_data
    except: print(f'error: temp_compare()')

def plotGraph(image, colors, counts):
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
                        await saveData(temp_compare(color_of_temp(date,colors)))
                        print(f'{date} success!!')
                    except:
                        date_error = f'{y}-{m}-{d}-{t}'
                        print(f'error: {date_error}')
    # plotGraph(modi_img,getValueFromKey(colors,'Color'),getValueFromKey(colors,'Count'))

async def saveData(temp_color):
    try:
        wb = load_workbook('./data/temperature_2016.xlsx')
        ws = wb.worksheets[0]
        ws.append(temp_color)
        wb.save('./data/temperature_2016.xlsx')
    except: print(f'error: saveData()')

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