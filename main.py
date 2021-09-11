import urllib
import PIL
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

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
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
    
    modi_img = center_crop(image, (330, 270,165,186))
    modified_image = modi_img.reshape(modi_img.shape[0]*modi_img.shape[1], 3)

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
    return hex_colors, counts, modi_img

def saveData(hex_rgb , counts):
    data = {'Color': [],
            'Count': []
        }
    if 'Color' in data and 'Count' in data:
        for i in range(len(counts)):
            data['Color'].append(hex[i])
            data['Count'].append(counts[i])
    df = pd.DataFrame(data, columns = ['Color', 'Count'])
    df.to_excel(r'./data/temperature.xlsx', header=True)

def plotGraph():
    f, ax = plt.subplots(1, 2, figsize = (8, 6))
    ax[0].imshow(modi_img)
    ax[1].pie(counts.values(), labels = counts.values(), colors = hex)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

if __name__ == "__main__":
    url_img = "http://tiwrmdev.hii.or.th/ContourImg/2021/09/10/hatempY2021M09D10T14.png"
    print("downloading : %s success!!" % url_img)
    image = get_image(url_img)
    hex, counts, modi_img = get_colors(image, 8)
    save_data = saveData(hex, counts)
    for i in range(len(counts)):
        print(f"{hex[i]}" ,counts[i])
    print("sum = ",sum(counts.values()))
    plotGraph()