from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import urllib.request
import collections
from until import constant
from openpyxl import load_workbook

class color_process_temperature():
    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def get_image(self, img):
        image = np.asarray(bytearray(img.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        imageRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        return imageRGB

    def center_crop(self, img, dim):
        height, width = img.shape[0], img.shape[1]
        # process crop width and height for max available dimension
        # crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1] # check width
        # crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] # check height
        mid_x, mid_y = int(width/2), int(height/2)
        # cw2, ch2 = int(crop_width/2), int(crop_height/2) # crop x,y stable
        crop_img = img[mid_y-dim[0]:mid_y+dim[1], mid_x-dim[2]:mid_x+dim[3]]
        return crop_img

    def get_colors(self, image, number_of_colors):
        img = self.center_crop(image, (270, 175, 165, 186))
        modified_image = img.reshape(img.shape[0]*img.shape[1], img.shape[2]) #reshape to 2D matrix

        k_means = KMeans(n_clusters = number_of_colors)
        labels = k_means.fit_predict(modified_image)
        counts_df = Counter(labels)
        center_colors = k_means.cluster_centers_

        counts = collections.OrderedDict(sorted(counts_df.items()))
        ordered_colors = [center_colors[i] for i in counts]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts]
        # rgb_colors = [ordered_colors[i] for i in counts]
        return list(map(lambda h, c: {"Color":h , "Count":c }, hex_colors , counts.values())), img

    def color_of_temp(self, date, color_data):
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
            data_c = self.combine_hex_color(data)
            data.clear()
            data.append(date)
            data.append(data_c)
            return data
        except: print(f'error: color_of_temp()')

    def combine_hex_color(self, data):
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

    def rgb_to_hsl(self, r, g, b):
        try:
            maxc = max(r, g, b)
            minc = min(r, g, b)
            sumc = (maxc+minc)
            rangec = (maxc-minc)
            l = sumc/2.0
            if minc == maxc: return 0, 0, l
            if l <= 0.5: s = rangec / sumc
            else: s = rangec / (1-abs(2.0*l-1))
            rc = (maxc-r) / rangec
            gc = (maxc-g) / rangec
            bc = (maxc-b) / rangec
            if r == maxc: h = bc-gc
            elif g == maxc: h = 2.0+rc-bc
            else: h = 4.0+gc-rc
            h = float(f'{(((h/6.0) % 1.0)*360):.1f}')
            l = float(f'{((l/255)*100):.0f}')
            s = int(f'{int(s*(-100))}')
            return h, l
        except: print(f'error: rgb_to_hsl()')

    def temp_compare(self, color_data):
        try:
            data = []
            data.append(color_data)
            red = int(sum([int(color_data[1][1:3], 16)]))
            green = int(sum([int(color_data[1][3:5], 16)]))
            blue = int(sum([int(color_data[1][5:7], 16)]))
            h = (self.rgb_to_hsl(red, green, blue))[0]
            l = (self.rgb_to_hsl(red, green, blue))[1]
            if 240.0 >= h > 234.0 and 30.75 >= l >= 29.0: color_data.append(15.0)
            elif 240.0 >= h > 234.0 and 34.25 >= l > 30.75: color_data.append(15.5)
            elif 240.0 >= h > 234.0 and 37.75 >= l > 34.25: color_data.append(16.0)
            elif 240.0 >= h > 234.0 and 41.25 >= l > 37.75: color_data.append(16.5)
            elif 240.0 >= h > 234.0 and 44.75 >= l > 41.25: color_data.append(17.0)
            elif 240.0 >= h > 234.0 and 48.25 >= l > 44.75: color_data.append(17.5)
            elif 240.0 >= h > 234.0 and 51.75 >= l > 48.25: color_data.append(18.0)
            elif 234.0 >= h > 232.0: color_data.append(18.5)
            elif 232.0 >= h > 229.75: color_data.append(19.0)
            elif 229.75 >= h > 225.25: color_data.append(19.5)
            elif 225.25 >= h > 220.75: color_data.append(20.0)
            elif 220.75 >= h > 216.25: color_data.append(20.5)
            elif 216.25 >= h > 212.0: color_data.append(21.0)
            elif 212.0 >= h > 208.0: color_data.append(21.5)
            elif 208.0 >= h > 203.75: color_data.append(22.0)
            elif 203.75 >= h > 199.25: color_data.append(22.5)
            elif 199.25 >= h > 194.75: color_data.append(23.0)
            elif 194.75 >= h > 190.25: color_data.append(23.5)
            elif 190.25 >= h > 186.0: color_data.append(24.0)
            elif 186.0 >= h > 182.0: color_data.append(24.5)
            elif 182.0 >= h > 177.5: color_data.append(25.0)
            elif 177.5 >= h > 172.5: color_data.append(25.5)
            elif 172.5 >= h > 169.0: color_data.append(26.0)
            elif 169.0 >= h > 162.0: color_data.append(26.5)
            elif 162.0 >= h > 150.75: color_data.append(27.0)
            elif 150.75 >= h > 140.25: color_data.append(27.5)
            elif 140.25 >= h > 127.5: color_data.append(28.0)
            elif 127.5 >= h > 112.5: color_data.append(28.5)
            elif 112.5 >= h > 99.75: color_data.append(29.0)
            elif 99.75 >= h > 89.25: color_data.append(29.5)
            elif 89.25 >= h > 80.5: color_data.append(30.0)
            elif 80.5 >= h > 73.5: color_data.append(30.5)
            elif 73.5 >= h > 67.5: color_data.append(31.0)
            elif 67.5 >= h > 62.5: color_data.append(31.5)
            elif 62.5 >= h > 58.0: color_data.append(32.0)
            elif 58.0 >= h > 54.0: color_data.append(32.5)
            elif 54.0 >= h > 49.75: color_data.append(33.0)
            elif 49.75 >= h > 45.25: color_data.append(33.5)
            elif 45.25 >= h > 40.75: color_data.append(34.0)
            elif 40.75 >= h > 36.25: color_data.append(34.5)
            elif 36.25 >= h > 32.0: color_data.append(35.0)
            elif 32.0 >= h > 28.0: color_data.append(35.5)
            elif 28.0 >= h > 23.75: color_data.append(36.0)
            elif 23.75 >= h > 19.25: color_data.append(36.5)
            elif 19.25 >= h > 14.75: color_data.append(37.0)
            elif 14.75 >= h > 10.25: color_data.append(37.5)
            elif 10.25 >= h > 6.0: color_data.append(38.0)
            elif 6.0 >= h > 2.0: color_data.append(38.5)
            elif 2.0 >= h >= 0.0 and 48.25 <= l < 51.75: color_data.append(39.0)
            elif 2.0 >= h >= 0.0 and 44.75 <= l < 48.25: color_data.append(39.5)
            elif 2.0 >= h >= 0.0 and 30.0 <= l < 44.75: color_data.append(40.0)
            elif len(color_data)==2: color_data.append('')
            # rgb = f'{red}, {green}, {blue}'
            # color_data.append(rgb)
            return color_data
        except: print(f'error: temp_compare()')

    def plotGraph(self, image, colors, counts):
        f, ax = plt.subplots(1, 2, figsize = (8, 6))
        ax[0].imshow(image)
        ax[1].pie(counts, labels = counts, colors = colors)
        ax[0].axis('off') #hide the axis
        ax[1].axis('off')
        f.tight_layout()
        # plt.show()

    async def get_data(self):
        data_lst = []
        temp_lst = []
        for y in constant.YEAR:
            for m in constant.MONTH:
                for d in constant.DAY:
                    try:
                        temp_lst.clear()
                        for t in constant.TIME:
                            data_lst.clear()
                            try:
                                url_img = f"http://tiwrmdev.hii.or.th/ContourImg/{y}/{m}/{d}/hatempY{y}M{m}D{d}T{t}.png"
                                response = urllib.request.urlopen(url_img)
                                # print("downloading : %s success!!" % url_img)
                                image = self.get_image(response)
                                colors, img = self.get_colors(image, 8)
                                colors = self.deleteItem(colors)
                                # print(f"Color : {self.getValueFromKey(colors,'Color')} | Count : {self.getValueFromKey(colors,'Count')}")
                                # print("sum =",sum(self.getValueFromKey(colors,'Count')))
                                date = f'{y}-{m}-{d}-{t}'
                                # print(self.temp_compare(self.color_of_temp(date,colors)))
                                temp_lst.append((self.temp_compare(self.color_of_temp(date,colors)))[2])
                            except:
                                print(f'error time {t}.00 : call_url')
                        date = f'{y}-{m}-{d}'
                        data_lst.append(date)
                        count_temp = float(f'{sum(temp_lst)/len(temp_lst):.1f}')
                        data_lst.append(count_temp)
                        await self.saveData(data_lst)
                        print(f'{date} completed!')
                    except: print(f'{date} failed~')
        # self.plotGraph(img,self.getValueFromKey(colors,'Color'),self.getValueFromKey(colors,'Count'))

    async def saveData(self, temp_data):
        try:
            wb = load_workbook('./data/temperature_2016_to_2018.xlsx')
            ws = wb.worksheets[2]
            ws.append(temp_data)
            wb.save('./data/temperature_2016_to_2018.xlsx')
        except: print(f'error: saveData()')

    def getValueFromKey(self, array , key): return [i[key] for i in array if key in i]

    def deleteItem(self, colors_l):
        colors_l = sorted(colors_l, key=lambda k:k['Count'])
        del_I = lambda c : c ['Color'] in ['#fefefe']
        for i in range(len(colors_l)):
            if del_I(colors_l[i]):
                colors_l.pop(i)
        return colors_l