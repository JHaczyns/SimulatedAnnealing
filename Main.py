import PySimpleGUI as sg
import glob
import copy
import math
import cv2
import numpy as np
import random as rnd
import logging
from PIL import Image
from alive_progress import alive_bar
def Energy(pointtab):
    Energy=0
    for i in pointtab:
        for j in pointtab:
            if(i!=j):
                x1, y1, = i
                x2, y2 = j
                E = 1 * 1 / np.math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
                Energy=Energy+E
    Energy=Energy/2
    return Energy

def createimage(width,height,color):
    img =np.ones((width,height,3),np.uint8)
    for i in range(len(img)):
        img[i]=color
    return img

def movepixel(img,x,y,deltax,deltay):
    img[x,y]=(255,255,255)
    xmax, ymax, size = np.shape(img)
    if (x + deltax >xmax-1)and(y + deltay >ymax-1):
        img[img.shape[0]-1, img.shape[1]-1] = (0, 0, 0)
    elif(x+deltax>xmax-1):
        img[img.shape[0]-1, y + deltay] = (0, 0, 0)
    elif(y + deltay >ymax-1):
        img[x + deltax, img.shape[0]-1] = (0, 0, 0)
    else:
        img[x+deltax,y+deltay]=(0,0,0)

def movepoit(point,delta):
    maxsize=99

    if(point[1] + delta[1] > maxsize and point[0] + delta[0] > maxsize):
        point[0] = maxsize
        point[1] = maxsize
    elif (point[1] + delta[1] < 0 and point[0] + delta[0] < 0):
        point[0] = 1
        point[1] = 1
    elif(point[0]+delta[0]>maxsize):
        point[0]=maxsize
        point[1]=point[1] + delta[1]
    elif (point[1] + delta[1] > maxsize):
        point[0] = point[0] + delta[0]
        point[1] = maxsize
    elif (point[0] + delta[0] <0):
        point[0] = 0
        point[1] = point[1] + delta[1]
    elif (point[1] + delta[1] < 0):
        point[0] = point[0] + delta[0]
        point[1] = 1
    else:
        point[0] = point[0] + delta[0]
        point[1] = point[1] + delta[1]

    return point

def movepoit2(point,delta,contour):
    newpoint=[point[1]+delta[1],point[0]+delta[0]]
    newpoint = tuple([int(round(newpoint[0])), int(round(newpoint[1]))])

    if(cv2.pointPolygonTest(contour[0],newpoint,True)<=0):
            point=point
    else:
        point[0]=point[0]+delta[0]
        point[1]=point[1]+delta[1]

    return point

# def drawpixel(img,b,color):
#     for i in b:
#         x,y:
#     img[x,y]=color
#     return img

def random(number,img):
    pointtab=[]
    x,y,size=np.shape(img)
    for i in range(number):
        randx=rnd.randint(0,x-1)
        randy = rnd.randint(0, y-1)
        while(img[randx,randy][0]==0):
            randx = rnd.randint(0, x)
            randy = rnd.randint(0, y)
        img[randx,randy]=(0,0,0)
        pointtab.append([randx,randy])
    return pointtab

def draw(pointab,img):
    for i in range(len(pointab)):
        x,y = pointab[i]
        img[x,y]=(0,0,0)

    return img

def drawcontours(pointab,img):
    for i in range(len(pointab)):
        x,y = pointab[i]
        img[x,y]=(255,0,0)

    return img

def simulated_annealing(pointtab,contours,initial_temp,final_temp,alpha,k):
    pointtab2=copy.deepcopy(pointtab)
    current_temp = initial_temp
    imagetab = []
    # Start by initializing the current state with the initial state
    current_state = pointtab
    solution = current_state
    iter=0
    while current_temp > final_temp:
        with alive_bar(int(initial_temp)) as bar:
            for i in range (len(pointtab)):
                iter += 1
                current_state=pointtab2
                choice = rnd.choice(get_neighbors())
                backup_copy=copy.deepcopy(pointtab)
                pointtab[i]=movepoit2(pointtab[i],choice,contours)#DO POPRAWY BO WYWALA NA UJEMNE WARTOŚCI
                neighbor = pointtab
                # Check if neighbor is best so far
                cost_diff = get_cost(neighbor)-get_cost(current_state)
                pointtab2=copy.deepcopy(neighbor)
            # if the new solution is better, accept it
                if cost_diff < 0:
                    solution = neighbor
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    if rnd.uniform(0, 1) < math.exp(-cost_diff / current_temp/k):
                        solution = neighbor
                    else:
                        pointtab=backup_copy
                # decrement the temperature
                current_temp = current_temp*alpha
                if(iter%10==0):
                    output2=drawimage(pointtab)
                    imagetab.append(output2)
                # if (iter % 100 == 0):
                #     print((initial_te yield
                #       mp-current_temp)/initial_temp*100,"%")
            bar(int(initial_temp-current_temp))

    output2 = drawimage(pointtab)
    imagetab.append(output2)
    return imagetab,solution

def get_cost(pointtab):
    """Calculates cost of the argument state for your solution."""
    Energy = 0
    for i in pointtab:
        for j in pointtab:
            if (i != j):
                x1, y1, = i
                x2, y2 = j
                E = 1 * 1 / np.math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
                Energy = Energy + E
    Energy = Energy / 2
    return Energy

def get_neighbors():
    """Returns neighbors of the argument state for your solution."""
    tab=[[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]
    tab2=[[1,0],[0,1],[-1,0],[0,-1]]
    return tab2

def drawimage(pointab):
    img = createimage(200, 200, (255, 255, 255))
    img2=draw(pointab,img)
    img2=drawcontours(conturyfinal,img2)
    scale_percent = 200
    src2 = img2
    # calculate the 50 percent of original dimensions
    width = int(src2.shape[1] * scale_percent / 100)
    height = int(src2.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output2 = cv2.resize(src2, dsize)
    return output2


def openImage(image):
    image=cv2.imread(image)
    redlow=[237, 28, 36]
    lower = np.array([0, 0, 254])
    upper = np.array([0, 0, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    redpoints = []
    points= np.where(image ==redlow)
    for i in range(len(points[0])):
        redpoints.append([points[1][i],points[0][i]])
    imagecopy=image
    cv2.drawContours(imagecopy, contours,-1, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
    return imagecopy,contours,redpoints

# dsize
def MAIN(pointab,contours,initial_temp,final_temp,alpha,k):
    img=createimage(200,200,(255,255,255))
    # pointab=redpoints
    print("BEFORE")
    print(Energy(pointab))
    print(pointab)
    neighbor = rnd.choice(get_neighbors())
    imagetab,solution=simulated_annealing(pointab,contours,initial_temp,final_temp,alpha,k)
    # img2=createimage(25,25,(255,255,255))
    # img2=draw(pointab,img2)

    print("AFTER")
    print(solution)
    print(Energy(solution))
    for i in imagetab:
        cv2.imshow('s',i)
        cv2.waitKey(10)
    cv2.waitKey(0)
    return 0
     # scale_percent = 1000
    # src = img
    # src2 = img2
    # #calculate the 50 percent of original dimensions
    # width = int(src.shape[1] * scale_percent / 100)
    # height = int(src.shape[0] * scale_percent / 100)
    #
    # # dsize
    # dsize = (width, height)
    #
    # output = cv2.resize(src, dsize)
    # output2 = cv2.resize(src2, dsize)
    #
    # cv2.imshow('s',output)
    # cv2.imshow('s2',output2)
    # cv2.waitKey()

filesList = glob.glob('Maps/*.png')
layout = [[sg.Text('%47s' %'Wybierz mapę:'), sg.Listbox(filesList, size=(20, 4), key='Choice',default_values=filesList[0])],
          [sg.Text('%47s' %'Temperatura początkowa:'),sg.Input(key='initial_temp', enable_events=False, default_text='1200')],
          [sg.Text('%47s' %'Temperatura zakończenia (>0 and <initial_temp):'),sg.Input(key='final_temp', enable_events=False, default_text='0.1')],
          [sg.Text('%47s' %'Współczynnik wyżarzania (alfa) :'),sg.Input(key='alfa', enable_events=False, default_text='0.999')],
          [sg.Text('%47s' %'Współczynnik k:'),sg.Input(key='k', enable_events=False, default_text='0.001')],
          [sg.Button('Ok')]]

window = sg.Window('Wybierz zestaw reguł',layout )

event, values = window.read(close=True)

if event == 'Ok':
    try:
        mapSRC=values["Choice"][0]
        tempInit=float(values["initial_temp"])
        finalTemp=float(values["final_temp"])
        alfa=float(values["alfa"])
        k=float(values["k"])
    except Exception:
        sg.popup("Podano niewłaściwe dane!")
        exit()
        print("error")
    print(alfa)
    image, contury, redpoints = openImage(mapSRC)
    conturyfinal = []
    redpointsfinal = []
    cv2.imshow('s', image)
    for i in contury:
        for x in i:
            conturyfinal.append([x[0][1], x[0][0]])
    for i in redpoints:
        redpointsfinal.append([i[1], i[0]])
    MAIN(redpointsfinal, contury, tempInit, finalTemp, alfa, k)
