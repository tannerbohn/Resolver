from __future__ import print_function

import Image, ImageFilter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import random
import numpy as np
import math

def weighted_choice(choices):
	total = sum(w for c, w in choices)
	r = random.uniform(0, total)
	upto = 0
	for c, w in choices:
		if upto + w >= r:
			return c
		upto += w
	assert False, "Shouldn't get here"


def getData(img, nb_samples):

	X = []
	y = []
	IMGS = [img]
	PIX = [img.load()]
	while True:
		newSize = [IMGS[-1].size[0]/2, IMGS[-1].size[1]/2]
		if newSize[0]%2==1: newSize[0] += 1
		if newSize[1]%2==1: newSize[1] += 1
		if min(newSize) < 4:
			break
		IMGS.append(img.resize(newSize, Image.ANTIALIAS))
		PIX.append(IMGS[-1].load())
	# IMGS are arranged from largest to smallest, we want to reverse that
	IMGS.reverse()
	PIX.reverse()


	print("IMGS: ", len(IMGS))

	for _ in range(nb_samples):

		imIndex = weighted_choice([[i,1] for i in range(1, len(IMGS))])


		hrimg = IMGS[imIndex]
		drimg = IMGS[imIndex-1]

		hrpix = PIX[imIndex]
		drpix = PIX[imIndex-1]

		#scale = 4./(math.log(hrimg.size[1]))
		scale = 8./math.sqrt(hrimg.size[1]/4.) - 3.

		# choose a random pixel location
		px = random.random()
		py = random.random()
		drx = int(px*drimg.size[0])
		dry = int(py*drimg.size[1])


		dr_center = drpix[drx,dry]
		dr_Neighbours = []

		D= [-1,0,1]
		for dx in D:
			for dy in D:

				if dx==0 and dy==0: continue

				try:
					n = drpix[drx+dx,dry+dy]
				except:
					n = drpix[drx,dry]
				dr_Neighbours.append(n)

		newx = np.array(dr_Neighbours).reshape(1,-1)[0]/255.
		newx = np.append(newx, scale)
		X.append(newx)


		hrx1, hry1 = drx*2, dry*2
		hrx2, hry2 = hrx1+1, hry1+1

		hr_center = []
		for hx in [hrx1, hrx2]:
			for hy in [hry1, hry2]:
				#print(hx, hy)
				try:
					hr_center.append(hrpix[hx,hy])
				except:
					print("SAY WHATTTT", hrimg.size, (hx, hy), (drx,dry))
					hr_center.append(dr_center)

		y.append(np.array(hr_center).reshape(1,-1)[0]/255.)

	return np.array(X), np.array(y)



def newImage(model, seed = [], start=(2,2), iters=6):

	IMGS = []
	PIX = []

	if seed == []:
		N1 = Image.new('RGB', start, "white")
		pix1 = N1.load()
		for x in range(N1.size[0]):
			for y in range(N1.size[1]):
				pix1[x,y] = tuple([random.randint(0,255) for _ in [0,1,2]])


		IMGS.append(N1)
		PIX.append(pix1)
	else:
		im = seed.resize(start, Image.ANTIALIAS)
		IMGS.append(im)
		PIX.append(im.load())





	for inum in range(iters):

		prevSize = IMGS[inum].size

		# scale is 1/log(size we are creating)
		#scale = 4./(math.log(prevSize[1]*2))
		scale = 8./math.sqrt(prevSize[1]*2/4.) - 3.

		

		N2 = Image.new('RGB', (prevSize[0]*2,prevSize[1]*2), "white")
		pix2 = N2.load()

		# for every pixel in N1, we find the neighbours and calculate what the 4 new
		# pixels are for N2
		n2x = 0
		for x in range(IMGS[inum].size[0]):
			n2y = 0
			for y in range(IMGS[inum].size[1]):

				dr_center = PIX[inum][x,y]
				dr_Neighbours = []
				D= [-1,0,1]
				for dx in D:
					for dy in D:
						if dx==0 and dy==0: continue

						try:
							n = PIX[inum][x+dx,y+dy]
						except:
							n = PIX[inum][x,y]
						dr_Neighbours.append(n)
				X = np.array(dr_Neighbours).reshape(1,-1)/255.
				X = np.array([np.append(X[0], scale)])
				

				output = model.predict(X)[0]*255
				output = [max(min(int(v),255),0) for v in output]
				output = np.array(output).reshape(4,-1)
				output = [tuple(v) for v in output]

				pix2[n2x,n2y]=output[0]
				pix2[n2x,n2y+1]=output[1]
				pix2[n2x+1,n2y]=output[2]
				pix2[n2x+1,n2y+1]=output[3]

				n2y += 2

			n2x += 2

		IMGS.append(N2)
		PIX.append(pix2)

		print("Done ",inum)


	return IMGS



if __name__ == "__main__":

	I = Image.open('imgs/starry.png').resize((256,256), Image.ANTIALIAS).convert('RGB')#.filter(ImageFilter.BLUR) #.resize((256,256))
	#I = Image.open('imgs/rain_princess.jpg').resize((256,256), Image.ANTIALIAS)
	X, y = getData(I, 6000)

	
	model = Sequential()
	model.add(Dense(25, activation='sigmoid', input_shape=(3*8+1,)))
	model.add(Dense(15, activation='sigmoid'))
	model.add(Dense(10, activation='sigmoid'))
	model.add(Dense(15, activation='sigmoid'))
	model.add(Dense(3*4, activation='sigmoid'))
	model.compile(optimizer='adam', loss='mse')

	model.fit(X, y, batch_size=6000, nb_epoch=5000) #0.0192
	IMGS = newImage(model, seed=I, start=(8,8), iters=5)
	IMGS[-1].show()
	

	# some test images to play with
	i2 = Image.open('imgs/i2.png').convert('RGB')
	d21 = Image.open('imgs/d21.png').convert('RGB')
	SF = Image.open('imgs/sanfrancisco.jpg')