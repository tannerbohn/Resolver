# Resolver

The intent of this project was to see if the style of an image could be recreated through learning how to iteratively increase the resolution of an image.

**Requirements**
  * Keras


The algorithm creates new images by taking an initial low res image, and then for every pixel in that image, it used a small window around it to predict what a higher resolution version of that pixel would be. Using these predictions, it creates a new image with double the resolution. The process is then repeated until the desired resolution is achieved.

To predict how pixels are "sharpened", a neural network is used (though any prediction model should work just fine), where the training data is of the form ((8 neighbours around original pixel, scale), (4 pixels the center pixel is sharpened to)). To get this data for multiple resolution of an image, I, the image is repeated scaled down by a factor of 2, and we can use two adjacent versions of the image to determine how a particular pixel in the lower resolution version is sharpened into 4 other pixels in the larger version.