from tkinter import *
from tkinter import messagebox as mb, simpledialog as sd, filedialog as fd
import cv2, traceback, glob, os, numpy as np, time, fnmatch, glitch_this, pyautogui, cv2.data, random, math, ctypes, shutil
from scipy.interpolate import UnivariateSpline
from os.path import exists
from bing_image_downloader import downloader
from PIL import Image, ImageFilter
starter = glitch_this.ImageGlitcher()
messagebox = ctypes.windll.user32.MessageBoxW

def imagepicker():
	try:
		if exists("originalimage.png"):
			mb.showwarning("Warning!", "You already have an image. It will be deleted to avoid conflicts.")
			os.remove("originalimage.png")
		filepicker = mb.askyesno("File Picker", "Do you want to use the file picker?")
		if filepicker == True:
			file = fd.askopenfilename(filetypes=[("Image Files", "*.png *.jpg")])
			if file == "":
				mb.showerror("Error!", "No file selected.")
				return
			else:
				img = cv2.imread(file)
				mb.showinfo("Success!", "Image loaded successfully with filename: " + file.split("/")[-1])
				os.rename(file, "originalimage.png")
				imgname = "originalimage.png"
				img = Image.open(imgname)
				fixed = 800
				height_percent = (fixed / float(img.size[1]))
				width_size = int((float(img.size[0]) * float(height_percent)))
				img = img.resize((width_size, fixed), Image.Resampling.LANCZOS)
				time.sleep(0.50)
				img.save(imgname, "PNG", quality=100)
		else:
			query = sd.askstring("Query", "What do you want to search for?")
			if query == "":
				mb.showerror("Error!", "You didn't enter anything. Please try again.")
				return
			if query == None:
				mb.showerror("Error!", "You didn't enter anything. Please try again.")
				return
			else:
				downloader.download(query, limit=1, output_dir="./", adult_filter_off=True)
				os.chdir(f"{query}/")
				os.chdir("..")
				for file in glob.glob(f"{query}/*.png"):
					if exists(file):
						os.rename(f"{file}", "originalimage.png")
					os.rmdir(f"{query}/")
				for file2 in glob.glob(f"{query}/*.jpg"):
					if exists(file2):
						os.rename(f"{file2}", "originalimage.png")
					os.rmdir(f"{query}/")
				mb.showinfo("Success!", "Image downloaded successfully.")
				imgname = "originalimage.png"
				img = Image.open(imgname)
				fixed = 800
				height_percent = (fixed / float(img.size[1]))
				width_size = int((float(img.size[0]) * float(height_percent)))
				img = img.resize((width_size, fixed), Image.Resampling.LANCZOS)
				os.remove(imgname)
				time.sleep(0.50)
				img.save(imgname, "PNG", quality=100)
				showimg = mb.askyesno("Show?", "Do you want to show the image?")
				if showimg == True:
					cv2.imshow("Image", cv2.imread(imgname))
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def blackandwhite():
	try:
		originalimage = cv2.imread("originalimage.png")
		grayimage = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
		(thresh, blackandwhiteimage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
		cv2.imshow("Black and White", blackandwhiteimage)
		cv2.imwrite("blackandwhite.png", blackandwhiteimage)
		mb.showinfo("Success!", "Black and White image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("blackandwhite.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def cartoon():
	try:
		altmethod = mb.askyesno("Alternative Method?", "Do you want to use the alternative method? It would be faster, but it might look a bit different.")
		if altmethod == True:
			img = cv2.imread("originalimage.png")
			grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			grayimg = cv2.GaussianBlur(grayimg, (3, 3), 0)
			edgeimg = cv2.Laplacian(grayimg, -1, ksize=5)
			edgeimg = 255 - edgeimg
			ret, edgeimg = cv2.threshold(edgeimg, 150, 255, cv2.THRESH_BINARY)
			edgepreserve = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)
			output = np.zeros(grayimg.shape)
			output = cv2.bitwise_and(edgepreserve, edgepreserve, mask=edgeimg)
			cv2.imshow("Cartoon", output)
			cv2.imwrite("cartoon.png", output)
			mb.showinfo("Success!", "Cartoon image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("cartoon.png")
		else:
			originalimage = cv2.imread("originalimage.png")
			line_size = 7
			blur_value = 7
			k = 9
			d = 7
			gray = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
			gray_blur = cv2.medianBlur(gray, blur_value)
			edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
			data = np.float32(originalimage).reshape((-1, 3))
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
			ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
			center = np.uint8(center)
			result = center[label.flatten()]
			result = result.reshape((originalimage.shape))
			blurred = cv2.bilateralFilter(result, d, sigmaColor=200, sigmaSpace=200)
			cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
			cv2.imshow("Cartoon", cartoon)
			cv2.imwrite("cartoon.png", cartoon)
			mb.showinfo("Success!", "Cartoon image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("cartoon.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def invert():
	try:
		originalimage = cv2.imread("originalimage.png", 0)
		invertedimg = cv2.bitwise_not(originalimage)
		cv2.imshow("Inverted", invertedimg)
		cv2.imwrite("inverted.png", invertedimg)
		mb.showinfo("Success!", "Inverted image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("inverted.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def pencilsketch():
	img = cv2.imread("originalimage.png")
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		gray, color = cv2.pencilSketch(img, sigma_s=40, sigma_r=0.15, shade_factor=0.06)
		cv2.imwrite("pencilsketch.png", gray)
		cv2.imshow("sketch", gray)
		mb.showinfo("Success!", "Sketch image created.")
		if autodraw == True:
			try:
				cv2.destroyAllWindows()
				mb.showwarning("Starting...", "Depending on the image, this may take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
				time.sleep(5)
				thresh, blackandwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
				xStart = 664
				yStart = 230
				try:
					for y in range(len(blackandwhite)):
						row = blackandwhite[y]
						for x in range(len(row)):
							if row[x] == 0:
								pyautogui.click(xStart + x, yStart + y, _pause=False)
								time.sleep(0.01)
					mb.showinfo("Success!", "Autodraw image done!")
				except pyautogui.FailSafeException:
					mb.showwarning("Failsafe triggered!", "Failsafe has been triggered, stopping...")
					return
				except:
					mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
				else:
					return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("pencilsketch.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def sepia():
	try:
		img = cv2.imread("originalimage.png")
		sepia = np.array(img, dtype=np.float64)
		sepia = cv2.transform(sepia, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
		sepia[np.where(sepia > 255)] = 255
		sepia = np.array(sepia, dtype=np.uint8)
		cv2.imwrite("sepia.png", sepia)
		cv2.imshow("Sepia", sepia)
		mb.showinfo("Success!", "Sepia image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("sepia.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def blur():
	try:
		img = Image.open("originalimage.png")
		numtoblur = sd.askinteger("Blur", "How many times do you want to blur the image?")
		if numtoblur == None:
			numtoblur = 1
		img = img.filter(ImageFilter.GaussianBlur(numtoblur))
		img.save("blur.png")
		img2 = cv2.imread("blur.png")
		cv2.imshow("Blur", img2)
		mb.showinfo("Success!", "Blur image created with a blur value of " + str(numtoblur) + ".")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("blur.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def contour():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(img, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
		cv2.imshow("Contours", img)
		cv2.imwrite("contours.png", img)
		mb.showinfo("Success!", "Contours image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("contours.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def detailed():
	try:
		img = Image.open("originalimage.png")
		img = img.filter(ImageFilter.DETAIL)
		img.save("detailed.png")
		img2 = cv2.imread("detailed.png")
		cv2.imshow("Detailed", img2)
		mb.showinfo("Success!", "Detailed image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("detailed.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def edgeenhance():
	try:
		img = Image.open("originalimage.png")
		img = img.convert("RGB")
		img = img.filter(ImageFilter.EDGE_ENHANCE)
		img.save("edgedetect.png")
		img2 = cv2.imread("edgedetect.png")
		cv2.imshow("Edge Detect", img2)
		mb.showinfo("Success!", "Edge Enhanced image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("edgedetect.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def emboss():
	try:
		img = Image.open("originalimage.png")
		img = img.convert("RGB")
		img = img.filter(ImageFilter.EMBOSS)
		img.save("emboss.png")
		img2 = cv2.imread("emboss.png")
		cv2.imshow("Emboss", img2)
		mb.showinfo("Success!", "Emboss image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("emboss.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def edgefinder():
	try:
		img = Image.open("originalimage.png")
		img = img.convert("RGB")
		img = img.filter(ImageFilter.FIND_EDGES)
		img.save("edgefinder.png")
		img2 = cv2.imread("edgefinder.png")
		cv2.imshow("Edge Finder", img2)
		mb.showinfo("Success!", "Edge Finder image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("edgefinder.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def sharpen():
	try:
		img = Image.open("originalimage.png")
		img = img.convert("RGB")
		img = img.filter(ImageFilter.SHARPEN)
		img.save("sharp.png")
		img2 = cv2.imread("sharp.png")
		cv2.imshow("Sharp", img2)
		mb.showinfo("Success!", "Sharp image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("sharp.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def smooth():
	try:
		img = Image.open("originalimage.png")
		img = img.convert("RGB")
		img = img.filter(ImageFilter.SMOOTH_MORE)
		img.save("smooth.png")
		img2 = cv2.imread("smooth.png")
		cv2.imshow("Smooth", img2)
		mb.showinfo("Success!", "Smooth image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("smooth.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def summer():
	try:
		img = cv2.imread("originalimage.png")
		def table(x, y):
			spline = UnivariateSpline(x, y)
			return spline(range(256))
		increase = table([0, 64, 128, 256], [0, 80, 160, 256])
		decrease = table([0, 64, 128, 256], [0, 50, 100, 256])
		blue, green, red = cv2.split(img)
		red = cv2.LUT(red, increase).astype(np.uint8)
		blue = cv2.LUT(blue, decrease).astype(np.uint8)
		sum = cv2.merge((blue, green, red))
		cv2.imwrite("summer.png", sum)
		img2 = cv2.imread("summer.png")
		cv2.imshow("Summer", img2)
		mb.showinfo("Success!", "Summer image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("summer.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
	
def winter():
	try:
		img = cv2.imread("originalimage.png")
		def table(x, y):
			spline = UnivariateSpline(x, y)
			return spline(range(256))
		increase = table([0, 64, 128, 256], [0, 80, 160, 256])
		decrease = table([0, 64, 128, 256], [0, 50, 100, 256])
		blue, green, red = cv2.split(img)
		red = cv2.LUT(red, decrease).astype(np.uint8)
		blue = cv2.LUT(blue, increase).astype(np.uint8)
		sum = cv2.merge((blue, green, red))
		cv2.imwrite("winter.png", sum)
		img2 = cv2.imread("winter.png")
		cv2.imshow("Winter", img2)
		mb.showinfo("Success!", "Winter image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("winter.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def glitch():
	try:
		img = Image.open("originalimage.png")
		amt = sd.askinteger("Amount", "Enter the amount of times you want to glitch the image.")
		coloroffset = mb.askyesno("Color Offset?", "Do you want to glitch the color offset?")
		scanlines = mb.askyesno("Scanlines?", "Do you want to add scanlines?")
		glitch = starter.glitch_image(img, amt, color_offset=coloroffset, scan_lines=scanlines)
		glitch.save("glitch.png")
		img2 = cv2.imread("glitch.png")
		cv2.imshow("Glitch", img2)
		mb.showinfo("Success!", "Glitch image created with a glitch amount of " + str(amt) + ", color offset = " + str(coloroffset) + ", and scanlines = " + str(scanlines))
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("glitch.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def asciiart():
	try:
		pngfiles = fnmatch.filter(os.listdir('.'), '*.png')
		imgask = sd.askstring("Image", "What is the name of the image you want to convert to ASCII art? (Enter the name of the file you want to convert)\n\nList of files in the current directory:\n" + str(pngfiles))
		if imgask == "":
			mb.showwarning("Error!", "You didn't enter anything. Defaulting to the first image in the directory.")
			imgask = pngfiles[0]
		if imgask not in pngfiles:
			mb.showerror("Error!", "The image you entered does not exist. This can happen if you entered the name of the file without the extension or if you misspelled the name.\n\nList of files in the current directory:\n" + str(pngfiles))
			return
		img = Image.open(imgask)
		width, height = img.size
		ratio = height / width
		new_width = 120
		new_height = ratio * new_width * 0.55
		img = img.resize((new_width, int(new_height)))
		img = img.convert('L')
		chars = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]
		pixels = img.getdata()
		new_pixels = [chars[pixels//25] for pixels in pixels]
		new_pixels = ''.join(new_pixels)
		new_pixels_count = len(new_pixels)
		ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
		ascii_image = "\n".join(ascii_image)
		with open("asciiart.txt", "w") as f:
			f.write(ascii_image)
		mb.showinfo("Success!", "ASCII art created. (Zoom out to see it more clearly, and when your done, delete asciiart.txt.)")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
	

def moon():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.equalizeHist(img)
		cv2.imwrite("moon.png", img)
		img2 = cv2.imread("moon.png")
		cv2.imshow("Moon", img2)
		mb.showinfo("Success!", "Moon image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("moon.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def clarendon():
	try:
		img = cv2.imread("originalimage.png")
		clarendon = img.copy()
		blue, green, red = cv2.split(clarendon)
		ogvalues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
		blueval = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
		redval = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
		greenval = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255])
		fullrange = np.arange(0, 256)
		bluelookup = np.interp(fullrange, ogvalues, blueval)
		greenlookup = np.interp(fullrange, ogvalues, greenval)
		redlookup = np.interp(fullrange, ogvalues, redval)
		bluechannel = cv2.LUT(blue, bluelookup)
		greenchannel = cv2.LUT(green, greenlookup)
		redchannel = cv2.LUT(red, redlookup)
		clarendon = cv2.merge([bluechannel, greenchannel, redchannel])
		clarendon = np.uint8(clarendon)
		cv2.imwrite("clarendon.png", clarendon)
		img2 = cv2.imread("clarendon.png")
		cv2.imshow("Clarendon", img2)
		mb.showinfo("Success!", "Clarendon image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("clarendon.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def laplacian():
	try:
		image = cv2.imread("originalimage.png")
		laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale=1, delta=0)
		logKernel = np.array(( [0.4038, 0.8021, 0.4038], [0.8021, -4.8233, 0.8021], [0.4038, 0.8021, 0.4038]), dtype="float")
		logimg = cv2.filter2D(image, cv2.CV_32F, logKernel)
		cv2.normalize(laplacian, laplacian, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		cv2.normalize(logimg, logimg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		cv2.imwrite("laplacian.png", laplacian)
		cv2.imshow("laplacian", laplacian)
		mb.showinfo("Success!", "Laplacian image created.")	
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("laplacian.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def kelvin():
	img = cv2.imread("originalimage.png")
	output = img.copy()
	bluechannel, greenchannel, redchannel = cv2.split(output)
	redValuesOriginal = np.array([0, 60, 110, 150, 235, 255])
	redValues = np.array([0, 102, 185, 220, 245, 245 ])
	greenValuesOriginal = np.array([0, 68, 105, 190, 255])
	greenValues = np.array([0, 68, 120, 220, 255 ])
	blueValuesOriginal = np.array([0, 88, 145, 185, 255])
	blueValues = np.array([0, 12, 140, 212, 255])
	allvalues = np.arange(0, 256)
	bluelookup = np.interp(allvalues, blueValuesOriginal, blueValues)
	greenlookup = np.interp(allvalues, greenValuesOriginal, greenValues)
	redlookup = np.interp(allvalues, redValuesOriginal, redValues)
	bluechannel = cv2.LUT(bluechannel, bluelookup)
	greenchannel = cv2.LUT(greenchannel, greenlookup)
	redchannel = cv2.LUT(redchannel, redlookup)
	output = cv2.merge([bluechannel, greenchannel, redchannel])
	output = np.uint8(output)
	cv2.imwrite("kelvin.png", output)
	img2 = cv2.imread("kelvin.png")
	cv2.imshow("Kelvin", img2)
	mb.showinfo("Success!", "Kelvin image created.")
	asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
	if asktodelete == True:
		cv2.destroyAllWindows()
		os.remove("kelvin.png")

def xpro():
	try:
		img = cv2.imread("originalimage.png")
		output = img.copy()
		B, G, R = cv2.split(output)
		vignettescale = 6
		k = np.min([output.shape[1], output.shape[0]]) / vignettescale
		kernelX = cv2.getGaussianKernel(output.shape[1], k)
		kernelY = cv2.getGaussianKernel(output.shape[0], k)
		kernel = kernelY * kernelX.T
		mask = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		B = B + B * mask
		G = G + G * mask
		R = R + R * mask
		output = cv2.merge([B, G, R])
		output = output / 2
		output = np.clip(output, 0, 255)
		output = np.uint8(output)
		B, G, R = cv2.split(output)
		redValuesOriginal = np.array([0, 42, 105, 148, 185, 255])
		redValues = np.array([0, 28, 100, 165, 215, 255 ])
		greenValuesOriginal = np.array([0, 40, 85, 125, 165, 212, 255])
		greenValues = np.array([0, 25, 75, 135, 185, 230, 255 ])
		blueValuesOriginal = np.array([0, 40, 82, 125, 170, 225, 255 ])
		blueValues = np.array([0, 38, 90, 125, 160, 210, 222])
		allvalues = np.arange(0, 256)
		redlookup = np.interp(allvalues, redValuesOriginal, redValues)
		R = cv2.LUT(R, redlookup)
		greenlookup = np.interp(allvalues, greenValuesOriginal, greenValues)
		G = cv2.LUT(G, greenlookup)
		bluelookup = np.interp(allvalues, blueValuesOriginal, blueValues)
		B = cv2.LUT(B, bluelookup)
		output = cv2.merge([B, G, R])
		output = np.uint8(output)
		output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
		output = np.float32(output)
		Y, Cr, Cb = cv2.split(output)
		Y = Y * 1.2
		Y = np.clip(Y, 0, 255)
		output = cv2.merge([Y, Cr, Cb])
		output = np.uint8(output)
		output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
		cv2.imwrite("xpro.png", output)
		img2 = cv2.imread("xpro.png")
		cv2.imshow("X-Pro", img2)
		mb.showinfo("Success!", "X-Pro image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("xpro.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def erode():
	try:
		img = cv2.imread("originalimage.png")
		interode = sd.askinteger("Erode", "How many times do you want to erode?")
		if interode == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 1.")
			interode = 1
		erosionsize = interode
		if interode <= -1:
			mb.showwarning("Warning!", "You can't do a number lower than 1. Defaulting to 1.")
			erosionsize = 1
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosionsize + 1, 2 * erosionsize + 1), (erosionsize, erosionsize))
		erodedimg = cv2.erode(img, element)
		cv2.imwrite("eroded.png", erodedimg)
		img2 = cv2.imread("eroded.png")
		cv2.imshow("Eroded", img2)
		mb.showinfo("Success!", "Eroded image created with an erode value of " + str(interode) + ".")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("eroded.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def dilate():
	try:
		img = cv2.imread("originalimage.png")
		intdilate = sd.askinteger("Dilate", "How many times do you want to dilate?")
		if intdilate == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 1.")
			intdilate = 1
		dilatesize = intdilate
		if intdilate <= -1:
			mb.showwarning("Warning!", "You can't do a number lower than 1. Defaulting to 1.")
			dilatesize = 1
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * dilatesize + 1, 2 * dilatesize + 1), (dilatesize, dilatesize))
		dilateimg = cv2.dilate(img, element)
		cv2.imwrite("dilated.png", dilateimg)
		img2 = cv2.imread("dilated.png")
		cv2.imshow("Dilated", img2)
		mb.showinfo("Success!", "Dilated image created with a dilate value of " + str(intdilate) + ".")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("dilated.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def gamma():
	try:
		img = cv2.imread("originalimage.png")
		gamma = sd.askinteger("Gamma", "How much gamma do you want to use?")
		if gamma == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 5.")
			gamma = 5
		values = np.arange(0, 256)
		lut = np.uint8(255 * np.power((values / 255.0), gamma))
		result = cv2.LUT(img, lut)
		cv2.imwrite("gamma.png", result)
		img2 = cv2.imread("gamma.png")
		cv2.imshow("Gamma", img2)
		mb.showinfo("Success!", "Gamma image created with a gamma value of " + str(gamma) + ".")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("gamma.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def contrast():
	try:
		img = cv2.imread("originalimage.png")
		imgYCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		imgYCB = np.float32(imgYCB)
		Y, C, B = cv2.split(imgYCB)
		scaleask = sd.askinteger("Contrast", "How much contrast do you want to use?")
		if scaleask == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 1.")
			scaleask = 1
		alpha = scaleask
		Y = Y * alpha
		Y = np.clip(Y, 0, 255)
		imgYCB = cv2.merge([Y, C, B])
		imgYCB = np.uint8(imgYCB)
		result = cv2.cvtColor(imgYCB, cv2.COLOR_YCrCb2BGR)
		cv2.imwrite("contrast.png", result)
		img2 = cv2.imread("contrast.png")
		cv2.imshow("Contrast", img2)
		mb.showinfo("Success!", "Contrast image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("contrast.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def brightness():
	try:
		img = cv2.imread("originalimage.png")
		imgYCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		imgYCB = np.float32(imgYCB)
		Y, C, B = cv2.split(imgYCB)
		scaleask = sd.askinteger("Brightness", "How much brightness do you want to use?")
		if scaleask == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 1.")
			scaleask = 1
		beta = scaleask
		Y = Y + beta
		Y = np.clip(Y, 0, 255)
		imgYCB = cv2.merge([Y, C, B])
		imgYCB = np.uint8(imgYCB)
		result = cv2.cvtColor(imgYCB, cv2.COLOR_YCrCb2BGR)
		cv2.imwrite("brightness.png", result)
		img2 = cv2.imread("brightness.png")
		cv2.imshow("Brightness", img2)
		mb.showinfo("Success!", "Brightness image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("brightness.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def hsv():
	try:
		img = cv2.imread("originalimage.png")
		hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsvimgcopy = hsvimg.copy()
		hsvimgcopy = np.float32(hsvimgcopy)
		saturationscale = 0.01
		H, S, V = cv2.split(hsvimgcopy)
		S = np.clip(S * saturationscale, 0, 255)
		hsvimgcopy = cv2.merge([H, S, V])
		hsvimgcopy = np.uint8(hsvimgcopy)
		hsvimgcopy = cv2.cvtColor(hsvimgcopy, cv2.COLOR_HSV2BGR)
		cv2.imwrite("hsv.png", hsvimg)
		img2 = cv2.imread("hsv.png")
		cv2.imshow("HSV", img2)
		mb.showinfo("Success!", "HSV image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("hsv.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def rotate():
	try:
		img = cv2.imread("originalimage.png")
		angle = sd.askinteger("Rotate", "What angle do you want to rotate the image by?")
		if angle == None:
			mb.showwarning("Warning!", "You didn't enter anything. Defaulting to 120.")
			angle = 120
		rotation = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
		result = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
		cv2.imwrite("rotate.png", result)
		img2 = cv2.imread("rotate.png")
		cv2.imshow("Rotate", img2)
		mb.showinfo("Success!", "Rotated image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("rotate.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def affine():
	try:
		img = cv2.imread("originalimage.png")
		rows, cols, ch = img.shape
		pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
		pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
		M = cv2.getAffineTransform(pts1, pts2)
		dst = cv2.warpAffine(img, M, (cols, rows))
		cv2.imwrite("affine.png", dst)
		img2 = cv2.imread("affine.png")
		cv2.imshow("Affine", img2)
		mb.showinfo("Success!", "Affine image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("affine.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def inverseaffine():
	try:
		img = cv2.imread("originalimage.png", cv2.IMREAD_GRAYSCALE)
		cv2.line(img, (450, 100), (750, 650), (0, 0, 255), 5, cv2.LINE_AA, 0)
		cv2.line(img, (750, 650), (1000, 300), (0, 0, 255), 5, cv2.LINE_AA, 0)
		cv2.line(img, (1000, 300), (450, 100), (0, 0, 255), 5, cv2.LINE_AA, 0)
		warpMat1 = np.float32([[1.2, 0.2, 2], [-0.3, 1.3, 1]])
		result1 = cv2.warpAffine(img, warpMat1, (int(1.5*img.shape[1]), int(1.4*img.shape[0])), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
		cv2.imwrite("inverseaffine.png", result1)
		img2 = cv2.imread("inverseaffine.png")
		cv2.imshow("Inverse Affine", img2)
		mb.showinfo("Success!", "Inverse Affine image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("inverseaffine.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def saturate():
	try:
		img = cv2.imread("originalimage.png")
		hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsvimg)
		cv2.imwrite("saturated.png", s)
		img2 = cv2.imread("saturated.png")
		cv2.imshow("Saturated", img2)
		mb.showinfo("Success!", "Saturated image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("saturated.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def hue():
	try:
		img = cv2.imread("originalimage.png")
		hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsvimg)
		cv2.imwrite("hue.png", h)
		img2 = cv2.imread("hue.png")
		cv2.imshow("Hue", img2)
		mb.showinfo("Success!", "Hue image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("hue.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def cca():
	try:
		askforcolor = mb.askyesno("Color?", "Do you want to use color?")
		if askforcolor == True:
			img = cv2.imread("originalimage.png", cv2.IMREAD_GRAYSCALE)
			th, binaryimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
			_, binaryimg = cv2.connectedComponents(binaryimg)
			binaryimgclone = np.copy(binaryimg)
			(minval, maxval, minpos, maxpos) = cv2.minMaxLoc(binaryimgclone)
			binaryimgclone = 255 * (binaryimgclone - minval) / (maxval - minval)
			binaryimgclone = np.uint8(binaryimgclone)
			binimgclonecolormap = cv2.applyColorMap(binaryimgclone, cv2.COLORMAP_JET)
			cv2.imwrite("cca.png", binimgclonecolormap)
			img2 = cv2.imread("cca.png")
			cv2.imshow("Connected Components", img2)
			mb.showinfo("Success!", "Connected Components image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("cca.png")
		else:
			img = cv2.imread("originalimage.png", cv2.IMREAD_GRAYSCALE)
			th, binaryimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
			_, binaryimg = cv2.connectedComponents(binaryimg)
			binaryimgclone = np.copy(binaryimg)
			(minval, maxval, minpos, maxpos) = cv2.minMaxLoc(binaryimgclone)
			binaryimgclone = 255 * (binaryimgclone - minval) / (maxval - minval)
			binaryimgclone = np.uint8(binaryimgclone)
			binimgclonecolormap = cv2.applyColorMap(binaryimgclone, cv2.COLORMAP_JET)
			cv2.imwrite("cca.png", binaryimgclone)
			img2 = cv2.imread("cca.png")
			cv2.imshow("Connected Components", img2)
			mb.showinfo("Success!", "Connected Components image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("cca.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def desaturate():
	try:
		img = cv2.imread("originalimage.png")
		hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsvimgcopy = hsvimg.copy()
		hsvimgcopy = np.float32(hsvimgcopy)
		scale = 0.01
		H, S, V = cv2.split(hsvimgcopy)
		S = np.clip(S * scale, 0, 255)
		hsvimgcopy = cv2.merge((H, S, V))
		hsvimgcopy = cv2.cvtColor(hsvimgcopy, cv2.COLOR_HSV2BGR)
		cv2.imwrite("desaturated.png", hsvimgcopy)
		img2 = cv2.imread("desaturated.png")
		cv2.imshow("Desaturated", img2)
		mb.showinfo("Success!", "Desaturated image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("desaturated.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def adjustthreshold():
	try:
		img = cv2.imread("originalimage.png", cv2.IMREAD_GRAYSCALE)
		amt = sd.askinteger("Threshold", "Enter the threshold value:")
		retval, threshold = cv2.threshold(img, amt, 255, cv2.THRESH_BINARY)
		cv2.imwrite("threshold.png", threshold)
		img2 = cv2.imread("threshold.png")
		cv2.imshow("Threshold", img2)
		mb.showinfo("Success!", "Threshold image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("threshold.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def solarize():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		imgcopy = img.copy()
		imgcopy = np.float32(imgcopy)
		H, S, V = cv2.split(imgcopy)
		V = np.clip(255 - V, 0, 255)
		imgcopy = cv2.merge((H, S, V))
		imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_HSV2BGR)
		cv2.imwrite("solarized.png", imgcopy)
		img2 = cv2.imread("solarized.png")
		cv2.imshow("Solarized", img2)
		mb.showinfo("Success!", "Solarized image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("solarized.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def pixelate():
	try:
		img = Image.open("originalimage.png")
		size = sd.askinteger("Pixelate", "Enter the size of the pixels:")
		img = img.resize((img.size[0] // size, img.size[1] // size), Image.Resampling.NEAREST)
		img = img.resize((img.size[0] * size, img.size[1] * size), Image.Resampling.NEAREST)
		img.save("pixelated.png")
		img2 = cv2.imread("pixelated.png")
		cv2.imshow("Pixelated", img2)
		mb.showinfo("Success!", "Pixelated image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("pixelated.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def oilpainting():
	try:
		cv2.imread("originalimage.png", cv2.IMREAD_GRAYSCALE)
		img = cv2.imread("originalimage.png")
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
		morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		result = cv2.normalize(morph, None, 20, 255, cv2.NORM_MINMAX)
		cv2.imwrite("oilpainted.png", result)
		img2 = cv2.imread("oilpainted.png")
		cv2.imshow("Oilpainted", img2)
		mb.showinfo("Success!", "Oilpainted image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("oilpainted.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def posterize():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		imgcopy = img.copy()
		imgcopy = np.float32(imgcopy)
		H, S, V = cv2.split(imgcopy)
		S = np.clip(S // 32 * 32, 0, 255)
		imgcopy = cv2.merge((H, S, V))
		imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_HSV2BGR)
		cv2.imwrite("posterized.png", imgcopy)
		img2 = cv2.imread("posterized.png")
		cv2.imshow("Posterized", img2)
		mb.showinfo("Success!", "Posterized image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("posterized.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def prewitt():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
			imgx = cv2.filter2D(gray, -1, kernelx)
			absoul = np.absolute(imgx)
			normalize = cv2.normalize(absoul, None, 0, 255, cv2.NORM_MINMAX)
			invert = cv2.bitwise_not(normalize)
			resize = cv2.resize(invert, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
			thresh, blackandwhite = cv2.threshold(resize, 190, 255, cv2.THRESH_BINARY)
			xStart = 664
			yStart = 230
			cv2.imwrite("blackandwhite.png", blackandwhite)
			cv2.imshow("Black and White", blackandwhite)
			mb.showinfo("Success!", "Prewitt image created.")
			cv2.destroyAllWindows()
			mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
			time.sleep(5)
			try:
				for y in range(len(blackandwhite)):
					row = blackandwhite[y]
					for x in range(len(row)):
						if row[x] == 0:
							pyautogui.click(xStart + x, yStart + y, _pause=False)
							time.sleep(0.01)
				mb.showinfo("Success!", "Autodraw done.")
				asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
				if asktodelete == True:
					cv2.destroyAllWindows()
					os.remove("blackandwhite.png")
			except pyautogui.FailSafeException:
				mb.showerror("Failsafe!", "Failsafe triggered. Stopping autodraw...")
				return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
			imgx = cv2.filter2D(gray, -1, kernelx)
			absoul = np.absolute(imgx)
			normalize = cv2.normalize(absoul, None, 0, 255, cv2.NORM_MINMAX)
			cv2.imwrite("prewittx.png", normalize)
			img2 = cv2.imread("prewittx.png")
			cv2.imshow("Prewitt X", img2)
			mb.showinfo("Success!", "Prewitt image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter images?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("prewittx.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def harris():
	try:
		mb.showinfo("Info!", "This is not a filter, but a corner detector.\nIt will detect corners in the image.")
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray, 2, 3, 0.04)
		dst = cv2.dilate(dst, None)
		img[dst > 0.01 * dst.max()] = [0, 0, 255]
		cv2.imwrite("harris.png", img)
		img2 = cv2.imread("harris.png")
		cv2.imshow("Harris", img2)
		mb.showinfo("Success!", "Harris image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("harris.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def sobel():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			xStart = 664
			yStart = 230
			mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
			scaleabs = cv2.convertScaleAbs(sobel)
			invert = cv2.bitwise_not(scaleabs)
			resize = cv2.resize(invert, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
			thresh, blackandwhite = cv2.threshold(resize, 230, 255, cv2.THRESH_BINARY)
			cv2.imwrite("sobel-resized.png", blackandwhite)
			img2 = cv2.imread("sobel-resized.png")
			cv2.imshow("Sobel", img2)
			mb.showinfo("Success!", "Sobel image created.")
			cv2.destroyAllWindows()
			mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
			time.sleep(5)
			try:
				for y in range(len(blackandwhite)):
					row = blackandwhite[y]
					for x in range(len(row)):
						if row[x] == 0:
							pyautogui.click(xStart + x, yStart + y, _pause=False)
							time.sleep(0.01)
				mb.showinfo("Success!", "Autodraw done.")
				asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
				if asktodelete == True:
					cv2.destroyAllWindows()
					os.remove("sobel-resized.png")
			except pyautogui.FailSafeException:
				mb.showerror("Failsafe!", "Failsafe triggered. Stopping autodraw...")
				return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
			img = cv2.convertScaleAbs(img)
			cv2.imwrite("sobel.png", img)
			img2 = cv2.imread("sobel.png")
			cv2.imshow("Sobel", img2)
			mb.showinfo("Success!", "Sobel image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter images?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("sobel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def canny():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			try:
				mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
				img = cv2.imread("originalimage.png")
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				canny = cv2.Canny(gray, 100, 200)
				invert = cv2.bitwise_not(canny)
				xStart = 664
				yStart = 230
				resize = cv2.resize(invert, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
				thresh, blackandwhite = cv2.threshold(resize, 140, 255, cv2.THRESH_BINARY)
				cv2.imwrite("canny-resized.png", blackandwhite)
				cv2.imshow("Canny", blackandwhite)
				mb.showinfo("Success!", "Canny image created.")
				cv2.destroyAllWindows()
				mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
				img2 = cv2.imread("canny-resized.png")
				time.sleep(5)
				try:
					for y in range(len(blackandwhite)):
						row = blackandwhite[y]
						for x in range(len(row)):
							if row[x] == 0:
								pyautogui.click(xStart + x, yStart + y, _pause=False)
								time.sleep(0.01)
					mb.showinfo("Success!", "Autodraw done.")
					asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter images?")
					if asktodelete == True:
						cv2.destroyAllWindows()
						os.remove("canny-resized.png")
				except pyautogui.FailSafeException:
					mb.showwarning("Failsafe!", "The failsafe has been triggered. Stopping autodraw...")
					return
				except:
					mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			canny = cv2.Canny(gray, 100, 200)
			invert = cv2.bitwise_not(canny)
			cv2.imwrite("canny.png", invert)
			img2 = cv2.imread("canny.png")
			cv2.imshow("Canny", img2)
			mb.showinfo("Success!", "Canny image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("canny.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def highpass():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (5, 5), 0)
		img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (5, 5), 0), -4, 128)
		cv2.imwrite("highpass.png", img)
		img2 = cv2.imread("highpass.png")
		cv2.imshow("Highpass", img2)
		mb.showinfo("Success!", "Highpass image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("highpass.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def laplace():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			try:
				mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
				img = cv2.imread("originalimage.png")
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				laplace = cv2.Laplacian(gray, cv2.CV_64F)
				scale = cv2.convertScaleAbs(laplace)
				invert = cv2.bitwise_not(scale)
				xStart = 664
				yStart = 230
				resize = cv2.resize(invert, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
				thresh, blackandwhite = cv2.threshold(resize, 140, 255, cv2.THRESH_BINARY)
				cv2.imwrite("laplace-resized.png", blackandwhite)
				cv2.imshow("Laplace", blackandwhite)
				mb.showinfo("Success!", "Laplace image created.")
				cv2.destroyAllWindows()
				mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
				img2 = cv2.imread("laplace-resized.png")
				time.sleep(5)
				try:
					for y in range(len(blackandwhite)):
						row = blackandwhite[y]
						for x in range(len(row)):
							if row[x] == 0:
								pyautogui.click(xStart + x, yStart + y, _pause=False)
								time.sleep(0.01)
					mb.showinfo("Success!", "Autodraw done.")
					asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter images?")
					if asktodelete == True:
						cv2.destroyAllWindows()
						os.remove("laplace-resized.png")
				except pyautogui.FailSafeException:
					mb.showwarning("Failsafe!", "The failsafe has been triggered. Stopping autodraw...")
					return
				except:
					mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.Laplacian(img, cv2.CV_64F)
			img = cv2.convertScaleAbs(img)
			cv2.imwrite("laplace.png", img)
			img2 = cv2.imread("laplace.png")
			cv2.imshow("Laplace", img2)
			mb.showinfo("Success!", "Laplace image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("laplace.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def equalization():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		equ = cv2.equalizeHist(img)
		cv2.imwrite("histogramequalization.png", equ)
		img2 = cv2.imread("histogramequalization.png")
		cv2.imshow("Histogram Equalization", img2)
		mb.showinfo("Success!", "Histogram Equalization image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("histogramequalization.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def skeleton():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			xStart = 664
			yStart = 230
			mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
			scale = cv2.convertScaleAbs(sobel)
			dilate = cv2.dilate(scale, None, iterations=2)
			erode = cv2.erode(dilate, None, iterations=2)
			invert = cv2.bitwise_not(erode)
			resize = cv2.resize(invert, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
			thresh, blackandwhite = cv2.threshold(resize, 50, 255, cv2.THRESH_BINARY)
			cv2.imwrite("skeleton-resize.png", resize)
			img2 = cv2.imread("skeleton-resize.png")
			cv2.imshow("Skeleton", img2)
			mb.showinfo("Success!", "Skeleton image created.")
			cv2.destroyAllWindows()
			mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
			time.sleep(5)
			try:
				for y in range(len(blackandwhite)):
					row = blackandwhite[y]
					for x in range(len(row)):
						if row[x] == 0:
							pyautogui.click(xStart + x, yStart + y, _pause=False)
							time.sleep(0.01)
				mb.showinfo("Success!", "Skeleton image drawn.")
				asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
				if asktodelete == True:
					cv2.destroyAllWindows()
					os.remove("skeleton-resize.png")
			except pyautogui.FailSafeException:
				mb.showwarning("Failsafe!", "The failsafe has been triggered. Stopping autodraw...")
				return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
			img = cv2.convertScaleAbs(img)
			img = cv2.dilate(img, None, iterations=2)
			img = cv2.erode(img, None, iterations=2)
			cv2.imwrite("skeleton.png", img)
			img2 = cv2.imread("skeleton.png")
			cv2.imshow("Skeleton", img2)
			mb.showinfo("Success!", "Skeleton image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("skeleton.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def thinning():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		cv2.imwrite("thinning.png", img)
		img2 = cv2.imread("thinning.png")
		cv2.imshow("Thinning", img2)
		mb.showinfo("Success!", "Thinning image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("thinning.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def thickening():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		img = cv2.dilate(img, None, iterations=2)
		img = cv2.erode(img, None, iterations=2)
		cv2.imwrite("thickening.png", img)
		img2 = cv2.imread("thickening.png")
		cv2.imshow("Thickening", img2)
		mb.showinfo("Success!", "Thickening image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("thickening.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def linefilter():
	try:
		img = cv2.imread("originalimage.png")
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
			time.sleep(0.50)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			line = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
			resized = cv2.resize(line, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
			thresh, blackandwhite = cv2.threshold(resized, 140, 255, cv2.THRESH_BINARY)
			cv2.imwrite("linefilter-resized.png", blackandwhite)
			cv2.imshow("Line Filter", blackandwhite)
			mb.showinfo("Success!", "Image resized. Now the program will autodraw the image.")
			cv2.destroyAllWindows()
			cv2.imread("linefilter-resized.png")
			mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
			time.sleep(5)
			xStart = 664
			yStart = 230
			try:
				for y in range(len(blackandwhite)):
					row = blackandwhite[y]
					for x in range(len(row)):
						if row[x] == 0:
							pyautogui.click(xStart + x, yStart + y, _pause=False)
							time.sleep(0.01)
				mb.showinfo("Success!", "Image drawn!")
				asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
				if asktodelete == True:
					cv2.destroyAllWindows()
					os.remove("linefilter-resized.png")
			except pyautogui.FailSafeException:
				mb.showwarning("Failsafe triggered!", "Failsafe has been triggered, stopping...")
				return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
				return
		else:
			cv2.imread("originalimage.png")
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
			cv2.imwrite("linefilter.png", img)
			img2 = cv2.imread("linefilter.png")
			cv2.imshow("Line Filter", img2)
			mb.showinfo("Success!", "Line Filter image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("linefilter.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def dft():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)
		magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
		cv2.imwrite("dft.png", magnitude_spectrum)
		img2 = cv2.imread("dft.png")
		cv2.imshow("DFT", img2)
		mb.showinfo("Success!", "DFT image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("dft.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def dst():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		dst = cv2.dilate(img, None, iterations=2)
		dst = cv2.erode(dst, None, iterations=2)
		cv2.imwrite("dst.png", dst)
		img2 = cv2.imread("dst.png")
		cv2.imshow("DST", img2)
		mb.showinfo("Success!", "DST image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("dst.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def mirror():
	try:
		img = cv2.imread("originalimage.png")
		flip = cv2.flip(img, 1)
		cv2.imwrite("mirror.png", flip)
		img2 = cv2.imread("mirror.png")
		cv2.imshow("Mirror", img2)
		mb.showinfo("Success!", "Mirror image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("mirror.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def otsu():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		cv2.imwrite("otsu.png", thresh)
		img2 = cv2.imread("otsu.png")
		cv2.imshow("Otsu", img2)
		mb.showinfo("Success!", "Otsu image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("otsu.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def upscale():
	try:
		img = cv2.imread("originalimage.png")
		resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite("upscale.png", resized)
		img2 = cv2.imread("upscale.png")
		cv2.imshow("Upscale", img2)
		mb.showinfo("Success!", "Upscale image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("upscale.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def downscale():
	try:
		img = cv2.imread("originalimage.png")
		resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite("downscale.png", resized)
		img2 = cv2.imread("downscale.png")
		cv2.imshow("Downscale", img2)
		mb.showinfo("Success!", "Downscale image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("downscale.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def noise():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		noise = np.random.randint(0, 255, gray.shape)
		noise = noise.astype(np.uint8)
		noise = cv2.add(gray, noise)
		cv2.imwrite("noise.png", noise)
		img2 = cv2.imread("noise.png")
		cv2.imshow("Noise", img2)
		mb.showinfo("Success!", "Noise image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("noise.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def highboost():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		dst = cv2.Laplacian(gray, cv2.CV_64F)
		dst = np.uint8(np.absolute(dst))
		cv2.imwrite("highboost.png", dst)
		img2 = cv2.imread("highboost.png")
		cv2.imshow("Highboost", img2)
		mb.showinfo("Success!", "Highboost image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("highboost.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def tophat():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kernel = np.ones((5, 5), np.uint8)
		opening = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
		cv2.imwrite("tophat.png", opening)
		img2 = cv2.imread("tophat.png")
		cv2.imshow("Tophat", img2)
		mb.showinfo("Success!", "Tophat image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("tophat.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def blackhat():
	try:
		img = cv2.imread("originalimage.png")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kernel = np.ones((5, 5), np.uint8)
		closing = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
		cv2.imwrite("blackhat.png", closing)
		img2 = cv2.imread("blackhat.png")
		cv2.imshow("Blackhat", img2)
		mb.showinfo("Success!", "Blackhat image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("blackhat.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
	
def powerlaw():
	try:
		img = cv2.imread("originalimage.png")
		power = sd.askinteger("Power", "Power:")
		img = cv2.pow(img, power)
		cv2.imwrite("powerlaw.png", img)
		img2 = cv2.imread("powerlaw.png")
		cv2.imshow("Powerlaw", img2)
		mb.showinfo("Success!", "Powerlaw image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("powerlaw.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def greenchannel():
	try:
		img = cv2.imread("originalimage.png")
		img[:,:,0] = 0
		img[:,:,2] = 0
		cv2.imwrite("greenchannel.png", img)
		img2 = cv2.imread("greenchannel.png")
		cv2.imshow("Green Channel", img2)
		mb.showinfo("Success!", "Green channel image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("greenchannel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())


def yiq():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		img[:,:,0] = img[:,:,0] * 1.5
		img[:,:,1] = img[:,:,1] * 1.5
		img[:,:,2] = img[:,:,2] * 1.5
		cv2.imwrite("yiq.png", img)
		img2 = cv2.imread("yiq.png")
		cv2.imshow("YIQ", img2)
		mb.showinfo("Success!", "YIQ image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("yiq.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def yuv():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		img[:,:,0] = img[:,:,0] * 1.5
		img[:,:,1] = img[:,:,1] * 1.5
		img[:,:,2] = img[:,:,2] * 1.5
		cv2.imwrite("yuv.png", img)
		img2 = cv2.imread("yuv.png")
		cv2.imshow("YUV", img2)
		mb.showinfo("Success!", "YUV image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("yuv.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def hsl():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img[:,:,0] = img[:,:,0] * 1.5
		img[:,:,1] = img[:,:,1] * 1.5
		img[:,:,2] = img[:,:,2] * 1.5
		cv2.imwrite("hsl.png", img)
		img2 = cv2.imread("hsl.png")
		cv2.imshow("HSL", img2)
		mb.showinfo("Success!", "HSL image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("hsl.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def hls():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		img[:,:,0] = img[:,:,0] * 1.5
		img[:,:,1] = img[:,:,1] * 1.5
		img[:,:,2] = img[:,:,2] * 1.5
		cv2.imwrite("hls.png", img)
		img2 = cv2.imread("hls.png")
		cv2.imshow("HLS", img2)
		mb.showinfo("Success!", "HLS image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("hls.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def cie():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
		img[:,:,0] = img[:,:,0] * 1.5
		img[:,:,1] = img[:,:,1] * 1.5
		img[:,:,2] = img[:,:,2] * 1.5
		cv2.imwrite("cie.png", img)
		img2 = cv2.imread("cie.png")
		cv2.imshow("CIE", img2)
		mb.showinfo("Success!", "CIE image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("cie.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def corrupt():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (random.randint(0,200), random.randint(0,200), random.randint(0,200))
		cv2.imwrite("corrupted.png", img)
		img2 = cv2.imread("corrupted.png")
		cv2.imshow("Corrupted", img2)
		mb.showinfo("Success!", "Corrupted image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("corrupted.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def stippling():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				if x % 2 == 0:
					img[y,x] = (random.randint(0,200), random.randint(0,200), random.randint(0,200))
		cv2.imwrite("stippled.png", img)
		img2 = cv2.imread("stippled.png")
		cv2.imshow("Stippled", img2)
		mb.showinfo("Success!", "Stippled image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("stippled.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def contraststretch():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (int(img[y,x][0]*1.5), int(img[y,x][1]*1.5), int(img[y,x][2]*1.5))
		cv2.imwrite("contraststretched.png", img)
		img2 = cv2.imread("contraststretched.png")
		cv2.imshow("Contrast Stretched", img2)
		mb.showinfo("Success!", "Contrast Stretched image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("contraststretched.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def quantize():
	try:
		mb.showwarning("Warning!", "This filter will take a while to process.")
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (int(img[y,x][0]/2)*2, int(img[y,x][1]/2)*2, int(img[y,x][2]/2)*2)
		cv2.imwrite("quantized.png", img)
		img2 = cv2.imread("quantized.png")
		cv2.imshow("Quantized", img2)
		mb.showinfo("Success!", "Quantized image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("quantized.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def acidtrip():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (int(math.pow(img[y,x][0], 1.5)), int(math.pow(img[y,x][1], 1.5)), int(math.pow(img[y,x][2], 1.5)))
		cv2.imwrite("acidtrip.png", img)
		img2 = cv2.imread("acidtrip.png")
		cv2.imshow("Acid Trip", img2)
		mb.showinfo("Success!", "Acid Trip image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("acidtrip.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def bluechannel():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (img[y,x][0], 0, 0)
		cv2.imwrite("bluechannel.png", img)
		img2 = cv2.imread("bluechannel.png")
		cv2.imshow("Blue Channel", img2)
		mb.showinfo("Success!", "Blue Channel image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("bluechannel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def redchannel():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (0, 0, img[y,x][2])
		cv2.imwrite("redchannel.png", img)
		img2 = cv2.imread("redchannel.png")
		cv2.imshow("Red Channel", img2)
		mb.showinfo("Success!", "Red Channel image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("redchannel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def cyanchannel():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (img[y,x][0], img[y,x][2], 0)
		cv2.imwrite("cyanchannel.png", img)
		img2 = cv2.imread("cyanchannel.png")
		cv2.imshow("Cyan Channel", img2)
		mb.showinfo("Success!", "Cyan Channel image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("cyanchannel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def purplechannel():
	try:
		img = cv2.imread("originalimage.png")
		height, width, channels = img.shape
		for x in range(0, width):
			for y in range(0, height):
				img[y,x] = (img[y,x][1], 0, img[y,x][2])
		cv2.imwrite("purplechannel.png", img)
		img2 = cv2.imread("purplechannel.png")
		cv2.imshow("Purple Channel", img2)
		mb.showinfo("Success!", "Purple Channel image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("purplechannel.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def scharr():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.Scharr(img, cv2.CV_8U, 1, 0)
		cv2.imwrite("scharr.png", img)
		img2 = cv2.imread("scharr.png")
		cv2.imshow("Scharr", img2)
		mb.showinfo("Success!", "Scharr image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("scharr.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def bubbles():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.bilateralFilter(img, 9, 75, 75)
		cv2.imwrite("bubbles.png", img)
		img2 = cv2.imread("bubbles.png")
		cv2.imshow("Bubbles", img2)
		mb.showinfo("Success!", "Bubbles image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("bubbles.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def hough():
	try:
		img = cv2.imread("originalimage.png")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (5, 5), 0)
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=30, minRadius=0, maxRadius=0)
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
			cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
		cv2.imwrite("hough.png", img)
		img2 = cv2.imread("hough.png")
		cv2.imshow("Hough", img2)
		mb.showinfo("Success!", "Hough image created.")
		asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
		if asktodelete == True:
			cv2.destroyAllWindows()
			os.remove("hough.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

def mandelbrot():
	try:
		autodraw = mb.askyesno("Auto Draw?", "Do you want to use the autodraw mode?")
		if autodraw == True:
			mb.showwarning("Resize!", "Unlike the Pencil Sketch autodraw mode, this filter will need to resize the image to avoid taking too much time.")
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			filter = cv2.bilateralFilter(gray, 9, 75, 75)
			blur = cv2.medianBlur(filter, 5)
			blur2 = cv2.GaussianBlur(blur, (5, 5), 0)
			canny = cv2.Canny(blur2, 100, 200)
			invert = cv2.bitwise_not(canny)
			resized = cv2.resize(invert, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
			thresh, blackandwhite = cv2.threshold(resized, 140, 255, cv2.THRESH_BINARY)
			cv2.imwrite("mandelbrot-resized.png", blackandwhite)
			img2 = cv2.imread("mandelbrot-resized.png")
			cv2.imshow("Mandelbrot", img2)
			mb.showinfo("Success!", "Mandelbrot image created.")
			xStart = 664
			yStart = 230
			cv2.destroyAllWindows()
			mb.showwarning("Starting...", "Depending on the image, this will take a while. Please be patient.\n\nOnce you click OK, you will have 5 seconds to go onto a painting app.\n\nTo stop drawing at any time, force your mouse to the top left corner of your screen, which will trigger the failsafe.")
			time.sleep(5)
			try:
				for y in range(len(blackandwhite)):
					row = blackandwhite[y]
					for x in range(len(row)):
						if row[x] == 0:
							pyautogui.click(xStart + x, yStart + y, _pause=False)
							time.sleep(0.01)
				mb.showinfo("Success!", "Mandelbrot image created.")
				asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
				if asktodelete == True:
					cv2.destroyAllWindows()
					os.remove("mandelbrot-resized.png")
			except pyautogui.FailSafeException:
				mb.showwarning("Failsafe!", "The failsafe has been triggered. Stopping autodraw...")
				return
			except:
				mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())
		else:
			img = cv2.imread("originalimage.png")
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			filter = cv2.bilateralFilter(gray, 9, 75, 75)
			blur = cv2.medianBlur(filter, 5)
			blur2 = cv2.GaussianBlur(blur, (5, 5), 0)
			canny = cv2.Canny(blur2, 100, 200)
			cv2.imwrite("mandelbrot.png", canny)
			img2 = cv2.imread("mandelbrot.png")
			cv2.imshow("Mandelbrot", img2)
			mb.showinfo("Success!", "Mandelbrot image created.")
			asktodelete = mb.askyesno("Delete?", "Do you want to delete the filter image?")
			if asktodelete == True:
				cv2.destroyAllWindows()
				os.remove("mandelbrot.png")
	except:
		mb.showerror("Error!", "Something went wrong. Error:\n" + traceback.format_exc())

if not os.path.isfile("readme.txt"):
	messagebox(None, "Version 1.60.1:\n\n-Added a lot more filters and effects.\n\n-Added some autodraw modes for some filters, with more planned eventually.\n\n-Bug fixes.\n\n\nThis changelog will only appear once unless the readme.txt file has been deleted.", "Changelog", 0)
with open("readme.txt", "w") as f:
	f.write("Hey there!")
	f.write("\n\n")
	f.write("If you delete this file, you will be greeted by the changelog.")
	f.write("\n\n")
	f.write("However, if you keep this file, you will not be greeted by it again.")
	f.close()

root = Tk()
root.geometry('880x770')
root.configure(background='#737373')
root.title('SpiltCoffee Photo Filter Tool')
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
Label(root, text='SpiltCoffee Photo Filter Tool', bg='#737373', font=('arial', 30, 'normal')).place(x=111, y=6)
Label(root, text='This tool will convert any photo into a filter of your choosing!', bg='#737373', font=('arial', 20, 'normal')).place(x=27, y=67)
Label(root, text='First, download your photo:', bg='#737373', font=('arial', 20, 'normal')).place(x=6, y=118)
Label(root, text='Autodraw filters are bolded.', bg='#737373', font=('arial', 20, 'italic')).place(x=215, y=167)
Label(root, text='The available filters/effects are below:', bg='#737373', font=('arial', 20, 'normal')).place(x=164, y=206)
Button(root, text='Choose image', bg='#999999', font=('arial', 15, 'normal'), command=imagepicker).place(x=398, y=117)
Button(root, text='Black and White', bg='#999999', font=('arial', 14, 'normal'), command=blackandwhite).place(x=163, y=246)
Button(root, text='Cartoon', bg='#999999', font=('arial', 14, 'normal'), command=cartoon).place(x=346, y=246)
Button(root, text='Invert', bg='#999999', font=('arial', 14, 'normal'), command=invert).place(x=446, y=246)
Button(root, text='Pencil sketch', bg='#999999', font=('arial', 13, 'bold'), command=pencilsketch).place(x=521, y=246)
Button(root, text='Sepia', bg='#999999', font=('arial', 14, 'normal'), command=sepia).place(x=176, y=296)
Button(root, text='Blur', bg='#999999', font=('arial', 14, 'normal'), command=blur).place(x=256, y=296)
Button(root, text='Contour', bg='#999999', font=('arial', 14, 'normal'), command=contour).place(x=340, y=486)
Button(root, text='Detailed Image', bg='#999999', font=('arial', 14, 'normal'), command=detailed).place(x=317, y=296)
Button(root, text='Edge enhance', bg='#999999', font=('arial', 14, 'normal'), command=edgeenhance).place(x=255, y=435)
Button(root, text='Emboss', bg='#999999', font=('arial', 14, 'normal'), command=emboss).place(x=421, y=435)
Button(root, text='Find edges', bg='#999999', font=('arial', 14, 'normal'), command=edgefinder).place(x=521, y=435)
Button(root, text='Sharpen', bg='#999999', font=('arial', 14, 'normal'), command=sharpen).place(x=491, y=296)
Button(root, text='Smooth', bg='#999999', font=('arial', 14, 'normal'), command=smooth).place(x=246, y=486)
Button(root, text='Summer', bg='#999999', font=('arial', 14, 'normal'), command=summer).place(x=435, y=486)
Button(root, text='Winter', bg='#999999', font=('arial', 14, 'normal'), command=winter).place(x=535, y=486)
Button(root, text='Glitch', bg='#999999', font=('arial', 14, 'normal'), command=glitch).place(x=599, y=296)
Button(root, text='ASCII Art', bg='#999999', font=('arial', 14, 'normal'), command=asciiart).place(x=241, y=535)
Button(root, text='Moon', bg='#999999', font=('arial', 14, 'normal'), command=moon).place(x=169, y=344)
Button(root, text='Clarendon', bg='#999999', font=('arial', 14, 'normal'), command=clarendon).place(x=247, y=344)
Button(root, text='Laplacian', bg='#999999', font=('arial', 14, 'normal'), command=laplacian).place(x=375, y=344)
Button(root, text='Kelvin', bg='#999999', font=('arial', 14, 'normal'), command=kelvin).place(x=493, y=344)
Button(root, text='X-Pro', bg='#999999', font=('arial', 14, 'normal'), command=xpro).place(x=573, y=344)
Button(root, text='Erode', bg='#999999', font=('arial', 14, 'normal'), command=erode).place(x=356, y=535)
Button(root, text='Dilate', bg='#999999', font=('arial', 14, 'normal'), command=dilate).place(x=435, y=535)
Button(root, text='Gamma', bg='#999999', font=('arial', 14, 'normal'), command=gamma).place(x=511, y=535)
Button(root, text='Contrast', bg='#999999', font=('arial', 14, 'normal'), command=contrast).place(x=650, y=435)
Button(root, text='Brightness', bg='#999999', font=('arial', 14, 'normal'), command=brightness).place(x=618, y=486)
Button(root, text='HSV', bg='#999999', font=('arial', 14, 'normal'), command=hsv).place(x=606, y=535)
Button(root, text='Rotate', bg='#999999', font=('arial', 14, 'normal'), command=rotate).place(x=163, y=435)
Button(root, text='Affine', bg='#999999', font=('arial', 14, 'normal'), command=affine).place(x=170, y=486)
Button(root, text='Inverse Affine', bg='#999999', font=('arial', 14, 'normal'), command=inverseaffine).place(x=80, y=535)
Button(root, text='Saturated', bg='#999999', font=('arial', 14, 'normal'), command=saturate).place(x=670, y=535)
Button(root, text='Hue', bg='#999999', font=('arial', 14, 'normal'), command=hue).place(x=760, y=435)
Button(root, text='CCA', bg='#999999', font=('arial', 14, 'normal'), command=cca).place(x=104, y=486)
Button(root, text='Desaturate', bg='#999999', font=('arial', 14, 'normal'), command=desaturate).place(x=31, y=435)
Button(root, text='Adjust Threshold', bg='#999999', font=('arial', 14, 'normal'), command=adjustthreshold).place(x=654, y=344)
Button(root, text='Solarize', bg='#999999', font=('arial', 14, 'normal'), command=solarize).place(x=675, y=246)
Button(root, text='Pixelate', bg='#999999', font=('arial', 14, 'normal'), command=pixelate).place(x=677, y=296)
Button(root, text='Oil Painting', bg='#999999', font=('arial', 14, 'normal'), command=oilpainting).place(x=42, y=296)
Button(root, text='Posterize', bg='#999999', font=('arial', 14, 'normal'), command=posterize).place(x=52, y=344)
Button(root, text='Prewitt', bg='#999999', font=('arial', 13, 'bold'), command=prewitt).place(x=75, y=246)
Button(root, text='Harris', bg='#999999', font=('arial', 14, 'normal'), command=harris).place(x=80, y=583)
Button(root, text='Sobel', bg='#999999', font=('arial', 13, 'bold'), command=sobel).place(x=167, y=583)
Button(root, text='Canny', bg='#999999', font=('arial', 13, 'bold'), command=canny).place(x=246, y=583)
Button(root, text='High Pass', bg='#999999', font=('arial', 14, 'normal'), command=highpass).place(x=333, y=583)
Button(root, text='Laplace', bg='#999999', font=('arial', 13, 'bold'), command=laplace).place(x=459, y=583)
Button(root, text='Equalization', bg='#999999', font=('arial', 14, 'normal'), command=equalization).place(x=80, y=630)
Button(root, text='Skeleton', bg='#999999', font=('arial', 13, 'bold'), command=skeleton).place(x=560, y=583)
Button(root, text='Thin', bg='#999999', font=('arial', 13, 'normal'), command=thinning).place(x=146, y=391)
Button(root, text='Thicken', bg='#999999', font=('arial', 13, 'normal'), command=thickening).place(x=52, y=391)
Button(root, text='Line Filter', bg='#999999', font=('arial', 12, 'bold'), command=linefilter).place(x=208, y=391)
Button(root, text='DFT', bg='#999999', font=('arial', 13, 'normal'), command=dft).place(x=323, y=391)
Button(root, text='DST', bg='#999999', font=('arial', 13, 'normal'), command=dst).place(x=387, y=391)
Button(root, text='Mirror', bg='#999999', font=('arial', 13, 'normal'), command=mirror).place(x=453, y=391)
Button(root, text='Otsu', bg='#999999', font=('arial', 13, 'normal'), command=otsu).place(x=533, y=391)
Button(root, text='Upscale', bg='#999999', font=('arial', 13, 'normal'), command=upscale).place(x=600, y=391)
Button(root, text='Downscale', bg='#999999', font=('arial', 13, 'normal'), command=downscale).place(x=700, y=391)
Button(root, text='Noise', bg='#999999', font=('arial', 14, 'normal'), command=noise).place(x=746, y=486)
Button(root, text='High Boost', bg='#999999', font=('arial', 14, 'normal'), command=highboost).place(x=670, y=583)
Button(root, text='Tophat', bg='#999999', font=('arial', 14, 'normal'), command=tophat).place(x=222, y=630)
Button(root, text='Blackhat', bg='#999999', font=('arial', 14, 'normal'), command=blackhat).place(x=313, y=630)
Button(root, text='Power Law', bg='#999999', font=('arial', 14, 'normal'), command=powerlaw).place(x=479, y=630)
Button(root, text='Green', bg='#999999', font=('arial', 14, 'normal'), command=greenchannel).place(x=610, y=630)
Button(root, text='YIQ', bg='#999999', font=('arial', 14, 'normal'), command=yiq).place(x=80, y=678)
Button(root, text='YUV', bg='#999999', font=('arial', 14, 'normal'), command=yuv).place(x=139, y=678)
Button(root, text='HSL', bg='#999999', font=('arial', 14, 'normal'), command=hsl).place(x=207, y=678)
Button(root, text='HLS', bg='#999999', font=('arial', 14, 'normal'), command=hls).place(x=273, y=678)
Button(root, text='CIE', bg='#999999', font=('arial', 14, 'normal'), command=cie).place(x=339, y=678)
Button(root, text='Corrupt', bg='#999999', font=('arial', 14, 'normal'), command=corrupt).place(x=399, y=678)
Button(root, text='Stippling', bg='#999999', font=('arial', 14, 'normal'), command=stippling).place(x=497, y=678)
Button(root, text='Contrast Stretch', bg='#999999', font=('arial', 14, 'normal'), command=contraststretch).place(x=601, y=678)
Button(root, text='Quantize', bg='#999999', font=('arial', 14, 'normal'), command=quantize).place(x=692, y=630)
Button(root, text='Acid Trip', bg='#999999', font=('arial', 14, 'normal'), command=acidtrip).place(x=80, y=726)
Button(root, text='Blue', bg='#999999', font=('arial', 14, 'normal'), command=bluechannel).place(x=418, y=630)
Button(root, text='Red', bg='#999999', font=('arial', 14, 'normal'), command=redchannel).place(x=191, y=726)
Button(root, text='Cyan', bg='#999999', font=('arial', 14, 'normal'), command=cyanchannel).place(x=255, y=726)
Button(root, text='Purple', bg='#999999', font=('arial', 14, 'normal'), command=purplechannel).place(x=330, y=726)
Button(root, text='Scharr', bg='#999999', font=('arial', 14, 'normal'), command=scharr).place(x=415, y=726)
Button(root, text='Bubbles', bg='#999999', font=('arial', 14, 'normal'), command=bubbles).place(x=503, y=726)
Button(root, text='Hough', bg='#999999', font=('arial', 14, 'normal'), command=hough).place(x=603, y=726)
Button(root, text='Mandelbrot', bg='#999999', font=('arial', 13, 'bold'), command=mandelbrot).place(x=690, y=726)
root.mainloop()
