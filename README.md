# SpiltCoffee
SpiltCoffee is a massive project of mine. It is an image editor, in GUI form. You can either download an image and put it into the same folder as the py file, or you can use the built in bing image downloader (requires pip module bing-image-downloader to be installed)

#### If any reference of google-image-downloader are still present, remove them in the script. The module doesnt work and is stupid anyway.
#### Same goes if you have it installed on your system, uninstall it.
 
## Instructions:
1. Clone the repo.
2. Run ```pip install -r requirements.txt```
3. Run ```spiltcoffee.py```
4. Profit.

### Can I use this script in a bot?
Yes you can, but running it from the current ```spiltcoffee.py``` file requires 2 things: 

1. Removing **_all_** the references of ```tkinter```, of which there are 2000+ lines of. (Don't worry, I will upload a bot friendly version soon enough.)
2. Installing ```hikari``` and ```hikari-lightbulb``` and adding all hikari required stuff (bot token, imports, etc.)

### If you are missing packages, open the file in something like VSC. Uninstalled packages will have a yellow underline underneath it.
