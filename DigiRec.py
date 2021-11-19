from PIL import Image
from math import sqrt, exp, pi
from os import listdir
from os.path import isfile

def transformToGrayScale(img):
    imgGs = img.copy()
    palette =  img.getpalette()

    pixel = None
    averageValue = None
    for y in range(img.height):
        for x in range(img.width):
            pixel = img.getpixel((x,y))

            if palette != None:
                pixel = palette[pixel : pixel + 3]

            averageValue = int(sum(pixel) / len(pixel))

            imgGs.putpixel((x,y), tuple([averageValue] * 3))
    
    return imgGs

def createHistogramGS(imgGs, quantGrayShades = 256):
    listHistogram = [0] * quantGrayShades

    palette = imgGs.getpalette()

    pixel = None
    for y in range(imgGs.height):
        for x in range(imgGs.width):
            pixel = imgGs.getpixel((x,y))

            if palette != None:
                pixel = palette[pixel : pixel + 3]
           
            listHistogram[pixel[0]] += 1
    
    return listHistogram

def histogramContrastEnhancedGS(img, quantGrayShades = 256):
    imgCE = img.copy()

    listHisto = createHistogramGS(imgCE, quantGrayShades)

    acc = 0
    listSubs = [0] * quantGrayShades

    total = imgCE.height * imgCE.width
    for i in range(quantGrayShades):
        acc += listHisto[i]
        listSubs[i] = int((quantGrayShades - 1) * (acc / total))

    palette = imgCE.getpalette()
    pixel = None
    for y in range(imgCE.height):
        for x in range(imgCE.width):
            pixel = imgCE.getpixel((x,y))

            if palette != None:
                pixel = palette[pixel : pixel + 3]

            imgCE.putpixel((x,y), tuple([listSubs[pixel[0]]] * 3))

    return imgCE

def matrixMultiply(matA, matB):
    if len(matA[0]) != len(matB):
        return None

    matRes = []

    for i in range(len(matA)):
        matRes.append([])
        for j in range(len(matB[0])):
            matRes[i].append(sum( [matA[i][n] * matB[n][j] for n in range(len(matB))] ))

    return matRes

def displayTextMatrix(mat):
    row = ""

    for i in range(len(mat)):
        row = ""

        for j in range(len(mat[0])):
            row += str(mat[i][j]) + ","

        print(row[:-1])

def edgeDetectSobelOperatorGS(img, sobelOpCenterVal):
    imgSo = img.copy()

    kernelX = matrixMultiply([[1],[sobelOpCenterVal],[1]], [[1,0,-1]])
    kernelY = matrixMultiply([[1],[0],[-1]], [[1,sobelOpCenterVal,1]])

    kernelRadius = 1
    xVal = 0
    yVal = 0
    pixelVal = 0

    for y in range(kernelRadius, img.height - kernelRadius):
        for x in range(kernelRadius, img.width - kernelRadius):
            xVal = imgConvolutionMatrixGS(img, (x,y), kernelX)
            yVal = imgConvolutionMatrixGS(img, (x,y), kernelY)

            pixelVal = round(sqrt(xVal ** 2 + yVal ** 2))

            imgSo.putpixel((x,y), tuple([pixelVal] * 3))

    return imgSo

def imgConvolutionMatrixGS(img, coordKernelCenter, kernel):
    kernelRadius = int(len(kernel) / 2)
    coordKernerInit = (coordKernelCenter[0] - kernelRadius, coordKernelCenter[1] - kernelRadius)

    palette = img.getpalette()
    pixel = None

    convVal = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            pixel = img.getpixel((coordKernerInit[0] + j, coordKernerInit[1] + i))

            if palette != None:
                pixel = palette[pixel : pixel + 3]
            
            convVal += pixel[0] * kernel[i][j]

    return convVal

def imgThresholding(img, limiteIniLim = 0, limiteFimLim = 255):
    imgLim = img.copy()

    palette = imgLim.getpalette()
    pixel = None

    for y in range(img.height):
        for x in range(img.width):
            pixel = img.getpixel((x,y))

            if palette != None:
                pixel = palette[pixel : pixel + 3]

            pixel = 0 if pixel[0] >= limiteIniLim and pixel[0] <= limiteFimLim else 255

            imgLim.putpixel((x,y), tuple([pixel] * 3))

    return imgLim

def imgSegment(img):
    boxCoord = {
        "left": {
            "coord" : img.width # min X
            ,"pixelVal": 0 
        }

        ,"top": {
            "coord" : img.height # min Y
            ,"pixelVal": 0 
        }

        ,"rigth": {
            "coord" : 0 #max X
            ,"pixelVal": 0 
        }

        ,"bottom": {
            "coord" : 0 #max Y
            ,"pixelVal": 0 
        }
    }

    palette = img.getpalette()
    pixel = None

    for y in range(img.height):
        for x in range(img.width):
            pixel = img.getpixel((x,y))

            if palette != None:
                pixel = palette[pixel : pixel + 3]

            pixel = pixel[0]

            if boxCoord["left"]["pixelVal"] >= pixel and boxCoord["left"]["coord"] > x:
                boxCoord["left"]["pixelVal"] = pixel
                boxCoord["left"]["coord"] = x

            if boxCoord["top"]["pixelVal"] >= pixel and boxCoord["top"]["coord"] > y:
                boxCoord["top"]["pixelVal"] = pixel
                boxCoord["top"]["coord"] = y

            if boxCoord["rigth"]["pixelVal"] >= pixel and boxCoord["rigth"]["coord"] < x:
                boxCoord["rigth"]["pixelVal"] = pixel
                boxCoord["rigth"]["coord"] = x

            if boxCoord["bottom"]["pixelVal"] >= pixel and boxCoord["bottom"]["coord"] < y:
                boxCoord["bottom"]["pixelVal"] = pixel
                boxCoord["bottom"]["coord"] = y    

    coord = tuple(
        [boxCoord["left"]["coord"]
        ,boxCoord["top"]["coord"]
        ,boxCoord["rigth"]["coord"]
        ,boxCoord["bottom"]["coord"]]
    )

    return img.crop(coord)

def boxBlur(img, kernelRadius):
    imgME = img.copy()

    kernelDim = kernelRadius * 2 + 1
    
    matA = [ [1] for size in range(kernelDim) ] 

    kernelSize = (kernelDim) ** 2

    matB = [ [1 / kernelSize for size in range(kernelDim)] ]

    kernel = matrixMultiply(matA, matB)

    pixelVal = 0
    for y in range(kernelRadius, imgME.height - kernelRadius):
        for x in range(kernelRadius, imgME.width - kernelRadius):
            pixelVal = round(imgConvolutionMatrixGS(imgME, (x,y), kernel))

            imgME.putpixel((x,y), tuple([pixelVal] * 3))
    
    return imgME

def calcGaussFuncion(x,y,desvio):
    return (1 / (2 * pi * (desvio ** 2))) * exp( -((((x + y) / desvio) ** 2) / 2) )

def gaussBlur(img, kernelRadius, desvio):
    imgGA = img.copy()

    kernelDim = kernelRadius * 2 + 1

    kernelVals = [calcGaussFuncion(x, -kernelRadius, desvio)  for x in range(kernelDim)]

    matA = [[kernelVal] for kernelVal in kernelVals]
    matB = [kernelVals]

    gaussKernel = matrixMultiply(matA, matB)

    pixelVal = 0
    for y in range(kernelRadius, img.height - kernelRadius):
        for x in range(kernelRadius, img.width - kernelRadius):
            pixelVal = round(imgConvolutionMatrixGS(img, (x,y), gaussKernel))

            imgGA.putpixel((x,y), tuple([pixelVal] * 3))

    return imgGA

def treatDigital(imgUntreated, debug = False):
    img = transformToGrayScale(imgUntreated)
    if debug:
        img.save("./debug_steps/digi_GS.jpeg")

    img = histogramContrastEnhancedGS(img, 256)
    if debug:
        img.save("./debug_steps/digi_CE.jpeg")

    #img = gaussBlur(img, 1, .55)
    #if debug:
    #    img.save("./debug_steps/digi_GA.jpeg")

    img = edgeDetectSobelOperatorGS(img, 2)
    if debug:
        img.save("./debug_steps/digi_SO.jpeg")

    img = img.crop((2, 2, img.width - 2, img.height - 2))

    img = imgThresholding(img, 180, 255) # 40% dos pixels mais claros tratados pelo kernel de sobel
    if debug:
        img.save("./debug_steps/digi_TH.jpeg")

    img = imgSegment(img)
    if debug:
        img.save("./debug_steps/digi_SE.jpeg")

    return img

def digiCompare(digis, debug = False):
    max_width = max([digi.width for digi in digis])
    max_height = max([digi.height for digi in digis])

    for i in range(len(digis)):
        digis[i] = digis[i].resize((max_width, max_height))

        if debug:
            digis[i].save("./debug_steps/digi_"+str(i)+"_re.jpeg")

    pixel = []
    palette = [digi.getpalette() for digi in digis]

    equalPixelsCount = 0
    totalBlackPixels = 0
    for y in range(max_height):
        for x in range(max_width):
            pixel = []

            for i in range(len(digis)):
                pixel.append(digis[i].getpixel((x,y)))

                if palette[i] != None:
                    pixel[i] = palette[i][pixel[i] : pixel[i] + 3]

                pixel[i] = pixel[i][0]

            if pixel[0] == 0:
                totalBlackPixels += 1
                equalPixelsCount += 1 if pixel[1] == pixel[0] else 0
            

    return equalPixelsCount / totalBlackPixels

if __name__ == "__main__":
    print("\n\nBem vindo ao DigiRec (Sistema de reconhecimento de digitais)")

    accessLevel = 0

    try: 
        accessLevel = int(input("\nInforme seu nivel de acesso: "))
    except:
        accessLevel = 0

    while accessLevel < 1 or accessLevel > 3:
        print("\nNivel de acesso não reconhecido. Tente novamente!")

        try: 
            accessLevel = int(input("\nInforme seu nivel de acesso: "))
        except:
            accessLevel = 0
        
    if accessLevel == 1:
        print("Acesso autorizado!")
        print("Tenha um bom dia.")
        exit(0)

    trys = 1
    imgInputDigi = None
    pathDigiInput = input("\nAcesso restrito. Por favor informe o caminho a imagem com sua digital: ")

    try:
        imgInputDigi = treatDigital(Image.open(pathDigiInput))
    except:
        imgInputDigi = None

    while (not isfile(pathDigiInput) or imgInputDigi == None) and trys < 3:
        print("Não foi possivel encontrar a imagem. Tente novamente!")

        trys += 1
        pathDigiInput = input("Acesso restrito. Por favor informe o caminho a imagem com sua digital: ")

        try:
            imgInputDigi = treatDigital(Image.open(pathDigiInput))
        except:
            imgInputDigi = None

    if trys >= 3:
        print("\nAcesso negado! Motivo: Não foi possivel encontrar a imagem com a digital")
        exit(1)

    pathDigiAccess = "./digi_access/"+str(accessLevel)+"/"

    digisAccess = listdir(pathDigiAccess)
    
    imgDigiAccess = None
    authorizedAccess = False
    for digiAccess in digisAccess:
        imgDigiAccess = Image.open(pathDigiAccess + digiAccess)

        if digiCompare([imgDigiAccess, imgInputDigi]) > .7:
            authorizedAccess = True
            break

    if authorizedAccess:
        print("\nAcesso autorizado!")
        print("Tenha um bom dia.")
    else:
        print("\nAcesso negado! Motivo: A digital apresentada não tem accesso a esse nível!")



