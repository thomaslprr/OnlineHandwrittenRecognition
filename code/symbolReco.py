################################################################
# symbolReco.py
#
# Program that select hypothesis of segmentation
#
# Author: H. Mouchere, Dec. 2018
# Copyright (c) 2018, Harold Mouchere
################################################################
import sys
import random
import itertools
import sys
import getopt
from convertInkmlToImg import parse_inkml, get_traces_data, getStrokesFromLG, convert_to_imgs, parseLG
from skimage.io import imsave
from model_class import classify, get_model
from model_class_2 import classify2, get_model2
import torchvision.transforms as transforms
import numpy as np


def usage():
    print(
        "usage: python3 symbolReco.py [-s] [-o fname][-w weigthFile] inkmlfile lgFile ")
    print("     inkmlfile  : input inkml file name ")
    print("     lgFile     : input LG file name")
    print("     -o fname / --output fname : output file name (LG file)")
    print("     -w fname / --weight fname : weight file name (nn pytorch file)")
    print("     -s         : save hyp images")


"""
take an hypothesis (from LG = list of stroke index), select the corresponding strokes (from allTraces) and 
return the probability of being each symbol as a dictionnary {class_0 : score_0 ; ... class_i : score_i } 
Keep only the classes with a score higher than a threshold
"""


def computeClProb(alltraces, hyp, min_threshol, model, image_transforms,model2, saveIm=False):
    im = convert_to_imgs(get_traces_data(alltraces, hyp[1]), 32)
    if saveIm:
        imsave(hyp[0] + '.png', im)
    # create the list of possible classes (maybe connected to your classifier ???)
    classes = ['!', '(', ')', '+', 'COMMA', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=',
               'A', 'B', 'C', r'\Delta', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R',
               'S', 'T', 'V', 'X', 'Y', '[', ']', 'a', r'\alpha', 'b', r'\beta', 'c', r'\cos', 'd', r'\div',
               '/', '.', 'e', r'\exists', 'f', r'\forall', 'g', r'\gamma', r'\geq', r'\gt', 'h', 'i', r'\in',
                    r'\infty', r'\int', 'j', 'k', 'l', r'\lambda', r'\ldots', r'\leq', r'\lim', r'\log', r'\lt', 'm', r'\mu', 'n',
               r'\neq', 'o', 'p', r'\phi', '\pi', '|', r'\pm', r'\prime', 'q', 'r', r'\rightarrow', 's',
               r'\sigma', r'\sin', r'\sqrt', '\sum', 't', r'\tan', r'\theta', r'\times', 'u', 'v', 'w', 'x', 'y', 'z', r'\{', r'\}']

    ##### call your classifier and fill the results ! #####
    probs = classify(model, image_transforms, im, classes)
    probs2 = classify2(model2, image_transforms, im, classes)
    probs = np.mean( np.array([probs, probs2]), axis=0 )

    result = {}
    for i, x in enumerate(classes):
        prob = probs[i]
        if prob > min_threshol:
            result[x] = prob
    return result


def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], "so:w:", [
                                   "output=", "weight="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) != 2:
        print("Not enough parameters")
        usage()
        sys.exit(2)
    inputInkml = args[0]
    inputLG = args[1]
    saveimg = False
    outputLG = ""
    weightFile = "myweight.nn"

    for o, a in opts:
        if o in ("-s"):
            saveimg = True
        elif o in ("-o", "--output"):
            outputLG = a
        elif o in ("-w", "--weight"):
            weightFile = a
        else:
            usage()
            assert False, "unhandled option"

    traces = parse_inkml(inputInkml)
    hyplist = open(inputLG, 'r').readlines()
    hyplist = parseLG(hyplist)
    output = ""

    model_path = "./100_classes_model_aug.pt"
    model_path2 = "./trained_model_87_accuracy.pt"
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])

    model = get_model(model_path)
    model2 = get_model2(model_path2)

    for h in hyplist:
        # for each hypo, call the classifier and keep only slected classes (only the best or more)
        prob_dict = computeClProb(
            traces, h, 0.05, model, image_transforms, model2,saveimg)
        # rewrite the new LG
        for cl, prob in prob_dict.items():
            output += "O;" + h[0]+";"+cl+";" + \
                str(prob)+";"+";".join([str(s) for s in h[1]]) + "\n"
    if outputLG != "":
        with open(outputLG, "w") as text_file:
            print(output, file=text_file)
    else:
        print(output)


if __name__ == "__main__":
    # execute only if run as a script
    main()
