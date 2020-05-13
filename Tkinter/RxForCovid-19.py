# Programador Sergio Luis Beleño Díaz

import cv2
import numpy as np
from tkinter import *
from tensorflow.keras.models import load_model
from easygui import *
from lime import lime_image
from PIL import ImageTk, Image
from skimage.segmentation import mark_boundaries
from time import sleep

#Load the best model trained
model = load_model('Model')

root = Tk()
root.title('Rx-Ray')


def show_image():
    try:
        global name_file
    except:
        print()
    name_file = fileopenbox()
    try:
        global inp_img
    except:
        print()
    inp_img = cv2.imread(name_file)
    inp_img = cv2.resize(inp_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    try:
        global input_image
    except:
        print()
    input_image = inp_img
    inp_img = Image.fromarray(inp_img)
    inp_img = ImageTk.PhotoImage(inp_img)
    my_label = Label(image=inp_img)
    my_label.grid(row=0, column=0, columnspan=1)
    my_label2 = Label(image=inp_img)
    my_label2.grid(row=0, column=2, columnspan=1)


def save_image():
    try:
        global contad
        contad = 1
    except:
        contad = contad + 1
    try:
        dir_save = diropenbox()
        name_of_save = (str(dir_save) + "\predicted_" + str(contad) + ".png")
        cv2.imwrite(name_of_save, endo)

    except:
        print("Don't use dots in the name of the file")


def detection():
    try:
        global end2
    except:
        print()
    num_c = 8
    cam = input_image
    # ImagenInput
    inputoimage = cam
    x = inputoimage.reshape((-1, 512, 512, 3))

    predictor = model.predict(x)

    predIdxs = np.argmax(predictor, axis=1)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x[-1], model.predict,
                                             hide_color=0,
                                             num_features=100,
                                             num_samples=1000)

    print("Predicted:", predIdxs[-1])

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=num_c,
                                                hide_rest=True)

    mask = np.array(mark_boundaries(temp / 2 + 1, mask))

    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.blur(mask, (30, 30))
    mask = cv2.blur(mask, (15, 15))
    mask = cv2.blur(mask, (10, 10))
    mask = cv2.blur(mask, (5, 5))
    mask = np.array(mask, dtype=np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if predIdxs[-1] == 0:
        message = 'Healthy'
      #mask = mask*0
    if predIdxs[-1] == 1:
        message = 'Pneumonia &'
        messagetwo = 'Covid-19'
    if predIdxs[-1] == 2:
        message = 'Cardiomegaly'
    if predIdxs[-1] == 3:
        message = 'Other Diseases'
    if predIdxs[-1] == 4:
        message = 'Pleural Effusion'

    mask2 = cv2.applyColorMap((mask), cv2.COLORMAP_JET)
    #heatmap

    mask = cv2.blur(mask, (60, 60))
    mask = cv2.blur(mask, (30, 30))
    mask = cv2.blur(mask, (15, 15))
    mask = cv2.blur(mask, (10, 10))
    mask = cv2.blur(mask, (5, 5))

    mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    #heatmap

    mask = ((mask * 1.1 + mask2 * 0.7) / 255) * (3 / 2)
    end = cv2.addWeighted(x[-1] / 255, 0.8, mask2 / 255, 0.3, 0)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (20, 50)

    # fontScale
    fontScale = 1.5

    # Blue color in BGR
    color = (203, 194, 126)

    # Line thickness of 3 px
    thickness = 3

    # Using cv2.putText() method
    end2 = cv2.putText((end * 250), str(message), org, font,
                   fontScale, (44, 4, 4), 10, cv2.LINE_AA)

    # Using cv2.putText() method
    end2 = cv2.putText((end2), str(message), org, font,
                   fontScale, color, thickness, cv2.LINE_AA)

    if message == 'Pneumonia &':
        # Using cv2.putText() method
        end2 = cv2.putText((end2), str(messagetwo), (20, 100), font,
        fontScale, (44, 4, 4), 10, cv2.LINE_AA)

        # Using cv2.putText() method
        end2 = cv2.putText((end2), str(messagetwo), (20, 100), font,
        fontScale, color, thickness, cv2.LINE_AA)

    preone = int(np.round((predictor[0][0]) * 100))
    pretwo = int(np.round(predictor[0][1] * 100))
    prethr = int(np.round(predictor[0][2] * 100))
    prefou = int(np.round(predictor[0][3] * 100))
    prefiv = int(np.round(predictor[0][4] * 100))

    try:
        print(str(type(end2)))
        print(str((end2.shape)))

    except:
        print("Nothing")

    cv2.imwrite("n.png", end2)
    sleep(10)


    try:
        global endo
    except:
        print()
    try:
        end2 = cv2.imread("n.png")
        endo = end2
        end2 = cv2.cvtColor(end2, cv2.COLOR_BGR2RGB)

    except:
        sleep(10)
        end2 = cv2.imread("n.png")
        endo = end2
        end2 = cv2.cvtColor(end2, cv2.COLOR_BGR2RGB)

    end2 = Image.fromarray(end2)
    end2 = ImageTk.PhotoImage(end2)
    my_label_end2 = Label(image=end2)
    my_label_end2.grid(row=0, column=2, columnspan=1)

    my_label_dis = Label(text=('Healthy: ' + str(preone) + '% \n' +
    'Pneumonia & Covid-19: ' + str(pretwo) + '% \n' +
    'Cardiomegaly: ' + str(prethr) + '% \n' +
    'Other Diseases: ' + str(prefou) + '% \n' +
    'Pleural Effusion: ' + str(prefiv) + '% \n'))
    my_label_dis.grid(row=0, column=1)

button_open = Button(root, text="Open", command=show_image)
button_save = Button(root, text="Save", command=save_image)
button_Detect = Button(root, text="Detect", command=detection)

button_open.grid(row=1, column=0)
button_save.grid(row=1, column=1)
button_Detect.grid(row=1, column=2)

root.mainloop()