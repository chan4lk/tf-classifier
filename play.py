import cv2 as cv
import math

def find_lghts(light, black, org, cr_h, cr_w):
    contours_light, _ = cv.findContours(
        light, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_black, _ = cv.findContours(
        black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    thresh_height = cr_h/2
    thresh_width = cr_w/4
    trafic_lights = []
    rects = []
    for _, c in enumerate(contours_black):
        area_black = cv.contourArea(c)
        if area_black > 0:
            # print(area_black)
            x, y, w, h = cv.boundingRect(c)
            if w < thresh_width and h < thresh_height:
                for _, cl in enumerate(contours_light):
                    area_light = cv.contourArea(cl)
                    if area_light > 0 and area_black > area_light:
                        xl, yl, wl, hl = cv.boundingRect(cl)
                        if (x < xl and x+w > xl + wl
                                and y < yl and y+h > yl+hl):
                            trafic_lights.append(c)
                            rects.append(org[y:y+h, x:x+w])
                            break
    output = org.copy()
    for c in trafic_lights:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return (output, rects)


def draw_frame(img):
    h, w, _ = img.shape
    cropped_img = img.copy()
    cropped_img = img[math.floor(h/4):math.floor(h*3/4)]
    cropped_img = cropped_img[:, math.floor(w/8):math.floor(w/2)]
    cr_h, cr_w, _ = cropped_img.shape

    gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    _, gb = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    _, black_shape = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)
    output, rects = find_lghts(gb, black_shape, cropped_img, cr_h, cr_w)

    return output


source = './videos/video-2.mp4'
vidcap = cv.VideoCapture(source)


while True:
    success, image = vidcap.read()

    if not success:
        vidcap = cv.VideoCapture(source)
        continue

    cv.imshow('video', draw_frame(image))

    key = cv.waitKey(25)
    if key == 27:
        break

cv.destroyAllWindows()
