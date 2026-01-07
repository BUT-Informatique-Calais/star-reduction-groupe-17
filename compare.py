import cv2 as cv
import numpy as np

original = cv.imread('./results/original.png')
final = cv.imread('./results/final.png')

if original.shape != final.shape:
    final = cv.resize(final, (original.shape[1], original.shape[0]))

mode = 0  # 0=original, 1=final, 2=blink
blink = False

def draw_menu(img, mode):
    menu = img.copy()
    overlay = img.copy()

    cv.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    alpha = 0.6
    cv.addWeighted(overlay, alpha, menu, 1 - alpha, 0, menu)

    text = [
        "Display mode:",
        "1 - Original image",
        "2 - Final result",
        "3 - Blink animation",
        "ESC - Exit"
    ]

    for i, line in enumerate(text):
        cv.putText(menu, line, (10, 25 + i * 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if mode == 0:
        cv.putText(menu, "Current: Original", (10, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    elif mode == 1:
        cv.putText(menu, "Current: Final", (10, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        cv.putText(menu, "Current: Blink", (10, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return menu

cv.namedWindow("Star Reduction Viewer", cv.WINDOW_NORMAL)

while True:
    if mode == 0:
        display = original.copy()
    elif mode == 1:
        display = final.copy()
    else:
        blink = not blink
        display = final.copy() if blink else original.copy()

    display = draw_menu(display, mode)
    cv.imshow("Star Reduction Viewer", display)

    key = cv.waitKey(300) & 0xFF

    # Handle close button
    if cv.getWindowProperty("Star Reduction Viewer", cv.WND_PROP_VISIBLE) < 1:
        break

    if key == 27:
        break
    elif key == ord('1'):
        mode = 0
    elif key == ord('2'):
        mode = 1
    elif key == ord('3'):
        mode = 2

cv.destroyAllWindows()
