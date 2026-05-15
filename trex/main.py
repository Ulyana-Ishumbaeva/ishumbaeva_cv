import cv2
import pyautogui
import mss
import numpy as np

def wait_for_s(text):
    screen = np.zeros((180, 650, 3), dtype=np.uint8)
    cv2.putText(screen, text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    while True:
        cv2.imshow("Press S", screen)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

def select_area(name, scale=0.6):
    wait_for_s(f"Открой игру, нажми S для {name}")

    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[0])
        img = np.array(screenshot)[:, :, :3].copy()

        h, w = img.shape[:2]
        crop_w = int(w * 0.7)
        crop_h = int(h * 0.5)

        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 4

        cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
        small = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)

        roi = cv2.selectROI(name, small, False, False)
        cv2.destroyAllWindows()

        x, y, w_roi, h_roi = roi

        return {
            "top": int(start_y + y / scale),
            "left": int(start_x + x / scale),
            "width": int(w_roi / scale),
            "height": int(h_roi / scale)
        }

def count_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    return cv2.countNonZero(thresh)

def main():
    print("1. Выбери всю область игры")
    game_area = select_area("Game Area")

    print("2. Выбери БЛИЖНЮЮ зону (красная)")
    near_area = select_area("Near Area")

    cooldown = 0
    jump_type = 1
    score = 0
    red_growth = 0
    blue_growth = 0

    with mss.mss() as sct:
        while True:
            frame = np.array(sct.grab(game_area))[:, :, :3].copy()

            nrx = near_area["left"] - game_area["left"]
            nry = near_area["top"] - game_area["top"] - 3

            # красная зона
            red_width = near_area["width"] + red_growth
            red_width = min(red_width, 230)

            # синяя зона автоматически справа
            blue_x = nrx + red_width + 5

            blue_width = int(red_width * 1.7)

# ограничение
            blue_width = min(blue_width, 170)

            if blue_x + blue_width > frame.shape[1]:
                blue_width = frame.shape[1] - blue_x - 1

            # красная
            top_strip = frame[nry:nry+near_area["height"]//2, nrx:nrx+red_width]
            bottom_strip = frame[nry+near_area["height"]//2:nry+near_area["height"], nrx:nrx+red_width]
            near_pixels = count_dark(top_strip) + count_dark(bottom_strip)

            # синяя
            far_img = frame[nry:nry+near_area["height"], blue_x:blue_x+blue_width]
            far_pixels = count_dark(far_img)

            far_total = far_img.shape[0] * far_img.shape[1]
            far_ratio = far_pixels / far_total

            if far_ratio > 0.050:
                jump_type = 3
            elif far_ratio > 0.018:
                jump_type = 2
            else:
                jump_type = 1

            if cooldown > 0:
                cooldown -= 1

            if near_pixels > 20 and cooldown == 0:
                pyautogui.keyDown('space')

                if jump_type == 1:
                    cv2.waitKey(35)
                elif jump_type == 2:
                    cv2.waitKey(65)
                else:
                    cv2.waitKey(110)

                pyautogui.keyUp('space')
                cooldown = 9

            # game over
            game_over_zone = frame[0:80, 250:520]
            over_pixels = count_dark(game_over_zone)

            if over_pixels > 500:
                cv2.waitKey(500)
                pyautogui.press('space')
                red_growth = 0
                blue_growth = 0
                score = 0

            if over_pixels < 500:
                score += 1

            # рост зон
            if score % 120 == 0:
                red_growth += 2
                blue_growth += 6

                red_growth = min(red_growth, 120)
                blue_growth = min(blue_growth, 140)

            # рисуем
            cv2.rectangle(frame,
                          (nrx, nry),
                          (nrx + red_width, nry + near_area["height"]),
                          (0, 0, 255), 2)

            cv2.rectangle(frame,
                          (blue_x, nry),
                          (blue_x + blue_width, nry + near_area["height"]),
                          (255, 0, 0), 2)

            cv2.putText(frame,
                        f"near={near_pixels} far={far_pixels} jump={jump_type}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            preview = cv2.resize(frame, (900, 320))
            cv2.imshow("T-Rex Bot", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()