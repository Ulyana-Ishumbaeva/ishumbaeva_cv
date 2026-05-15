import cv2
import pyautogui
import mss
import numpy as np

def wait_for_s(text):
    screen = np.zeros((180, 600, 3), dtype=np.uint8)
    cv2.putText(screen, text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    while True:
        cv2.imshow("Press S", screen)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

def select_area(name, scale=0.8):
    wait_for_s(f"Открой игру, нажми S для {name}")

    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[0])
        img = np.array(screenshot)[:, :, :3].copy()

        # берём не весь экран, а центральный кусок
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

def main():
    print("1. Выбери ВСЮ область игры (динозавр + путь)")
    game_area = select_area("Game Area", scale=0.5)

    print("2. Выбери область препятствий")
    obstacle_area = select_area("Obstacle Area", scale=0.5)

    print("Бот готов. Кликни в игру — прыжки будут работать даже без фокуса окна.")
    
    min_pixels = 40
    cooldown = 0
    speed_factor = 1.0  # коэффициент для увеличения зоны препятствий

    with mss.mss() as sct:
        while True:
            # живой кадр всей игры
            game_img = np.array(sct.grab(game_area))[:, :, :3].copy()

            # вычисляем obstacle из game_img
            rx = obstacle_area["left"] - game_area["left"]
            ry = obstacle_area["top"] - game_area["top"]

            # увеличиваем ширину зоны вперед пропорционально скорости
            obs_width = int(obstacle_area["width"] * speed_factor)
            obs_width = min(obs_width, game_area["width"] - rx)

            obstacle_img = game_img[ry:ry + obstacle_area["height"], rx:rx + obs_width]

            gray = cv2.cvtColor(obstacle_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
            dark = cv2.countNonZero(thresh)

            if cooldown > 0:
                cooldown -= 1

            if dark > min_pixels and cooldown == 0:
                # клик внутрь игры + прыжок
                click_x = game_area['left'] + game_area['width']//4
                click_y = game_area['top'] + game_area['height']//2
                pyautogui.click(click_x, click_y)
                pyautogui.press('space')
                cooldown = 8

            # рамка obstacles на игровом кадре
            cv2.rectangle(
                game_img,
                (rx, ry),
                (rx + obs_width, ry + obstacle_area["height"]),
                (0, 0, 255),
                2
            )

            # текст
            cv2.putText(
                game_img,
                f"Pixels: {dark}  Speed: {speed_factor:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # живое окно с масштабированием
            preview = cv2.resize(game_img, (900, 300))
            cv2.imshow("T-Rex Bot", preview)

            # увеличение зоны вперед для повышения скорости реакции
            speed_factor += 0.002

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()