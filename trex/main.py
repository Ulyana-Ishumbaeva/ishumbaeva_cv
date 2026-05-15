import cv2
import pyautogui
import mss
import numpy as np

def wait_for_s(text):
    screen = np.zeros((180, 700, 3), dtype=np.uint8)
    cv2.putText(screen, text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    while True:
        cv2.imshow("Setup", screen)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

def select_area(name):
    wait_for_s(f"Открой игру, нажми S для {name}")

    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[0])
        img = np.array(screenshot)[:, :, :3].copy()

        roi = cv2.selectROI(name, img, False, False)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        return {"left": x, "top": y, "width": w, "height": h}

def main():
    print("1. Выбери всю область игры")
    game_area = select_area("Game Area")

    print("2. Выбери динозавра")
    dino_area = select_area("Dino")

    print("3. Выбери зону перед динозавром")
    detect_area = select_area("Detect Area")

    cooldown = 0
    frames = 0
    second_jump_needed = False

    with mss.mss() as sct:
        first_frame = np.array(sct.grab(game_area))[:, :, :3].copy()

        dx = dino_area["left"] - game_area["left"]
        dy = dino_area["top"] - game_area["top"]
        dw = dino_area["width"]
        dh = dino_area["height"]

        template = cv2.cvtColor(first_frame[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY)
        ground_y = dy

        while True:
            frames += 1
            frame = np.array(sct.grab(game_area))[:, :, :3].copy()

            gx = detect_area["left"] - game_area["left"]
            gy = detect_area["top"] - game_area["top"]
            base_w = detect_area["width"]
            gh = detect_area["height"]

            # НОРМАЛЬНЫЙ рост зоны
            dynamic_w = base_w + frames // 50
            dynamic_w = min(dynamic_w, base_w + 180)

            zone = frame[gy:gy+gh, gx:gx+dynamic_w]

            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
            thresh[-8:, :] = 0

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            obstacles = []

            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh > 12 and bw > 2:
                    obstacles.append((bx, bw, bh))

            obstacles.sort(key=lambda obj: obj[0])

            game_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            dino_y = max_loc[1]
            dino_on_ground = dino_y >= ground_y - 5

            if cooldown > 0:
                cooldown -= 1

            if dino_on_ground and second_jump_needed and cooldown == 0:
                pyautogui.press('space')
                cooldown = 5
                second_jump_needed = False

            if obstacles and cooldown == 0:
                first_x, first_w, first_h = obstacles[0]
                center_x = first_x + first_w // 2

                trigger = 45

                if first_w > 16 or first_h > 25:
                    hold = 3
                else:
                    hold = 1

                if center_x < trigger:
                    pyautogui.keyDown('space')

                    for _ in range(hold):
                        cv2.waitKey(12)

                    pyautogui.keyUp('space')
                    cooldown = 4

            if not dino_on_ground and len(obstacles) > 1:
                first_x, first_w, _ = obstacles[0]
                second_x = obstacles[1][0]

                gap = second_x - (first_x + first_w)

                if 0 < gap < 35:
                    second_jump_needed = True

            center = frame[0:100, frame.shape[1]//3:frame.shape[1]//3*2]
            gray2 = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
            _, over = cv2.threshold(gray2, 170, 255, cv2.THRESH_BINARY_INV)

            if cv2.countNonZero(over) > 500:
                pyautogui.press('space')
                second_jump_needed = False
                frames = 0

            cv2.rectangle(frame, (gx, gy), (gx+dynamic_w, gy+gh), (0, 0, 255), 2)

            for obs in obstacles:
                ox, ow, oh = obs
                cv2.rectangle(frame,
                              (gx+ox, gy+gh-oh),
                              (gx+ox+ow, gy+gh),
                              (255, 0, 0), 2)

            cv2.putText(frame, f"width={dynamic_w}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("T-Rex Bot", cv2.resize(frame, (1000, 300)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()