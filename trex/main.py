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
    wait_for_s(f"Open game and press S for {name}")

    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[0])
        img = np.array(screenshot)[:, :, :3].copy()

        roi = cv2.selectROI(name, img, False, False)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        return {"left": x, "top": y, "width": w, "height": h}

def main():
    game_area = select_area("Game Area")
    dino_area = select_area("Dino")
    detect_area = select_area("Cactus Area")
    bird_area = select_area("Bird Area")

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
            gw = detect_area["width"]
            gh = detect_area["height"]

            bx = bird_area["left"] - game_area["left"]
            by = bird_area["top"] - game_area["top"]
            bw = bird_area["width"]
            bh = bird_area["height"]

            dynamic_w = gw + frames // 50
            dynamic_w = min(dynamic_w, gw + 180)

            cactus_zone = frame[gy:gy+gh, gx:gx+dynamic_w]
            bird_zone = frame[by:by+bh, bx:bx+bw]

            cactus_gray = cv2.cvtColor(cactus_zone, cv2.COLOR_BGR2GRAY)
            _, cactus_thresh = cv2.threshold(cactus_gray, 170, 255, cv2.THRESH_BINARY_INV)
            cactus_thresh[-8:, :] = 0

            bird_gray = cv2.cvtColor(bird_zone, cv2.COLOR_BGR2GRAY)
            _, bird_thresh = cv2.threshold(bird_gray, 170, 255, cv2.THRESH_BINARY_INV)

            cactus_contours, _ = cv2.findContours(
                cactus_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            bird_contours, _ = cv2.findContours(
                bird_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            obstacles = []
            birds = False

            for c in cactus_contours:
                x0, y0, w0, h0 = cv2.boundingRect(c)
                if h0 > 12 and w0 > 2:
                    obstacles.append((x0, w0, h0))

            for c in bird_contours:
                x0, y0, w0, h0 = cv2.boundingRect(c)
                if w0 > 6 and h0 > 4:
                    birds = True

            obstacles.sort(key=lambda obj: obj[0])

            game_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            dino_y = max_loc[1]
            dino_on_ground = dino_y >= ground_y - 5

            if cooldown > 0:
                cooldown -= 1

            if dino_on_ground and second_jump_needed and cooldown == 0:
                pyautogui.press("space")
                cooldown = 5
                second_jump_needed = False

            if obstacles and cooldown == 0:
                first_x, first_w, first_h = obstacles[0]
                center_x = first_x + first_w // 2

                trigger = 45

                hold = 3 if (first_w > 16 or first_h > 25) else 1

                if center_x < trigger:
                    pyautogui.keyDown("space")
                    for _ in range(hold):
                        cv2.waitKey(12)
                    pyautogui.keyUp("space")
                    cooldown = 4

            if birds and dino_on_ground and cooldown == 0:
                pyautogui.press("space")
                cooldown = 4

            if not dino_on_ground and len(obstacles) > 1:
                first_x, first_w, _ = obstacles[0]
                second_x = obstacles[1][0]

                gap = second_x - (first_x + first_w)

                if 0 < gap < 35:
                    second_jump_needed = True

            cv2.rectangle(frame, (gx, gy), (gx + dynamic_w, gy + gh), (0, 0, 255), 2)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)

            for obs in obstacles:
                ox, ow, oh = obs
                cv2.rectangle(frame,
                              (gx + ox, gy + gh - oh),
                              (gx + ox + ow, gy + gh),
                              (255, 0, 0), 2)

            cv2.imshow("T-Rex Bot", cv2.resize(frame, (1000, 300)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()