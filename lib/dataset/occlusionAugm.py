import cv2
import numpy as np
import random

def occlusion_augmentation(img, landmarks, visibility, occlusion_prob=0.3, max_occ_size=10):
    img_h, img_w, _ = img.shape
    new_visibility = visibility.copy()

    for i in range(len(landmarks)):
        if np.random.rand() < occlusion_prob:
            x, y = int(landmarks[i][0]), int(landmarks[i][1])
            if x < 0 or y < 0 or x >= img_w or y >= img_h:
                continue

            occ_size = np.random.randint(10, max_occ_size)
            x1 = max(0, x - occ_size // 2)
            y1 = max(0, y - occ_size // 2)
            x2 = min(img_w, x + occ_size // 2)
            y2 = min(img_h, y + occ_size // 2)

            mode = random.choice(["noise", "dark", "blur"])

            if mode == "noise":
                img[y1:y2, x1:x2] = np.random.randint(0, 255, size=(y2 - y1, x2 - x1, 3), dtype=np.uint8)
            elif mode == "dark":
                img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * 0.2).astype(np.uint8)
            elif mode == "blur":
                region = img[y1:y2, x1:x2]
                if region.size > 0:
                    blurred = cv2.GaussianBlur(region, (5, 5), 0)
                    img[y1:y2, x1:x2] = blurred

            new_visibility[i] = 0.0

    debug = False

    if debug:
        # Mark landmarks on image: green = visible, red = occluded
        vis_img = img.copy()
        for i in range(len(landmarks)):
            x = int(landmarks[i][0])
            y = int(landmarks[i][1])
            color = (0, 255, 0) if new_visibility[i] == 1.0 else (0, 0, 255)
            cv2.circle(vis_img, (x, y), 3, color, -1)
        cv2.imshow("Occlusion Augmentation", vis_img)
        cv2.waitKey(0)  # Press any key to continue
        cv2.destroyAllWindows()

    return img, new_visibility
