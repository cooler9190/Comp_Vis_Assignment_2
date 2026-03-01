import cv2 as cv

test_frame = cv.imread('data/cam4/test_frame_cam4.png')
ground_truth = cv.imread('data/cam4/ground_truth_cam4.png', cv.IMREAD_GRAYSCALE)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

best_error = float('inf')
best_params = {}

kerne_sizes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

print("MOG2 optimization in progress. Please wait...")

# 1. Only loop var_thresh here
for var_thresh in range(2, 101, 2): 
    
    fgbg = cv.createBackgroundSubtractorMOG2(varThreshold=var_thresh, detectShadows=True)
    cap_bg = cv.VideoCapture('data/cam4/background.avi')

    # Train the background model
    while True:
        ret, frame = cap_bg.read()
        if not ret:
            break
        fgbg.apply(frame)
        
    cap_bg.release() # Moved OUTSIDE the loop!

    # Get the raw mask for this var_thresh
    raw_mask = fgbg.apply(test_frame, learningRate=0)
    _, binary_mask = cv.threshold(raw_mask, 254, 255, cv.THRESH_BINARY)

    for kernel_size in kerne_sizes:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)

        # 2. NOW loop through your morphological operations
        for open_iters in range(0, 10):
            for close_iters in range(0, 10):
                
                # Note: if iterations are 0, we just pass the previous image forward
                erosion = cv.erode(binary_mask, kernel, iterations=open_iters) if open_iters > 0 else binary_mask
                opening = cv.dilate(erosion, kernel, iterations=open_iters) if open_iters > 0 else erosion

                dilation = cv.dilate(opening, kernel, iterations=close_iters) if close_iters > 0 else opening
                final_mask = cv.erode(dilation, kernel, iterations=close_iters) if close_iters > 0 else dilation

                error_img = cv.bitwise_xor(final_mask, ground_truth)
                error_score = cv.countNonZero(error_img)

                print(f"varThreshold: {var_thresh}, open_iters: {open_iters}, close_iters: {close_iters} | Error: {error_score}")

                if error_score < best_error:
                    best_error = error_score
                    best_params = {
                        'varThreshold': var_thresh,
                        'kernel_size': kernel_size,
                        'open_iters': open_iters,
                        'close_iters': close_iters
                    }

print("\nOptimization complete!")
print(f"Best Parameters: {best_params}")
print(f"Best Error Score: {best_error}")