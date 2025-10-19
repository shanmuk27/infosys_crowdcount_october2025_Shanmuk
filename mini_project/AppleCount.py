from flask import Flask, render_template, redirect, url_for, request, send_file
import cv2
from io import BytesIO
import numpy as np

def dectect_apples(input_image_np):
    image = input_image_np.copy()

    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((40, 40), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    apple_count = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 30000:
            apple_count += 1
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(image, center, radius, (0, 255, 0), 3)

            cv2.putText(image, str(apple_count), (int(x), int(y) - radius - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    height, width = image.shape[:2]
    max_display_width = 800

    if width > max_display_width:
        scale_factor = max_display_width / width
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (max_display_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image, apple_count

app = Flask(__name__)

@app.route('/')
def root():
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "File not found", 400
    file = request.files['image']
    if file.filename == "":
        return "No file selected", 400
    if file:
        try:
            image_data = file.read()

            np_array = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if cv_image is None:
                return "Could not decode image file", 400

            processed_img_np, count = dectect_apples(cv_image)

            _, encoded_img = cv2.imencode('.png', processed_img_np)
            output_stream = BytesIO(encoded_img.tobytes())
            
            output_mimetype = 'image/png'

            response = send_file(
                output_stream,
                mimetype=output_mimetype,
                as_attachment=False
            )
            response.headers['X-Apple-Count'] = str(count) 
            return response

        except Exception as e:
            print(f"Internal Server Error: {e}") 
            return f'Internal Server Error during processing: {e}', 500

    return 'An unexpected error occurred', 500

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)