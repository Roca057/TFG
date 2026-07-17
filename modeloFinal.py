import sensor, image, time, gc, network, math, ml, json
from machine import Pin, I2C
from vl53l1x import VL53L1X
from ml.utils import NMS
from mqtt import MQTTClient

# --- 1. NETWORK AND INDUSTRIAL BROKER CONFIGURATION ---
SSID = "DATAROOM"
KEY = "spider8stunt=otter"
MQTT_BROKER = "172.21.1.200"
MQTT_CLIENT_ID = "NiclaVision"
MQTT_TOPIC_BASE = "MQTT/VISION/CALIDAD_TAPAS"

# --- 2. VISION AND REGION OF INTEREST (ROI) CONFIGURATION ---
TRIGGER_ROI = (600, 60, 15, 30)
DETECTION_ROI = (45, 0, 496, 260)

# --- Exposure registers ---
reg_high = 0x00
reg_low = 0x14

# --- RGB registers ---
reg_wb_red = 0x20
reg_wb_green = 0x40
reg_wb_blue = 0x49

# --- DISTANCE AND LIGHT THRESHOLDS ---
MIN_DISTANCE = 0
MAX_DISTANCE = 380
MIN_LUMINANCE = 40

# --- COLORS ---
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (255, 165, 0)
COLOR_WHITE = (255, 255, 255)

# --- Logo inspection parameters ---
ORANGE_THRESHOLD = [(0, 100, 0, 47, 10, 127)]
LOGO_ROI = (450, 83, 50, 90)
IDEAL_DENSITY = 0.70
DENSITY_MARGIN = 0.10
IDEAL_PROPORTION = 0.34
PROPORTION_MARGIN = 0.1

# --- DETERMINISTIC ANOMALY DETECTION PARAMETERS ---
WHITE_THRESHOLD = [(55, 78, -47, 2, -128, 19)]
BROWN_THRESHOLD = [(35, 76, -128, 5, 9, 50)]
WHITE_PART_ROI = (355, 30, 90, 200)
BROWN_CAP_ROI = (168, 64, 150, 120)
MIN_COVERAGE = 0.60

# --- 3. GLOBAL ML MODEL LOADING ---
print("Loading ML model from firmware...")
net_m = ml.Model("trained")

spot_labels = net_m.labels
MIN_CONFIDENCE = 0.6
threshold_list = [(math.ceil(MIN_CONFIDENCE * 255), 255)]


# --- ENUMS AND DATA STRUCTURES ---
class LogoStatus:
    VALID = 0
    MISSING = 1
    DENSITY_ERROR = 2
    RATIO_ERROR = 3


# --- GLOBAL HARDWARE OBJECTS ---
wlan = network.WLAN(network.STA_IF)
mosfet = Pin('PG1', Pin.OUT_PP)
mosfet.low()

I2C_PANEL_ADDR = 0x70
mqtt_client = None


# --- 4. HARDWARE CONFIGURATION FUNCTIONS ---
def config_camera_sensor():
    """
    Initializes and configures the OpenMV camera sensor hardware.
    """
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.set_windowing((640, 260))
    sensor.set_framebuffers(1)

    sensor.set_auto_exposure(False)
    sensor.set_auto_whitebal(False)
    sensor.set_auto_gain(False)

    # Manual registers are locked BEFORE skipping frames to freeze color state
    sensor.__write_reg(0x8b, 0x30)
    sensor.__write_reg(0x03, reg_high)
    sensor.__write_reg(0x04, reg_low)

    sensor.__write_reg(0xb1, reg_wb_red)
    sensor.__write_reg(0xb2, reg_wb_green)
    sensor.__write_reg(0xb3, reg_wb_blue)

    sensor.skip_frames(time=1000)


def config_tof():
    """
    Initializes the TOF sensor on I2C bus 2.
    """
    i2c_bus_tof = I2C(2)
    return VL53L1X(i2c_bus_tof)


def config_i2c_panel():
    """
    Initializes the LED Matrix panel on I2C bus 1.
    """
    i2c_bus = I2C(1, freq=100000)
    try:
        i2c_bus.writeto(I2C_PANEL_ADDR, bytes([0x21]))
        i2c_bus.writeto(I2C_PANEL_ADDR, bytes([0x81]))
        i2c_bus.writeto(I2C_PANEL_ADDR, bytes([0xE7]))

        empty_buffer = bytearray(17)
        i2c_bus.writeto(I2C_PANEL_ADDR, empty_buffer)
    except Exception as e:
        print("Error initializing I2C LED Matrix:", e)
    return i2c_bus


# --- 5. INDUSTRIAL NETWORK & MQTT FUNCTIONS ---
def connect_wifi():
    """
    Attempts to establish a robust Wi-Fi connection with the Dataroom router (Blocking - Use ONLY at boot).
    Returns True on success, False on failure.
    """
    for attempt in range(5):
        print("Wi-Fi connection attempt %d..." % (attempt + 1))
        wlan.active(False)
        time.sleep_ms(500)
        wlan.active(True)
        wlan.connect(SSID, KEY)

        for _ in range(10):
            time.sleep_ms(1000)
            if wlan.isconnected():
                return True
    return False


def connect_mqtt():
    """
    Establishes an asynchronous client connection to the central Dataroom broker.
    """
    try:
        print("Establishing connection with MQTT Broker...")
        client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER, port=1883)
        client.connect()
        return client
    except Exception as e:
        print("MQTT Connection Error:", e)
        return None


def send_data_mqtt(img, val_anomaly, val_spots, val_logo):
    """
    Publishes telemetry data splitting it into four independent topics.
    Attempts synchronous reconnection if the network or broker is lost.
    Returns True on success, False on failure.
    """
    global mqtt_client
    if not wlan.isconnected():
        print("Network lost. Attempting WiFi reconnection...")
        mqtt_client = None
        if not connect_wifi():
            print("WiFi reconnection failed. Skipping telemetry for this part.")
            return False

    if mqtt_client is None:
        print("MQTT client disconnected. Attempting to reconnect...")
        mqtt_client = connect_mqtt()
        if mqtt_client is None:
            print("MQTT reconnection failed. Skipping telemetry for this part.")
            return False

    try:
        current_time = time.ticks_ms()

        payload_anomalia = {"timestamp": current_time, "value": val_anomaly}
        payload_manchas = {"timestamp": current_time, "value": val_spots}
        payload_logo = {"timestamp": current_time, "value": val_logo}

        mqtt_client.publish(MQTT_TOPIC_BASE + "/ANOMALIA", json.dumps(payload_anomalia))
        mqtt_client.publish(MQTT_TOPIC_BASE + "/MANCHAS", json.dumps(payload_manchas))
        mqtt_client.publish(MQTT_TOPIC_BASE + "/LOGO", json.dumps(payload_logo))

        gc.collect()
        img_jpeg = img.to_jpeg(quality=75).bytearray()
        mqtt_client.publish(MQTT_TOPIC_BASE + "/IMAGEN", img_jpeg)

        print("Telemetry and Image frames published via MQTT.")
        return True

    except Exception as e:
        print("MQTT Transmission Error:", e)
        mqtt_client = None
        return False


def check_logo(img):
    """
    Scans the specified region of interest for the logo.
    """
    print("Scanning execution region for logo...")
    blobs_orange = img.find_blobs(ORANGE_THRESHOLD, roi=LOGO_ROI, pixels_threshold=50, area_threshold=50, merge=True, margin=10)
    if not blobs_orange:
        print("Inspection failure: Logo absent.")
        return LogoStatus.MISSING

    logo = max(blobs_orange, key=lambda b: b.pixels())

    size_ok = logo.w() > 15 and logo.h() > 10
    current_density = logo.density()
    density_ok = (IDEAL_DENSITY - DENSITY_MARGIN) < current_density < (IDEAL_DENSITY + DENSITY_MARGIN)

    current_ratio = logo.w() / logo.h()
    ratio_ok = (IDEAL_PROPORTION - PROPORTION_MARGIN) < current_ratio < (IDEAL_PROPORTION + PROPORTION_MARGIN)

    if size_ok and density_ok and ratio_ok:
        print("Verification successful: Density=%.2f | Ratio=%.2f" % (current_density, current_ratio))
        img.draw_rectangle(logo.rect(), color=COLOR_GREEN, thickness=2)
        img.draw_string(logo.x(), logo.y() - 15, "PASS", color=COLOR_GREEN)
        return LogoStatus.VALID


    print("Verification anomaly: Density=%.2f | Ratio=%.2f" % (current_density, current_ratio))
    img.draw_rectangle(logo.rect(), color=COLOR_ORANGE, thickness=2)

    if not density_ok:
        reason = "DENSITY_ERROR"
        status = LogoStatus.DENSITY_ERROR
    else:
        reason = "RATIO_ERROR"
        status = LogoStatus.RATIO_ERROR

    img.draw_string(logo.x(), logo.y() - 15, reason, color=COLOR_ORANGE)
    return status

def check_color_anomalies(input_img):
    """
    Validates deterministic color regions (White base and Brown cap).
    """
    is_anomalous = False

    # Check White Part
    total_white_area = WHITE_PART_ROI[2] * WHITE_PART_ROI[3]
    white_blobs = input_img.find_blobs(WHITE_THRESHOLD, roi=WHITE_PART_ROI)
    white_coverage = 0

    if white_blobs:
        largest_white = max(white_blobs, key=lambda b: b.pixels())
        white_coverage = largest_white.pixels() / total_white_area

    if white_coverage < MIN_COVERAGE:
        input_img.draw_rectangle(WHITE_PART_ROI, color=COLOR_RED, thickness=3)
        input_img.draw_string(WHITE_PART_ROI[0], WHITE_PART_ROI[1] - 15, "PART COLOR ERROR", color=COLOR_RED)
        is_anomalous = True
    else:
        input_img.draw_rectangle(WHITE_PART_ROI, color=COLOR_GREEN, thickness=1)

    # Check Brown Cap
    total_brown_area = BROWN_CAP_ROI[2] * BROWN_CAP_ROI[3]
    brown_blobs = input_img.find_blobs(BROWN_THRESHOLD, roi=BROWN_CAP_ROI)
    brown_coverage = 0

    if brown_blobs:
        largest_brown = max(brown_blobs, key=lambda b: b.pixels())
        brown_coverage = largest_brown.pixels() / total_brown_area

    if brown_coverage < MIN_COVERAGE:
        input_img.draw_rectangle(BROWN_CAP_ROI, color=COLOR_RED, thickness=3)
        input_img.draw_string(BROWN_CAP_ROI[0], BROWN_CAP_ROI[1] - 15, "CAP COLOR ERROR", color=COLOR_RED)
        is_anomalous = True
    else:
        input_img.draw_rectangle(BROWN_CAP_ROI, color=COLOR_GREEN, thickness=1)

    return is_anomalous


def detect_spots(input_img):
    """
    Executes the FOMO model to detect specified anomalies/spots with spatial filtering.
    """
    gc.collect()
    spots_detected = False

    def fomo_post_process(model, inputs, outputs):
        n, oh, ow, oc = model.output_shape[0]
        nms = NMS(ow, oh, inputs[0].roi)
        for i in range(oc):
            img_map = image.Image(outputs[0][0, :, :, i] * 255)
            blobs = img_map.find_blobs(threshold_list, x_stride=1, area_threshold=1, pixels_threshold=1)
            for b in blobs:
                rect = b.rect()
                x, y, w, h = rect
                score = (img_map.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0)
                nms.add_bounding_box(x, y, x + w, y + h, score, i)
        return nms.get_bounding_boxes()

    try:
        detections = net_m.predict([input_img], callback=fomo_post_process)
        rx, ry, rw, rh = DETECTION_ROI

        for i, detection_list in enumerate(detections):
            if i == 0 or not detection_list: continue

            for (x, y, w, h), score in detection_list:
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))

                if center_x < rx or center_x > (rx + rw) or center_y < ry or center_y > (ry + rh):
                    continue

                spots_detected = True
                input_img.draw_circle((center_x, center_y, 12), color=COLOR_RED, thickness=2)
                label_text = "{} {:.2f}".format(spot_labels[i], score)
                input_img.draw_string(x, y - 10, label_text, color=COLOR_RED)

    except Exception as e:
        print("Error executing FOMO model:", e)

    gc.collect()
    return spots_detected


# --- 7. HARDWARE CONTROL FUNCTIONS ---
def draw_shape(is_valid, i2c_bus):
    """
    Draws a green checkmark or a red X on a completely black background.
    """
    pattern_x = [0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81]
    pattern_tick = [0x00, 0x01, 0x02, 0x04, 0x88, 0x50, 0x20, 0x00]

    buffer = bytearray(17)
    buffer[0] = 0x00

    for row in range(8):
        green_index = 1 + (row * 2)
        red_index = 2 + (row * 2)

        if is_valid:
            buffer[green_index] = pattern_tick[row]
            buffer[red_index] = 0x00
        else:
            buffer[green_index] = 0x00
            buffer[red_index] = pattern_x[row]

    try: i2c_bus.writeto(I2C_PANEL_ADDR, buffer)
    except Exception as e:
        print("Failed to write shape to I2C panel:", e)


def draw_connection_error(i2c_bus):
    """
    Draws a Yellow exclamation mark (!) for network failure alerts.
    """
    pattern_err = [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x18]
    buffer = bytearray(17)
    buffer[0] = 0x00
    for row in range(8):
        green_index = 1 + (row * 2)
        red_index = 2 + (row * 2)
        buffer[green_index] = pattern_err[row]
        buffer[red_index] = pattern_err[row]
    try: i2c_bus.writeto(I2C_PANEL_ADDR, buffer)
    except Exception as e:
        print("Failed to write connection error to I2C panel:", e)


def clear_panel(i2c_bus):
    """Clears the LED matrix display."""
    buffer = bytearray(17)
    try: i2c_bus.writeto(I2C_PANEL_ADDR, buffer)
    except Exception as e:
        print("Failed to clear I2C panel:", e)


# --- 8. INITIALIZATION SEQUENCE ---
def setup_system():
    """
    Runs once at boot to initialize all hardware and network interfaces.
    """
    global mqtt_client

    print("Starting hardware initialization...")
    config_camera_sensor()
    tof_sensor = config_tof()
    led_panel = config_i2c_panel()
    clear_panel(led_panel)

    if connect_wifi():
        mqtt_client = connect_mqtt()
    else:
        print("Wi-Fi connection failed. Entering standalone operation mode.")
        mqtt_client = None

    return tof_sensor, led_panel


# --- 9. MAIN EXECUTION LOOP ---
def main_loop():
    """
    Core execution pipeline. Runs continuously after setup.
    """
    tof, i2c_led = setup_system()

    print("System polling active. Monitoring distance limits...")

    while True:
        distance = tof.read()

        if distance is None or distance <= 0:
            print("Error: invalid distance reading (%s)" % distance)
            continue

        if MIN_DISTANCE < distance < MAX_DISTANCE:
            mosfet.high()

            img = sensor.snapshot()

            img.draw_rectangle(DETECTION_ROI, color=COLOR_WHITE, thickness=1)
            img.draw_rectangle(LOGO_ROI, color=COLOR_BLUE)

            luminance = img.get_statistics(roi=TRIGGER_ROI).l_mean()

            if luminance > MIN_LUMINANCE:
                print("Target confirmation verified at %d mm | Luminance: %.1f" % (distance, luminance))

                # Execution Pipeline
                logo_status = check_logo(img)
                gc.collect()

                has_color_anomaly = check_color_anomalies(img)
                has_spots = detect_spots(img)

                is_part_correct = (not has_spots) and (not has_color_anomaly) and (logo_status == LogoStatus.VALID)
                draw_shape(is_part_correct, i2c_led)

                val_anomaly = 1 if has_color_anomaly else 0
                val_spots = 1 if has_spots else 0
                val_logo = 0 if logo_status == LogoStatus.VALID else 1

                # Submits structured telemetry via MQTT
                if not send_data_mqtt(img, val_anomaly, val_spots, val_logo):
                    draw_connection_error(i2c_led)

                time.sleep_ms(1200)
                clear_panel(i2c_led)

            mosfet.low()

        gc.collect()


# --- 10. SYSTEM START ---
main_loop()
