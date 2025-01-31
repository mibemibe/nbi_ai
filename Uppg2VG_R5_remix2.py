import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageOps
import cv2
import pandas as pd
import time  # Lägg till import av time-modulen

# Global camera variable
camera = None

def initialize_camera():
    """Initialize camera once and reuse"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Kameran kunde inte startas.")
            return None
        # Set smaller resolution for faster capture
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Set a timeout to avoid long waits
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def load_mnist_data():
    """Ladda och normalisera MNIST-data."""
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype(int)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Dela upp data i tränings- och testset."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train, max_iter=1000):
    """Träna en logistisk regressionsmodell med justerade parametrar."""
    lr_model = SGDClassifier(max_iter=max_iter, tol=1e-3)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_decision_tree(X_train, y_train, max_depth=10):
    """Träna en beslutsmodell."""
    dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=5)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_svm(X_train, y_train, kernel='linear'):
    """Träna en Support Vector Machine-modell med angiven kernel."""
    svm_model = SVC(kernel=kernel, probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(model, X_test, y_test):
    """Utvärdera en modell och returnera dess noggrannhet."""
    return model.score(X_test, y_test)

def save_model(model, model_name):
    """Spara modellen i en fil."""
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model(model_name):
    """Ladda modellen från en fil."""
    with open(f'{model_name}_model.pkl', 'rb') as f:
        return pickle.load(f)

def save_results(model_name, accuracy):
    """Spara resultaten i en fil."""
    with open(f'{model_name}_results.pkl', 'wb') as f:
        pickle.dump(accuracy, f)

def load_results(model_name):
    """Ladda resultaten från en fil."""
    with open(f'{model_name}_results.pkl', 'rb') as f:
        return pickle.load(f)

def check_file_exists(filepath):
    """Kontrollera om filen finns."""
    return os.path.exists(filepath)

def retrain_model(model_name, max_iter=None, max_depth=None, kernel=None):
    X, y = load_mnist_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    if model_name == 'logistic_regression':
        model = train_logistic_regression(X_train, y_train, max_iter)
        accuracy = evaluate_model(model, X_test, y_test)
        save_results('logistic_regression', accuracy)
        save_model(model, 'logistic_regression')
    elif model_name == 'decision_tree':
        model = train_decision_tree(X_train, y_train, max_depth)
        accuracy = evaluate_model(model, X_test, y_test)
        save_results('decision_tree', accuracy)
        save_model(model, 'decision_tree')
    elif model_name == 'svm':
        model = train_svm(X_train, y_train, kernel)
        accuracy = evaluate_model(model, X_test, y_test)
        save_results('svm', accuracy)
        save_model(model, 'svm')
    return accuracy

# Visa noggrannhet och knappar för att räkna om modellerna
st.title("---- Modeller ----")
st.write("Klassificering påbörjas...")
cols = st.columns(3)

# Initialize session state for button states
if 'retrain_button' not in st.session_state:
    st.session_state.retrain_button = False

def disable_buttons():
    st.session_state.retrain_button = True

def enable_buttons():
    st.session_state.retrain_button = False

with cols[0]:
    st.write("Logistisk regression")
    max_iter = st.slider('Antal iterationer', min_value=100, max_value=3000, value=1000, step=100, key='lr_max_iter_slider')
    if check_file_exists('logistic_regression_results.pkl'):
        lr_accuracy = load_results('logistic_regression')
        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Noggrannhet:</span><span>{lr_accuracy:.4f}</span></div>", unsafe_allow_html=True)
        st.write("(laddad från fil)")
    else:
        st.write("Noggrannhet: Ej beräknad")
    if st.session_state.retrain_button:
        st.button('BERÄKNAR', key='retrain_lr_button_disabled', disabled=True)
    else:
        if st.button('Räknar om', key='retrain_lr_button'):
            disable_buttons()
            with st.spinner('Beräknar...'):
                lr_accuracy = retrain_model('logistic_regression', max_iter=max_iter)
                enable_buttons()
                st.experimental_set_query_params()

with cols[1]:
    st.write("Beslutsträd")
    max_depth = st.slider('Maxdjup', min_value=1, max_value=50, value=10, step=1, key='dt_max_depth_slider')
    if check_file_exists('decision_tree_results.pkl'):
        dt_accuracy = load_results('decision_tree')
        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Noggrannhet:</span><span>{dt_accuracy:.4f}</span></div>", unsafe_allow_html=True)
        st.write("(laddad från fil)")
    else:
        st.write("Noggrannhet: Ej beräknad")
    if st.session_state.retrain_button:
        st.button('BERÄKNAR', key='retrain_dt_button_disabled', disabled=True)
    else:
        if st.button('Räknar om', key='retrain_dt_button'):
            disable_buttons()
            with st.spinner('Beräknar...'):
                dt_accuracy = retrain_model('decision_tree', max_depth=max_depth)
                enable_buttons()
                st.experimental_set_query_params()

with cols[2]:
    st.write("Support Vector Machine")
    kernel = st.selectbox('Välj kernel', ['linear', 'poly', 'rbf'], key='svm_kernel_selectbox')
    if check_file_exists('svm_results.pkl'):
        svm_accuracy = load_results('svm')
        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Noggrannhet:</span><span>{svm_accuracy:.4f}</span></div>", unsafe_allow_html=True)
        st.write("(laddad från fil)")
    else:
        st.write("Noggrannhet: Ej beräknad")
    if st.session_state.retrain_button:
        st.button('BERÄKNAR', key='retrain_svm_button_disabled', disabled=True)
    else:
        if st.button('Räknar om', key='retrain_svm_button'):
            disable_buttons()
            with st.spinner('Beräknar...'):
                svm_accuracy = retrain_model('svm', kernel=kernel)
                enable_buttons()
                st.experimental_set_query_params()

# Bildbehandlingsfunktioner
def preprocess_image(roi):
    """Förbehandla bilden för att passa modellen."""
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.equalizeHist(roi)
    alpha = 2.0
    beta = 50
    roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
    roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    roi = cv2.filter2D(roi, -1, kernel)
    roi = cv2.bitwise_not(roi)
    img_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    return img_resized

def preprocess_camera_image(roi, contrast, brightness, edge_detection, threshold_value):
    """Förbehandla kamerabilden för att passa modellen."""
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.equalizeHist(roi)
    alpha = 1 + contrast / 100.0
    beta = brightness
    roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
    _, roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    kernel = np.array([[0, -1, 0], [-1, 4 + edge_detection / 25.0, -1], [0, -1, 0]])
    roi = cv2.filter2D(roi, -1, kernel)
    roi = cv2.bitwise_not(roi)
    img_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    return img_resized

def predict_image(image):
    """Gör en prediktion på en bild och returnera sannolikheter."""
    if model is None:
        st.error("Ingen modell är laddad.")
        return [], []
    image_np = np.array(image)
    digit_img = image_np.reshape(1, -1)
    prediction = model.predict(digit_img)
    probabilities = model.predict_proba(digit_img)[0]
    return prediction[0], probabilities

def process_image(image):
    """Förbehandla bilden för att passa modellen och gör en prediktion."""
    image = preprocess_image(np.array(image))
    st.write("Gör en prediktion...")
    label, _ = predict_image(image)
    st.write(f"Predikterade siffra: {label}" if label else "Inga siffror hittades.")

def upload_image():
    """Ladda upp en bild och gör en prediktion."""
    uploaded_file = st.file_uploader("Välj en bild...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        process_image(image)

def draw_image():
    """Rita en bild och gör en prediktion."""
    st.write("Rita en bild i 28x28 pixel canvas.")
    
    # Lägg till alternativ för penselstorlek
    stroke_width = st.slider("Välj penselstorlek:", 1, 25, 20)
    
    cols = st.columns(2)
    with cols[0]:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color='#000000',  # Svart färg
            background_color='#FFFFFF',  # Vit bakgrund
            height=280,
            width=280,
            drawing_mode='freedraw',
            key='canvas'
        )

    if canvas_result.image_data is not None:
        with cols[1]:
            img_array = canvas_result.image_data
            image = Image.fromarray(img_array.astype('uint8'), 'RGBA')
            image = image.resize((28, 28))
            image = image.convert('L')
            process_image(image)

def capture_image():
    """Ta en bild med kamera och gör en prediktion."""
    st.write("Startar kameran...")
    camera = initialize_camera()
    if not camera.isOpened():
        st.error("Kameran kunde inte startas.")
        return

    st.write("Kameran är igång.")
    cols = st.columns(2)
    captured_image_slot = cols[0].empty()
    preprocessed_image_slot = cols[1].empty()

    contrast = st.slider('Justera kontrast', min_value=-100, max_value=100, value=0, step=1, key='contrast_slider')
    brightness = st.slider('Justera ljusstyrka', min_value=0, max_value=100, value=50, step=1, key='brightness_slider')
    edge_detection = st.slider('Justera kantavkänning', min_value=0, max_value=100, value=0, step=1, key='edge_detection_slider')
    threshold_value = st.slider('Justera tröskelvärde', min_value=0, max_value=255, value=180, step=1, key='threshold_slider')

    take_picture = st.button('Ta en bild')
    
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Kunde inte fånga en bild.")
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((280, 280))
        captured_image_slot.image(img, caption='Bild från kamera.', use_container_width=True)

        img_np = np.array(img)
        img_morph = preprocess_camera_image(img_np, contrast, brightness, edge_detection, threshold_value)
        preprocessed_image_slot.image(img_morph, caption='Förbehandlad bild.', use_container_width=True)

        # Add a small delay to avoid high CPU usage
        time.sleep(0.1)

        if take_picture:
            break

    if take_picture:
        # Gör en prediktion på den förbehandlade bilden
        st.write("Gör en prediktion...")
        label, _ = predict_image(img_morph)
        st.write(f"Predikterade siffra: {label}" if label else "Inga siffror hittades.")

    release_camera()

# Load best model
st.write("Laddar den bästa modellen...")
model = None
if check_file_exists('svm_model.pkl'):
    model = load_model('svm')
    accuracy = load_results('svm')
    st.write(f"Support Vector Machine noggrannhet: {accuracy:.4f}")
elif check_file_exists('logistic_regression_model.pkl'):
    model = load_model('logistic_regression')
    accuracy = load_results('logistic_regression')
    st.write(f"Logistisk regression noggrannhet: {accuracy:.4f}")
elif check_file_exists('decision_tree_model.pkl'):
    model = load_model('decision_tree')
    accuracy = load_results('decision_tree')
    st.write(f"Beslutsmodell noggrannhet: {accuracy:.4f}")

if model is None:
    st.error("Modellen är inte laddad eller klassificeringen är inte slutförd. Vänligen kör klassificeringen först.")
else:
    st.title("---- Bildklassificering ----")
    options = ['Ladda upp en bild', 'Rita en bild', 'Ta en bild med kamera']
    option = st.selectbox('Välj ett alternativ:', options)

    if option == 'Ladda upp en bild':
        upload_image()
    elif option == 'Rita en bild':
        draw_image()
    elif option == 'Ta en bild med kamera':
        capture_image()