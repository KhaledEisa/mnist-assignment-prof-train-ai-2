"""
MNIST Digit Classifier
Draw digits and test our trained models
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import joblib
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-digit {
        font-size: 5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence {
        font-size: 1.5rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔢 MNIST Digit Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Draw a digit and let AI recognize it!</p>', unsafe_allow_html=True)

# load models
@st.cache_resource
def load_models():
    """load all 3 trained models"""
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        nn_model = keras.models.load_model('models/neural_network_model.h5')
        cnn_model = keras.models.load_model('models/cnn_model.h5')
        return rf_model, nn_model, cnn_model, None
    except Exception as e:
        return None, None, None, str(e)

# Preprocessing function
def preprocess_image(image_data):
    """preprocess the canvas image to match MNIST format"""
    # convert to grayscale
    if len(image_data.shape) == 3:
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_data
    
    # threshold to get clean binary image
    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # find non-zero pixels (the drawn digit)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # crop to bounding box with some padding
        padding = 40
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        cropped = gray[y:y+h, x:x+w]
        
        # resize maintaining aspect ratio - MNIST style
        # make it square first
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - cropped.shape[1]) // 2
        y_offset = (size - cropped.shape[0]) // 2
        square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        
        # now resize to 20x20
        resized_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        
        # place in center of 28x28 (like MNIST)
        final = np.zeros((28, 28), dtype=np.uint8)
        final[4:24, 4:24] = resized_20
        
    else:
        final = np.zeros((28, 28), dtype=np.uint8)
    
    # normalize to 0-1
    normalized = final.astype('float32') / 255.0
    
    # prepare for models
    img_ml = normalized.reshape(1, -1)
    img_nn = normalized.reshape(1, -1)
    img_cnn = normalized.reshape(1, 28, 28, 1)
    
    return img_ml, img_nn, img_cnn, final

# main app
def main():
    # load models
    rf_model, nn_model, cnn_model, error = load_models()
    
    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("Make sure you trained the models first (run the notebook)")
        return
    
    st.success("✅ Models loaded!")
    
    # sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    model_choice = st.sidebar.selectbox(
        "Select Model:",
        ["Random Forest", "Neural Network", "CNN"],
        index=2
    )
    
    stroke_width = st.sidebar.slider("Brush Size:", 5, 50, 25)
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ How to use")
    st.sidebar.markdown("""
    1. Draw a digit (0-9)
    2. Pick a model
    3. Click Predict
    4. Clear and try again
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Info")
    
    model_info = {
        "Random Forest": {
            "Type": "Traditional ML",
            "Description": "Ensemble of decision trees",
            "Best for": "Fast inference"
        },
        "Neural Network": {
            "Type": "Deep Learning",
            "Description": "Fully connected layers",
            "Best for": "Speed and accuracy balance"
        },
        "CNN": {
            "Type": "Deep Learning",
            "Description": "Convolutional network",
            "Best for": "Best accuracy"
        }
    }
    
    info = model_info[model_choice]
    st.sidebar.markdown(f"**Type:** {info['Type']}")
    st.sidebar.markdown(f"**Description:** {info['Description']}")
    st.sidebar.markdown(f"**Best for:** {info['Best for']}")
    
    # canvas and prediction columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✏️ Draw Here")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                if np.sum(canvas_result.image_data) > 0:
                    # preprocess
                    img_ml, img_nn, img_cnn, processed_img = preprocess_image(canvas_result.image_data)
                    
                    st.write("**Processed (28x28):**")
                    st.image(processed_img, width=150)
                    
                    # predict based on selected model
                    if model_choice == "Random Forest":
                        prediction = rf_model.predict(img_ml)[0]
                        probabilities = rf_model.predict_proba(img_ml)[0]
                        confidence = np.max(probabilities) * 100
                        
                    elif model_choice == "Neural Network":
                        probabilities = nn_model.predict(img_nn, verbose=0)[0]
                        prediction = np.argmax(probabilities)
                        confidence = probabilities[prediction] * 100
                        
                    else:  # CNN
                        probabilities = cnn_model.predict(img_cnn, verbose=0)[0]
                        prediction = np.argmax(probabilities)
                        confidence = probabilities[prediction] * 100
                    
                    # show prediction
                    st.markdown(f"""
                        <div class="prediction-box">
                            <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted Digit:</p>
                            <p class="prediction-digit">{prediction}</p>
                            <p class="confidence">Confidence: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # probability chart
                    st.write("**Probability Distribution:**")
                    import pandas as pd
                    df = pd.DataFrame({
                        'Digit': list(range(10)),
                        'Probability': probabilities
                    })
                    st.bar_chart(df.set_index('Digit'))
                    
                    # top 3
                    st.write("**Top 3:**")
                    top_3_indices = np.argsort(probabilities)[-3:][::-1]
                    for i, idx in enumerate(top_3_indices, 1):
                        st.write(f"{i}. Digit **{idx}** - {probabilities[idx]*100:.2f}%")
                    
                else:
                    st.warning("⚠️ Draw something first!")
            else:
                st.warning("⚠️ Canvas is empty!")
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # extra info
    with st.expander("📈 Model Comparison"):
        st.markdown("""
        | Model | Accuracy | Speed | Best for |
        |-------|----------|-------|----------|
        | Random Forest | ~97% | Fast | Quick results |
        | Neural Network | ~98% | Medium | Balanced |
        | CNN | ~99% | Slower | Best accuracy |
        
        CNN works best because it understands the 2D structure of images
        """)
    
    with st.expander("🎓 About"):
        st.markdown("""
        ### MNIST Digit Classification
        
        3 different approaches to recognize handwritten digits:
        
        1. **Random Forest**: Traditional ML with decision trees
        2. **Neural Network**: Fully connected layers
        3. **CNN**: Convolutional network (best one)
        
        **Dataset**: MNIST
        - 60,000 training images
        - 10,000 test images
        - 28x28 grayscale
        - 10 classes (0-9)
        """)

if __name__ == "__main__":
    main()
