import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_resource
def load_model():
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# ---------------------------
# Check model files
# ---------------------------
if not os.path.exists('fake_news_model.pkl') or not os.path.exists('vectorizer.pkl'):
    st.error("⚠️ **MODEL FILES NOT FOUND!**")
    st.info("🔧 Please run `train_model.py` first to train the model.")
    st.code("python train_model.py", language="bash")
    st.stop()

model, vectorizer = load_model()
if model is None:
    st.error("❌ SYSTEM ERROR: Failed to load model. Please check the model files.")
    st.stop()

# ---------------------------
# Dark Theme Styling
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;700&display=swap');
    
    /* Main background - Deep dark with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 2px solid #7c3aed;
    }
    
    /* Text colors */
    .stApp, p, label, div {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #c084fc !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 20px rgba(192, 132, 252, 0.5);
    }
    
    /* Robot icon glow */
    .robot-icon {
        text-align: center;
        font-size: 80px;
        animation: float 3s ease-in-out infinite, glow-purple 2s ease-in-out infinite alternate;
        margin: 20px 0;
        filter: drop-shadow(0 0 30px #a855f7);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes glow-purple {
        from { filter: drop-shadow(0 0 20px #a855f7); }
        to { filter: drop-shadow(0 0 40px #c084fc); }
    }
    
    @keyframes glow-green {
        from { text-shadow: 0 0 10px #10b981, 0 0 20px #10b981; }
        to { text-shadow: 0 0 20px #10b981, 0 0 40px #34d399; }
    }
    
    /* Glowing text effects */
    .glow-purple {
        animation: glow-purple-text 2s ease-in-out infinite alternate;
    }
    
    .glow-green {
        animation: glow-green 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow-purple-text {
        from { text-shadow: 0 0 10px #a855f7, 0 0 20px #a855f7, 0 0 30px #7c3aed; }
        to { text-shadow: 0 0 20px #c084fc, 0 0 30px #a855f7, 0 0 40px #7c3aed; }
    }
    
    /* Input areas */
    .stTextArea textarea {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 2px solid #7c3aed !important;
        border-radius: 15px !important;
        color: #e0e0e0 !important;
        font-family: 'Roboto Mono', monospace !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.5) !important;
        transform: scale(1.01);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 30px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 0 30px rgba(124, 58, 237, 0.6) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #a855f7 0%, #c084fc 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 0 40px rgba(168, 85, 247, 0.8) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 32px !important;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.6);
    }
    
    [data-testid="stMetricLabel"] {
        color: #c084fc !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed 0%, #10b981 100%) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.6);
    }
    
    /* Cards and containers */
    div[class*="stMarkdown"] > div {
        border-radius: 15px;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background: rgba(26, 26, 46, 0.8) !important;
        border-radius: 15px !important;
        border-left: 4px solid #a855f7 !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 1px solid #7c3aed !important;
        border-radius: 8px !important;
        color: #10b981 !important;
    }
    
    /* Horizontal rules */
    hr {
        border-color: #7c3aed !important;
        opacity: 0.3;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #a855f7 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="robot-icon">🤖</div>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">⚡ AI FAKE NEWS DETECTOR ⚡</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-family: "Roboto Mono", monospace; color: #c084fc; font-size: 18px; margin-bottom: 30px;'>
    <strong>ADVANCED AI-POWERED ANALYSIS SYSTEM</strong><br>
    <span style='color: #10b981;'>Analyzing news articles with machine learning precision 🎯</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Layout: Input & Instructions
# ---------------------------
col1, col2 = st.columns([2.5, 1.5], gap="large")

with col1:
    st.markdown("### 📝 INPUT ARTICLE TEXT")
    news_text = st.text_area(
        "Paste the article text here:",
        height=350,
        placeholder="⌨️ Enter or paste a news article here for AI analysis...",
          )
    analyze_button = st.button("🔍 ANALYZE ARTICLE", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📋 INSTRUCTIONS")
    st.markdown("""
    <div style='background: rgba(26, 26, 46, 0.8); padding: 20px; border-radius: 15px; border: 2px solid #7c3aed; box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);'>
        <p style='color: #10b981; font-family: "Roboto Mono", monospace; margin-bottom: 10px;'><strong>📌 How to Use:</strong></p>
        <ol style='color: #e0e0e0; font-family: "Roboto Mono", monospace; font-size: 14px; line-height: 1.8;'>
            <li>Paste article text in the input box</li>
            <li>Click "ANALYZE ARTICLE" button</li>
            <li>Review AI confidence score</li>
            <li>Check probability breakdown</li>
        </ol>
        <p style='color: #a855f7; font-family: "Roboto Mono", monospace; margin-top: 15px; font-size: 13px;'>
            <strong>💡 Tip:</strong> Longer articles provide more accurate results!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Analyze button logic
# ---------------------------
if analyze_button:
    if not news_text.strip():
        st.warning("⚠️ **INPUT REQUIRED:** Please enter some text to analyze.")
    else:
        with st.spinner("🤖 AI ANALYSIS IN PROGRESS..."):
            text_vectorized = vectorizer.transform([news_text])
            
            # Handle both scikit-learn and XGBoost models
            try:
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
            except AttributeError:
                prediction = model.predict(text_vectorized)[0]
                probability = [0.5, 0.5] if prediction == 1 else [0.5, 0.5]

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📊 AI ANALYSIS RESULTS")
            
            if prediction == 1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.2) 100%); 
                     padding: 30px; border-radius: 20px; border: 3px solid #10b981; text-align: center; 
                     box-shadow: 0 0 40px rgba(16, 185, 129, 0.5);'>
                    <h2 class='glow-green' style='color: #10b981; font-family: "Orbitron", sans-serif; font-size: 32px; margin: 0;'>
                        ✅ REAL NEWS DETECTED
                    </h2>
                    <p style='color: #34d399; font-family: "Roboto Mono", monospace; font-size: 16px; margin-top: 10px;'>
                        This article appears to be <strong>AUTHENTIC</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                confidence = probability[1] * 100
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(192, 132, 252, 0.2) 100%); 
                     padding: 30px; border-radius: 20px; border: 3px solid #a855f7; text-align: center; 
                     box-shadow: 0 0 40px rgba(168, 85, 247, 0.5);'>
                    <h2 class='glow-purple' style='color: #c084fc; font-family: "Orbitron", sans-serif; font-size: 32px; margin: 0;'>
                        ⚠️ FAKE NEWS DETECTED
                    </h2>
                    <p style='color: #a855f7; font-family: "Roboto Mono", monospace; font-size: 16px; margin-top: 10px;'>
                        This article appears to be <strong>MISLEADING</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                confidence = probability[0] * 100

            st.progress(confidence / 100)
            st.metric("🎯 AI CONFIDENCE LEVEL", f"{confidence:.2f}%")

            # Probability breakdown
            st.markdown("### 📈 PROBABILITY BREAKDOWN")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🔴 FAKE PROBABILITY", f"{probability[0]*100:.2f}%")
            with col_b:
                st.metric("🟢 REAL PROBABILITY", f"{probability[1]*100:.2f}%")
            
            if confidence < 70:
                st.warning("⚠️ **LOW CONFIDENCE WARNING:** The AI model shows uncertainty about this article. Please verify from multiple trusted sources.")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### 🤖 ABOUT SYSTEM")
    st.markdown("""
    <div style='background: rgba(26, 26, 46, 0.8); padding: 15px; border-radius: 10px; border: 2px solid #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);'>
        <p style='color: #e0e0e0; font-family: "Roboto Mono", monospace; font-size: 13px; line-height: 1.6;'>
            This AI system uses advanced <span style='color: #10b981;'><strong>Natural Language Processing</strong></span> 
            and <span style='color: #a855f7;'><strong>Machine Learning</strong></span> algorithms to detect potentially 
            misleading or fabricated news content.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 MODEL STATISTICS")
    if os.path.exists('fake_news_model.pkl'):
        model_size = os.path.getsize('fake_news_model.pkl') / 1024
        st.metric("📦 Model Size", f"{model_size:.1f} KB")
    if os.path.exists('vectorizer.pkl'):
        vec_size = os.path.getsize('vectorizer.pkl') / 1024
        st.metric("🗂️ Vectorizer Size", f"{vec_size:.1f} KB")
    
    st.markdown("### ⚠️ DISCLAIMER")
    st.markdown("""
    <div style='background: rgba(26, 26, 46, 0.8); padding: 15px; border-radius: 10px; border: 2px solid #a855f7; box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);'>
        <p style='color: #e0e0e0; font-family: "Roboto Mono", monospace; font-size: 12px; line-height: 1.6;'>
            This is an <span style='color: #c084fc;'><strong>AI-assisted tool</strong></span>. 
            Results should not be considered definitive. Always verify information through 
            <span style='color: #10b981;'><strong>multiple trusted sources</strong></span>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-family: "Orbitron", sans-serif; color: #c084fc;'>
        <strong>Powered by AI 🤖</strong><br>
        <small style='font-family: "Roboto Mono", monospace; color: #10b981;'>Machine Learning • NLP</small>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: rgba(26, 26, 46, 0.8); 
     border-radius: 15px; border: 2px solid #7c3aed; box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);'>
    <p style='color: #c084fc; font-family: "Orbitron", sans-serif; font-size: 14px; margin: 0;'>
        <strong>🤖 AI-POWERED FAKE NEWS DETECTION SYSTEM</strong>
    </p>
    <p style='color: #10b981; font-family: "Roboto Mono", monospace; font-size: 12px; margin-top: 5px;'>
        Protecting information integrity through advanced machine learning
    </p>
</div>
""", unsafe_allow_html=True)