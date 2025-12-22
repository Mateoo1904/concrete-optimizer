import streamlit as st
import sys
from pathlib import Path

# Thêm đường dẫn
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

st.set_page_config(page_title="Concrete Optimizer", layout="wide")

st.title("🏗️ Concrete Mix Optimizer - TEST")
st.success("✅ Streamlit is working!")

# Thêm form đơn giản
with st.form("test_form"):
    st.subheader("Test Input Form")
    strength = st.slider("Target Strength (MPa)", 20, 60, 40)
    slump = st.slider("Target Slump (mm)", 50, 200, 120)
    
    if st.form_submit_button("Run Test"):
        st.info(f"Input received: f_c = {strength}MPa, slump = {slump}mm")
        st.balloons()

st.markdown("---")
st.write("If you see this, Streamlit is working correctly! 🎉")
