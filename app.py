# Importing Required Packages
import streamlit as st
import tempfile
import os
from main import calculate_paintable_area, find_defects  # ✅ Import defect detection too

# Streamlit Page Config
st.set_page_config(
    page_title="Building Paintable Area Calculator",
    layout="wide",
    page_icon="🏠"
)

st.title("🏗️ Building Paintable Area Estimator")
st.markdown(
    "Upload up to **five building images** and configure thresholds & dimensions to estimate the total paintable area."
)

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Configuration")

# Threshold sliders
building_threshold = st.sidebar.slider(
    "Building Detection Threshold", min_value=0.01, max_value=1.0, value=0.5, step=0.01
)
object_threshold = st.sidebar.slider(
    "Object Detection Threshold", min_value=0.01, max_value=1.0, value=0.5, step=0.01
)
defect_detection_threshold = st.sidebar.slider(
    "Defect Detection Threshold", min_value=0.01, max_value=1.0, value=0.2, step=0.01
)

# Real-world height inputs
real_building_height = st.sidebar.number_input(
    "Real Building Height (feet)", min_value=1.0, max_value=200.0, value=40.0, step=1.0
)
real_window_height = st.sidebar.number_input(
    "Real Window Height (feet)", min_value=1.0, max_value=20.0, value=3.5, step=0.5
)
real_door_height = st.sidebar.number_input(
    "Real Door Height (feet)", min_value=1.0, max_value=20.0, value=7.0, step=0.5
)
real_pipe_height = st.sidebar.number_input(
    "Real Pipe Height (feet)", min_value=1.0, max_value=50.0, value=20.0, step=0.5
)

st.sidebar.markdown("---")

# --- Restart Session Button ---
if st.sidebar.button("🔄 Restart Session"):
    st.session_state.clear()
    st.experimental_rerun()

st.sidebar.info("Adjust thresholds and heights as needed, then press the button below.")
st.sidebar.caption("Default values are tuned for most standard buildings.")

# --- File Upload Section ---
st.subheader("📸 Upload Building Images")
st.write("Please upload up to **five clear images** of the same building from different angles.")

uploaded_files = st.file_uploader(
    "Upload up to 5 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Display uploaded images side by side
if uploaded_files:
    st.markdown("### 🖼️ Uploaded Images")
    cols = st.columns(len(uploaded_files))
    for i, file in enumerate(uploaded_files):
        with cols[i]:
            st.image(file, caption=f"Image {i+1}", use_container_width=True)

# --- Calculate Button ---
if st.button("🧮 Calculate Paintable Area", use_container_width=True):
    if len(uploaded_files) < 2:
        st.error("⚠️ Please upload at least two images to proceed.")
    elif len(uploaded_files) > 5:
        st.error("⚠️ Please upload a maximum of five images.")
    else:
        st.info("🔍 Running paintable area and defect analysis... Please wait.")

        # Save uploaded files temporarily
        temp_paths = []
        for file in uploaded_files:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(temp_path)

        # Run the main function and capture logs
        with st.spinner("Processing images..."):
            try:
                # Only pass the first 4 images if 5 are uploaded
                args = dict(
                    image1_path=temp_paths[0],
                    image2_path=temp_paths[1],
                    image3_path=temp_paths[2] if len(temp_paths) > 2 else temp_paths[1],
                    image4_path=temp_paths[3] if len(temp_paths) > 3 else temp_paths[2],
                    building_threshold=building_threshold,
                    object_threshold=object_threshold,
                    real_building_height=real_building_height,
                    real_window_height=real_window_height,
                    real_door_height=real_door_height,
                    real_pipe_height=real_pipe_height
                )

                # ✅ Unpack full tuple from calculate_paintable_area
                final_h, final_w, wc, dc, pc, total_building_area, total_wdp_area, paintable_area = calculate_paintable_area(**args)

                # ✅ Run defect detection
                defect_labels = find_defects(
                    temp_paths[0],
                    temp_paths[1],
                    temp_paths[2] if len(temp_paths) > 2 else temp_paths[1],
                    temp_paths[3] if len(temp_paths) > 3 else temp_paths[2],
                    defect_threshold=defect_detection_threshold
                )

                st.success("✅ Calculation and defect detection completed successfully!")

                # --- Display Results ---
                st.markdown("### 📊 Calculation Results")

                col1, col2_null, col3 = st.columns(3)
                col1.metric("Building Height", float(final_h))
                col3.metric("Building Width", float(final_w))

                col4, col5, col6 = st.columns(3)
                col4.metric("🪟 Window Count", int(wc))
                col5.metric("🚪 Door Count", int(dc))
                col6.metric("🧱 Pipe Count", int(pc))

                col7, col8, col9 = st.columns(3)
                col7.metric("🏢 Total Building Area", f"{total_building_area:.2f} sq.ft")
                col8.metric("🚪🪟 Total WDP Area", f"{total_wdp_area:.2f} sq.ft")
                col9.metric("🎨 Paintable Area", f"{paintable_area:.2f} sq.ft")

                st.markdown("---")

                # ✅ Display Defect Labels
                st.markdown("### 🧩 Detected Defects on Building Walls")
                if defect_labels:
                    unique_labels = sorted(set([lbl for lbl in defect_labels if isinstance(lbl, str) and lbl.strip() and lbl.lower() != "wall"]))
                    if unique_labels:
                        st.write(", ".join(unique_labels))
                    else:
                        st.info("No major wall defects detected.")
                else:
                    st.info("No defects found in the uploaded images.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

        # Clean up temp files
        for p in temp_paths:
            try:
                os.remove(p)
            except:
                pass
else:
    st.info("👆 Upload images and press the button to calculate paintable area.")