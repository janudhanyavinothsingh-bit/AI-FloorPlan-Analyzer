import streamlit as st
import cv2
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Floor Plan Analyzer Pro",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
    <style>
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
            padding: 3rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
        }
        
        .header-container h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header-container p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .ai-badge {
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 0.5rem;
        }
        
        .method-card {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(34, 197, 94, 0.1) 100%);
            border-left: 4px solid #4faafe;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None

# ==========================================
# LOAD AI MODELS
# ==========================================
@st.cache_resource
def load_yolo_model():
    """Load pre-trained YOLOv8 model for object detection"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.warning(f"Could not load YOLO model: {e}. Using CV-only mode.")
        return None

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess_image(file):
    """Enhanced image preprocessing"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    
    max_dim = 1200
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img_enhanced = cv2.merge([l, a, b])
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    
    return img_rgb

# ==========================================
# AI DETECTION - YOLO
# ==========================================
def detect_objects_with_yolo(img_rgb, model):
    """Detect objects using YOLOv8"""
    if model is None:
        return []
    
    try:
        results = model(img_rgb)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > 0.5:
                        detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': conf,
                            'class': cls
                        })
        
        return detections
    except Exception as e:
        st.warning(f"YOLO detection error: {e}")
        return []

# ==========================================
# COMPUTER VISION DETECTION
# ==========================================
def detect_rooms_cv(img_rgb):
    """Computer vision based room detection"""
    h, w = img_rgb.shape[:2]
    min_area = 800
    max_area = h * w * 0.5
    
    rooms = []
    room_id = 0
    
    pixels = img_rgb.reshape(-1, 3)
    pixels_quantized = (pixels // 20) * 20
    unique_colors = np.unique(pixels_quantized, axis=0)
    
    for color in unique_colors:
        lower = np.maximum(color - 30, 0)
        upper = np.minimum(color + 30, 255)
        
        mask = cv2.inRange(img_rgb, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if 0.2 < circularity < 0.95:
                    room_id += 1
                    rooms.append({
                        "id": room_id,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w_rect),
                        "h": int(h_rect),
                        "area": int(area),
                        "perimeter": int(perimeter),
                        "circularity": round(circularity, 3),
                        "type": classify_room(area),
                        "method": "Computer Vision"
                    })
    
    return rooms

# ==========================================
# HYBRID DETECTION - AI + CV
# ==========================================
def detect_rooms_hybrid(img_rgb, yolo_model):
    """Hybrid detection combining YOLO + Computer Vision"""
    
    cv_rooms = detect_rooms_cv(img_rgb)
    yolo_detections = detect_objects_with_yolo(img_rgb, yolo_model)
    
    hybrid_rooms = []
    room_id = 0
    
    for cv_room in cv_rooms:
        room_id += 1
        
        ai_confidence = 0
        if yolo_detections:
            for yolo_det in yolo_detections:
                x1, y1, x2, y2 = yolo_det['x1'], yolo_det['y1'], yolo_det['x2'], yolo_det['y2']
                cv_x1, cv_y1 = cv_room['x'], cv_room['y']
                cv_x2, cv_y2 = cv_x1 + cv_room['w'], cv_y1 + cv_room['h']
                
                inter_x1, inter_y1 = max(cv_x1, x1), max(cv_y1, y1)
                inter_x2, inter_y2 = min(cv_x2, x2), min(cv_y2, y2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    cv_area = cv_room['w'] * cv_room['h']
                    iou = inter_area / cv_area
                    
                    if iou > 0.3:
                        ai_confidence = max(ai_confidence, yolo_det['confidence'])
        
        hybrid_rooms.append({
            **cv_room,
            "id": room_id,
            "ai_confidence": round(ai_confidence, 3),
            "method": "Hybrid (AI + Vision)" if ai_confidence > 0 else "Computer Vision"
        })
    
    return hybrid_rooms

# ==========================================
# CLASSIFICATION
# ==========================================
def classify_room(area):
    """Classify room type based on area"""
    if area is None or area <= 0:
        return "Unknown"
    
    if area > 80000:
        return "Living Room"
    elif area > 50000:
        return "Hall"
    elif area > 35000:
        return "Bedroom"
    elif area > 20000:
        return "Kitchen"
    elif area > 10000:
        return "Bathroom"
    else:
        return "Closet"

# ==========================================
# STATISTICS
# ==========================================
def calculate_statistics(rooms):
    """Calculate analysis statistics"""
    if not rooms:
        return {}
    
    areas = [room['area'] for room in rooms]
    types = [room['type'] for room in rooms]
    methods = [room.get('method', 'Unknown') for room in rooms]
    
    stats = {
        'total_rooms': len(rooms),
        'total_area': sum(areas),
        'avg_area': int(np.mean(areas)),
        'max_area': max(areas),
        'min_area': min(areas),
        'room_types': dict(Counter(types)),
        'detection_methods': dict(Counter(methods)),
        'area_per_sqft': sum(areas) / 1000,
        'avg_ai_confidence': round(np.mean([r.get('ai_confidence', 0) for r in rooms]), 3)
    }
    
    return stats

# ==========================================
# VISUALIZATION
# ==========================================
def draw_results(img_rgb, rooms, show_labels=True, show_ids=True, show_ai_confidence=False):
    """Draw detection results on image"""
    img_display = img_rgb.copy()
    
    colors = {
        'Hall': (255, 107, 107),
        'Bedroom': (74, 144, 226),
        'Kitchen': (76, 175, 80),
        'Bathroom': (255, 152, 0),
        'Living Room': (156, 39, 172),
        'Closet': (158, 158, 158),
        'Unknown': (100, 100, 100)
    }
    
    for room in rooms:
        x, y, w, h = room["x"], room["y"], room["w"], room["h"]
        room_type = room["type"]
        color = colors.get(room_type, (100, 100, 100))
        
        thickness = 4 if room.get('ai_confidence', 0) > 0.5 else 3
        cv2.rectangle(img_display, (x, y), (x + w, y + h), color, thickness)
        
        if show_labels:
            label_parts = [f"{room_type}"]
            
            if show_ids:
                label_parts.append(f"ID:{room['id']}")
            
            if show_ai_confidence and room.get('ai_confidence', 0) > 0:
                label_parts.append(f"AI:{room.get('ai_confidence', 0):.2f}")
            
            label = " | ".join(label_parts)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            thickness_text = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
            y_label = max(y - 25, 0)
            
            cv2.rectangle(img_display, (x, y_label), (x + text_size[0] + 10, y_label + text_size[1] + 8), color, -1)
            cv2.putText(img_display, label, (x + 5, y_label + text_size[1] + 5), font, font_scale, (255, 255, 255), thickness_text)
    
    return img_display

# ==========================================
# EXPORT
# ==========================================
def export_to_csv(rooms):
    """Export results to CSV"""
    df = pd.DataFrame([
        {
            'Room ID': room['id'],
            'Type': room['type'],
            'Area (pixels)': room['area'],
            'Width (pixels)': room['w'],
            'Height (pixels)': room['h'],
            'Perimeter (pixels)': room['perimeter'],
            'Detection Method': room.get('method', 'Unknown'),
            'AI Confidence': f"{room.get('ai_confidence', 0):.3f}"
        }
        for room in rooms
    ])
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# MAIN APP
# ==========================================

st.markdown("""
    <div class="header-container">
        <h1>🏗 AI Floor Plan Analyzer Pro</h1>
        <p>Advanced Hybrid AI + Computer Vision Detection</p>
        <span class="ai-badge">POWERED BY YOLOV8 + DEEP LEARNING</span>
    </div>
""", unsafe_allow_html=True)

yolo_model = load_yolo_model()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Settings")
    
    detection_method = st.radio(
        "Detection Method",
        ["Hybrid (AI + Vision)", "Computer Vision Only", "AI Detection Only"]
    )
    
    show_labels = st.checkbox("Show Room Labels", value=True)
    show_ids = st.checkbox("Show Room IDs", value=True)
    show_ai_conf = st.checkbox("Show AI Confidence", value=True)
    
    st.markdown("---")
    
    st.subheader("Analysis History")
    if st.session_state.analysis_history:
        for i, analysis in enumerate(st.session_state.analysis_history[-5:], 1):
            method = analysis.get('method', 'Unknown')
            st.write(f"{i}. {analysis['filename']}")
            st.caption(f"{analysis['rooms']} rooms | {method}")
    else:
        st.write("No analyses yet")

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Floor Plan")
    uploaded_file = st.file_uploader(
        "Drag and drop your floor plan image here",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )

with col2:
    st.subheader("Detection Methods")
    st.markdown("""
    <div class="method-card">
    <strong>AI Detection (YOLO)</strong><br/>
    Deep learning for accurate object detection<br/>
    <small>Confidence score provided</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="method-card">
    <strong>Computer Vision</strong><br/>
    Color & shape-based detection<br/>
    <small>Fast & reliable for color-coded floors</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="method-card">
    <strong>Hybrid (Recommended)</strong><br/>
    AI validation + CV accuracy<br/>
    <small>Best of both worlds</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# PROCESSING
# ==========================================
if uploaded_file is not None:
    try:
        with st.spinner("Analyzing floor plan with AI..."):
            img_rgb = preprocess_image(uploaded_file)
            
            if detection_method == "Hybrid (AI + Vision)":
                rooms = detect_rooms_hybrid(img_rgb, yolo_model)
                method_used = "Hybrid"
            elif detection_method == "Computer Vision Only":
                rooms = detect_rooms_cv(img_rgb)
                method_used = "Computer Vision"
            else:
                yolo_detections = detect_objects_with_yolo(img_rgb, yolo_model)
                rooms = []
                for i, det in enumerate(yolo_detections, 1):
                    rooms.append({
                        "id": i,
                        "x": det['x1'],
                        "y": det['y1'],
                        "w": det['x2'] - det['x1'],
                        "h": det['y2'] - det['y1'],
                        "area": (det['x2'] - det['x1']) * (det['y2'] - det['y1']),
                        "perimeter": 0,
                        "circularity": 0,
                        "type": "Detected Object",
                        "method": "AI (YOLO)",
                        "ai_confidence": det['confidence']
                    })
                method_used = "AI Only"
            
            stats = calculate_statistics(rooms)
            
            st.session_state.analysis_history.append({
                'filename': uploaded_file.name,
                'rooms': len(rooms),
                'method': method_used,
                'timestamp': datetime.now()
            })
        
        if rooms:
            st.success(f"Detected {len(rooms)} rooms using {method_used}!")
        else:
            st.warning("No rooms detected. Try a different image.")
        
        st.markdown("---")
        
        # ==========================================
        # RESULTS TABS
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs(["Visualization", "Statistics", "Details", "Export"])
        
        # TAB 1: VISUALIZATION
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                img_display = draw_results(img_rgb, rooms, show_labels, show_ids, show_ai_conf)
                st.image(img_display, caption=f"Detected Rooms ({method_used})", use_column_width=True)
            
            with col2:
                st.subheader("Room Legend")
                legend = {
                    'Hall': '🔴',
                    'Bedroom': '🔵',
                    'Kitchen': '🟢',
                    'Bathroom': '🟠',
                    'Living Room': '🟣',
                    'Closet': '⚪',
                }
                for room_type, emoji in legend.items():
                    st.write(f"{emoji} {room_type}")
                
                if show_ai_conf:
                    st.info("Thick borders = High AI Confidence")
        
        # TAB 2: STATISTICS
        with tab2:
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rooms", stats['total_rooms'])
                with col2:
                    st.metric("Total Area", f"{stats['total_area']:,} px")
                with col3:
                    st.metric("Average Area", f"{stats['avg_area']:,} px")
                with col4:
                    st.metric("AI Confidence", f"{stats['avg_ai_confidence']:.2f}")
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Room Type Distribution")
                    room_counts = pd.DataFrame(
                        list(stats['room_types'].items()),
                        columns=['Room Type', 'Count']
                    )
                    st.bar_chart(room_counts.set_index('Room Type'))
                
                with col2:
                    st.subheader("Detection Method Breakdown")
                    for method, count in stats['detection_methods'].items():
                        st.write(f"**{method}:** {count} room(s)")
        
        # TAB 3: DETAILS
        with tab3:
            if rooms:
                st.subheader("Detailed Room Information")
                
                df_rooms = pd.DataFrame([
                    {
                        'ID': room['id'],
                        'Type': room['type'],
                        'Area (px)': f"{room['area']:,}",
                        'Method': room.get('method', 'Unknown'),
                        'AI Confidence': f"{room.get('ai_confidence', 0):.3f}",
                    }
                    for room in sorted(rooms, key=lambda x: x['area'], reverse=True)
                ])
                
                st.dataframe(df_rooms, use_container_width=True, hide_index=True)
        
        # TAB 4: EXPORT
        with tab4:
            st.subheader("Export Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = export_to_csv(rooms)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"floor_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                img_display = draw_results(img_rgb, rooms, show_labels, show_ids, show_ai_conf)
                is_success, buffer = cv2.imencode('.png', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
                
                st.download_button(
                    label="Download Image",
                    data=buffer.tobytes(),
                    file_name=f"floor_plan_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            with col3:
                summary_text = f"""FLOOR PLAN ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}
Detection Method: {method_used}

SUMMARY
Total Rooms: {stats['total_rooms']}
Total Area: {stats['total_area']:,} pixels
Average AI Confidence: {stats['avg_ai_confidence']:.3f}

ROOM BREAKDOWN
"""
                for room_type, count in stats['room_types'].items():
                    summary_text += f"{room_type}: {count}\n"
                
                st.download_button(
                    label="Download Report",
                    data=summary_text,
                    file_name=f"floor_plan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 2rem;'>
        <p>AI Floor Plan Analyzer Pro v2.0 | Hybrid AI + Computer Vision</p>
        <p>Powered by YOLOv8 Deep Learning & OpenCV</p>
        <p>© 2024 All rights reserved</p>
    </div>
""", unsafe_allow_html=True)
