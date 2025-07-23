import gradio as gr
import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
import tempfile
import sqlite3
import datetime
import matplotlib.pyplot as plt
import json

# Load class names from classes.txt
CLASSES_PATH = "datasets/data/taco/classes.txt"
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = [str(i) for i in range(60)]  # fallback

# Load waste categories
WASTE_CATEGORIES_PATH = "datasets/data/taco/waste_categories.json"
if os.path.exists(WASTE_CATEGORIES_PATH):
    with open(WASTE_CATEGORIES_PATH, 'r') as f:
        WASTE_CATEGORIES = json.load(f)
else:
    WASTE_CATEGORIES = {"reusable": {}, "non_reusable": {}}

# Load language translations
LANGUAGES_PATH = "datasets/data/taco/languages.json"
if os.path.exists(LANGUAGES_PATH):
    with open(LANGUAGES_PATH, 'r', encoding='utf-8') as f:
        LANGUAGES = json.load(f)
else:
    LANGUAGES = {"en": {}}

# Default language
DEFAULT_LANGUAGE = "en"

def get_text(key, language=DEFAULT_LANGUAGE):
    """Get translated text for a given key and language"""
    if language in LANGUAGES and key in LANGUAGES[language]:
        return LANGUAGES[language][key]
    elif key in LANGUAGES.get(DEFAULT_LANGUAGE, {}):
        return LANGUAGES[DEFAULT_LANGUAGE][key]
    else:
        return key  # Fallback to key if translation not found

def get_waste_info(class_name):
    """Get waste category and recyclability info for a class"""
    for category, items in WASTE_CATEGORIES.items():
        if class_name in items:
            return items[class_name]
    return {"category": "unknown", "recyclable": False, "impact": "unknown", "sorting": "unknown", "instructions": "Check local guidelines", "tips": "When in doubt, put in general waste", "landfill_co2": 0.1, "recycling_co2": 0.1, "carbon_savings": 0.0, "impact_score": 5}

def get_sorting_icon(sorting_type):
    """Get emoji icon for sorting type"""
    icons = {
        "recycling": "♻️",
        "general_waste": "🗑️",
        "compost": "🌱",
        "special_waste": "⚠️",
        "donation": "🎁",
        "unknown": "❓"
    }
    return icons.get(sorting_type, "❓")

def calculate_carbon_equivalents(co2_kg):
    """Convert CO2 to relatable equivalents"""
    equivalents = {
        "trees": co2_kg / 22,  # 1 tree absorbs ~22kg CO2/year
        "miles_driven": co2_kg / 0.404,  # 1 mile = 0.404kg CO2
        "lightbulb_hours": co2_kg / 0.0001,  # 1 hour = 0.0001kg CO2
        "smartphone_charges": co2_kg / 0.0003  # 1 charge = 0.0003kg CO2
    }
    return equivalents

def calculate_trash_percentage(detections):
    """Calculate percentage of detected objects that are trash"""
    if not detections:
        return 0.0
    
    total_objects = len(detections)
    trash_objects = 0
    
    for detection in detections:
        class_name = detection.split(':')[0].strip()
        waste_info = get_waste_info(class_name)
        if waste_info["category"] == "non_reusable":
            trash_objects += 1
    
    return (trash_objects / total_objects) * 100

def analyze_detections(boxes, confidences, language=DEFAULT_LANGUAGE):
    """Analyze detections and return detailed information"""
    detections = []
    reusable_count = 0
    non_reusable_count = 0
    recyclable_count = 0
    sorting_summary = {"recycling": 0, "general_waste": 0, "compost": 0, "special_waste": 0, "donation": 0}
    total_carbon_savings = 0.0
    total_landfill_co2 = 0.0
    total_recycling_co2 = 0.0
    impact_scores = []
    
    if boxes is not None and boxes.cls.numel() > 0:
        for c, conf in zip(boxes.cls, boxes.conf):
            class_idx = int(c.item())
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            confidence = conf.item()
            
            waste_info = get_waste_info(class_name)
            category = waste_info["category"]
            recyclable = waste_info["recyclable"]
            impact = waste_info["impact"]
            sorting = waste_info["sorting"]
            instructions = waste_info["instructions"]
            tips = waste_info["tips"]
            landfill_co2 = waste_info["landfill_co2"]
            recycling_co2 = waste_info["recycling_co2"]
            carbon_savings = waste_info["carbon_savings"]
            impact_score = waste_info["impact_score"]
            
            # Count categories
            if category == "reusable":
                reusable_count += 1
            elif category == "non_reusable":
                non_reusable_count += 1
            
            if recyclable:
                recyclable_count += 1
            
            # Count sorting types
            if sorting in sorting_summary:
                sorting_summary[sorting] += 1
            
            # Calculate carbon footprint
            total_carbon_savings += carbon_savings
            total_landfill_co2 += landfill_co2
            total_recycling_co2 += recycling_co2
            impact_scores.append(impact_score)
            
            # Create detailed detection string with carbon info
            icon = get_sorting_icon(sorting)
            detection_str = f"{icon} **{class_name}** ({confidence:.2f})\n"
            detection_str += f"   📍 **{get_text('sorting', language)}:** {sorting.replace('_', ' ').title()}\n"
            detection_str += f"   📋 **{get_text('instructions', language)}:** {instructions}\n"
            detection_str += f"   💡 **{get_text('tip', language)}:** {tips}\n"
            detection_str += f"   ♻️ **{get_text('recyclable', language)}:** {get_text('yes' if recyclable else 'no', language)}\n"
            detection_str += f"   🌍 **{get_text('impact', language)}:** {impact.title()} (Score: {impact_score}/10)\n"
            detection_str += f"   🌱 **{get_text('co2_savings', language)}:** {carbon_savings:.3f} kg CO2\n"
            detections.append(detection_str)
    else:
        detections.append(get_text("no_detections", language))
    
    return detections, reusable_count, non_reusable_count, recyclable_count, sorting_summary, total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores

def create_sorting_summary(sorting_summary, language=DEFAULT_LANGUAGE):
    """Create a summary of sorting recommendations"""
    total_items = sum(sorting_summary.values())
    if total_items == 0:
        return get_text("no_detections", language)
    
    summary = f"🗂️ **{get_text('sorting_summary', language)}:**\n"
    for sorting_type, count in sorting_summary.items():
        if count > 0:
            icon = get_sorting_icon(sorting_type)
            percentage = (count / total_items) * 100
            summary += f"   {icon} **{sorting_type.replace('_', ' ').title()}:** {count} items ({percentage:.1f}%)\n"
    
    return summary

def create_carbon_summary(total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores, language=DEFAULT_LANGUAGE):
    """Create a comprehensive carbon footprint summary"""
    if not impact_scores:
        return get_text("no_carbon_data", language)
    
    avg_impact = sum(impact_scores) / len(impact_scores)
    carbon_equivalents = calculate_carbon_equivalents(total_carbon_savings)
    
    # Determine impact level and recommendation
    if avg_impact > 7:
        impact_level = get_text("high", language)
        recommendation = get_text("consider_reusable", language)
    elif avg_impact > 4:
        impact_level = get_text("medium", language)
        recommendation = get_text("good_recycling", language)
    else:
        impact_level = get_text("low", language)
        recommendation = get_text("excellent_choice", language)
    
    summary = f"""
🌱 **{get_text('carbon_footprint', language)}:**
• **{get_text('total_co2_savings', language)}:** {total_carbon_savings:.3f} kg CO2
• **{get_text('landfill_co2', language)}:** {total_landfill_co2:.3f} kg CO2
• **{get_text('recycling_co2', language)}:** {total_recycling_co2:.3f} kg CO2
• **{get_text('average_impact', language)}:** {avg_impact:.1f}/10

🌍 **{get_text('environmental_equivalents', language)}:**
• **{get_text('trees_saved', language)}:** {carbon_equivalents['trees']:.1f} {get_text('trees_absorb', language)}
• **{get_text('miles_driven', language)}:** {carbon_equivalents['miles_driven']:.1f} {get_text('miles_avoided', language)}
• **{get_text('lightbulb_hours', language)}:** {carbon_equivalents['lightbulb_hours']:.0f} {get_text('hours_energy_saved', language)}
• **{get_text('smartphone_charges', language)}:** {carbon_equivalents['smartphone_charges']:.0f} {get_text('charges_saved', language)}

💚 **{get_text('environmental_impact', language)}:**
• **{get_text('impact_level', language)}:** {impact_level}
• **{get_text('recommendation', language)}:** {recommendation}
"""
    
    return summary

# Patch torch.load to use weights_only=False by default
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch globally
torch.load = patched_torch_load

# Try to load the trained model, fallback to pre-trained if not available
try:
    model = YOLO("saved_models/yolov8n_custom/weights/best.pt")
    print("Loaded trained model")
except Exception as e:
    print(f"Could not load trained model: {e}")
    print("Using pre-trained model instead")
    model = YOLO("yolov8n.pt")

# --- DATABASE SETUP ---
DB_PATH = "waste_stats.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source TEXT,
        class_counts TEXT,
        reusable_count INTEGER,
        non_reusable_count INTEGER,
        recyclable_count INTEGER,
        trash_percentage REAL
    )''')
    conn.commit()
    conn.close()
init_db()

def log_detection(source, class_counts, reusable_count, non_reusable_count, recyclable_count, trash_percentage):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO detections (timestamp, source, class_counts, reusable_count, non_reusable_count, recyclable_count, trash_percentage) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.datetime.now().isoformat(), source, str(class_counts), reusable_count, non_reusable_count, recyclable_count, trash_percentage))
    conn.commit()
    conn.close()

def get_recent_detections(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, source, class_counts, reusable_count, non_reusable_count, recyclable_count, trash_percentage FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]  # reverse for chronological order

def get_class_totals():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT class_counts FROM detections")
    rows = c.fetchall()
    conn.close()
    total_counts = {name: 0 for name in CLASS_NAMES}
    for (counts_str,) in rows:
        try:
            counts = eval(counts_str)
            for k, v in counts.items():
                if k in total_counts:
                    total_counts[k] += v
        except Exception:
            continue
    return total_counts

# --- DETECTION FUNCTIONS ---
def detect_image(image, language=DEFAULT_LANGUAGE):
    """Detect objects in uploaded image"""
    if image is None:
        return None, get_text("please_upload_image", language)
    
    result = model(image)[0]
    boxes = result.boxes
    annotated_img = result.plot()
    
    # Analyze detections
    detections, reusable_count, non_reusable_count, recyclable_count, sorting_summary, total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores = analyze_detections(boxes, boxes.conf if boxes is not None else None, language)
    
    # Calculate trash percentage
    trash_percentage = calculate_trash_percentage(detections)
    
    # Create sorting summary
    sorting_info = create_sorting_summary(sorting_summary, language)
    
    # Create carbon footprint summary
    carbon_info = create_carbon_summary(total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores, language)
    
    # Create comprehensive summary
    total_objects = reusable_count + non_reusable_count
    summary = f"""
📊 **{get_text('detection_summary', language)}:**
• {get_text('total_objects', language)}: {total_objects}
• {get_text('reusable_items', language)}: {reusable_count} ({(reusable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('non_reusable_items', language)}: {non_reusable_count} ({(non_reusable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('recyclable_items', language)}: {recyclable_count} ({(recyclable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('trash_percentage', language)}: {trash_percentage:.1f}%

{sorting_info}

{carbon_info}

🗑️ **{get_text('detailed_analysis', language)}:**
{chr(10).join(detections)}
"""
    
    # Log to database
    class_counter = {name: 0 for name in CLASS_NAMES}
    if boxes is not None and boxes.cls.numel() > 0:
        for c in boxes.cls:
            class_idx = int(c.item())
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            class_counter[class_name] += 1
    
    log_detection("image", class_counter, reusable_count, non_reusable_count, recyclable_count, trash_percentage)
    
    return annotated_img, summary

def detect_webcam(image, language=DEFAULT_LANGUAGE):
    """Detect objects in webcam frame"""
    if image is None:
        return None, get_text("no_webcam_input", language)
    
    result = model(image)[0]
    boxes = result.boxes
    annotated_img = result.plot()
    
    # Analyze detections
    detections, reusable_count, non_reusable_count, recyclable_count, sorting_summary, total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores = analyze_detections(boxes, boxes.conf if boxes is not None else None, language)
    
    # Calculate trash percentage
    trash_percentage = calculate_trash_percentage(detections)
    
    # Create sorting summary
    sorting_info = create_sorting_summary(sorting_summary, language)
    
    # Create carbon footprint summary
    carbon_info = create_carbon_summary(total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores, language)
    
    # Create comprehensive summary
    total_objects = reusable_count + non_reusable_count
    summary = f"""
📊 **{get_text('detection_summary', language)}:**
• {get_text('total_objects', language)}: {total_objects}
• {get_text('reusable_items', language)}: {reusable_count} ({(reusable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('non_reusable_items', language)}: {non_reusable_count} ({(non_reusable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('recyclable_items', language)}: {recyclable_count} ({(recyclable_count/total_objects*100) if total_objects > 0 else 0:.1f}%)
• {get_text('trash_percentage', language)}: {trash_percentage:.1f}%

{sorting_info}

{carbon_info}

🗑️ **{get_text('detailed_analysis', language)}:**
{chr(10).join(detections)}
"""
    
    # Log to database
    class_counter = {name: 0 for name in CLASS_NAMES}
    if boxes is not None and boxes.cls.numel() > 0:
        for c in boxes.cls:
            class_idx = int(c.item())
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            class_counter[class_name] += 1
    
    log_detection("webcam", class_counter, reusable_count, non_reusable_count, recyclable_count, trash_percentage)
    
    return annotated_img, summary

def detect_video(video, language=DEFAULT_LANGUAGE):
    """Detect objects in each frame of the uploaded video and return annotated video."""
    if video is None:
        return None, get_text("please_upload_video", language)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
    frame_count = 0
    class_counter = {name: 0 for name in CLASS_NAMES}
    total_reusable = 0
    total_non_reusable = 0
    total_recyclable = 0
    sorting_summary = {"recycling": 0, "general_waste": 0, "compost": 0, "special_waste": 0, "donation": 0}
    total_carbon_savings = 0.0
    total_landfill_co2 = 0.0
    total_recycling_co2 = 0.0
    impact_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(rgb_frame)[0]
        annotated = result.plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        # Count detections in this frame
        boxes = result.boxes
        if boxes is not None and boxes.cls.numel() > 0:
            for c in boxes.cls:
                class_idx = int(c.item())
                class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
                class_counter[class_name] += 1
                
                # Count categories
                waste_info = get_waste_info(class_name)
                if waste_info["category"] == "reusable":
                    total_reusable += 1
                elif waste_info["category"] == "non_reusable":
                    total_non_reusable += 1
                
                if waste_info["recyclable"]:
                    total_recyclable += 1
                
                # Count sorting types
                sorting = waste_info["sorting"]
                if sorting in sorting_summary:
                    sorting_summary[sorting] += 1
                
                # Calculate carbon footprint
                total_carbon_savings += waste_info["carbon_savings"]
                total_landfill_co2 += waste_info["landfill_co2"]
                total_recycling_co2 += waste_info["recycling_co2"]
                impact_scores.append(waste_info["impact_score"])
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calculate overall statistics
    total_objects = sum(class_counter.values())
    trash_percentage = (total_non_reusable / total_objects * 100) if total_objects > 0 else 0
    
    # Create sorting summary
    sorting_info = create_sorting_summary(sorting_summary, language)
    
    # Create carbon footprint summary
    carbon_info = create_carbon_summary(total_carbon_savings, total_landfill_co2, total_recycling_co2, impact_scores, language)
    
    log_detection("video", class_counter, total_reusable, total_non_reusable, total_recyclable, trash_percentage)
    
    summary = f"{get_text('processed_frames', language).format(frames=frame_count)}\n\n📊 **{get_text('video_analysis', language)}:**\n• {get_text('total_objects_video', language)}: {total_objects}\n• {get_text('reusable_video', language)}: {total_reusable}\n• {get_text('non_reusable_video', language)}: {total_non_reusable}\n• {get_text('recyclable_video', language)}: {total_recyclable}\n• {get_text('trash_percentage_video', language)}: {trash_percentage:.1f}%\n\n{sorting_info}\n\n{carbon_info}"
    
    return temp_out.name, summary

# --- DASHBOARD TAB ---
def dashboard_view():
    # Table of recent detections
    rows = get_recent_detections()
    table = [[ts, src, f"R:{r} NR:{nr} RC:{rc} TP:{tp:.1f}%" if r is not None else cc] for ts, src, cc, r, nr, rc, tp in rows]
    
    # Bar chart of total detections per class
    totals = get_class_totals()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(list(totals.keys()), list(totals.values()))
    ax.set_ylabel('Count')
    ax.set_xlabel('Class')
    ax.set_title('Total Detections per Class')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    return table, fig

# --- GRADIO UI ---
with gr.Blocks(title="WasteVision - Multi-Class Waste Detection") as demo:
    # Language selector
    language_selector = gr.Dropdown(
        choices=["English", "Español", "Français", "Deutsch", "हिन्दी"],
        value="English",
        label="🌐 Language / Idioma / Langue / Sprache / भाषा",
        interactive=True
    )
    
    # Language mapping
    language_map = {"English": "en", "Español": "es", "Français": "fr", "Deutsch": "de", "हिन्दी": "hi"}
    
    gr.Markdown("# 🗑️ WasteVision - Smart Waste Detection")
    gr.Markdown("Detect and categorize waste objects as reusable vs non-reusable, with recyclability and environmental impact analysis.")
    
    with gr.Tabs():
        with gr.TabItem("📷 Image Upload"):
            gr.Markdown("### Upload an image to detect and categorize waste objects")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_button = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Detection Result")
                    image_detections = gr.Markdown(label="Analysis Results")
            
            def detect_image_wrapper(image, language):
                return detect_image(image, language_map.get(language, "en"))
            
            image_button.click(
                fn=detect_image_wrapper,
                inputs=[image_input, language_selector],
                outputs=[image_output, image_detections]
            )
        
        with gr.TabItem("📹 Real-time Webcam"):
            gr.Markdown("### Use your webcam for real-time waste detection and categorization")
            gr.Markdown("**Instructions:** Click 'Start Webcam' to activate your camera, then click 'Detect' to analyze the current frame.")
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Webcam Feed")
                    webcam_button = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    webcam_output = gr.Image(type="numpy", label="Detection Result")
                    webcam_detections = gr.Markdown(label="Analysis Results")
            
            def detect_webcam_wrapper(image, language):
                return detect_webcam(image, language_map.get(language, "en"))
            
            webcam_button.click(
                fn=detect_webcam_wrapper,
                inputs=[webcam_input, language_selector],
                outputs=[webcam_output, webcam_detections]
            )
        
        with gr.TabItem("🎥 Video File"):
            gr.Markdown("### Upload a video to detect and categorize waste objects frame-by-frame")
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    video_button = gr.Button("Detect Objects in Video", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Detection Result Video")
                    video_detections = gr.Markdown(label="Processing Info")
            
            def detect_video_wrapper(video, language):
                return detect_video(video, language_map.get(language, "en"))
            
            video_button.click(
                fn=detect_video_wrapper,
                inputs=[video_input, language_selector],
                outputs=[video_output, video_detections]
            )
        
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("### Waste Detection Statistics Dashboard")
            dashboard_table = gr.Dataframe(headers=["Timestamp", "Source", "Summary"], interactive=False)
            dashboard_plot = gr.Plot()
            dashboard_refresh = gr.Button("Refresh Dashboard", variant="primary")
            def update_dashboard():
                table, fig = dashboard_view()
                return table, fig
            dashboard_refresh.click(fn=update_dashboard, inputs=None, outputs=[dashboard_table, dashboard_plot])
            # Show initial dashboard
            table, fig = dashboard_view()
            dashboard_table.value = table
            dashboard_plot.value = fig
    
    gr.Markdown("---")
    gr.Markdown("### 🎯 Features")
    gr.Markdown("• **Multi-class detection:** 60+ waste types")
    gr.Markdown("• **Reusability categorization:** Reusable vs Non-reusable")
    gr.Markdown("• **Recyclability analysis:** Recyclable vs Non-recyclable")
    gr.Markdown("• **Environmental impact:** Low, Medium, High impact assessment")
    gr.Markdown("• **Trash percentage:** Calculate waste composition")
    gr.Markdown("• **Real-time analytics:** Live statistics and tracking")
    gr.Markdown("• **Multi-language support:** English, Spanish, French, German")

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
