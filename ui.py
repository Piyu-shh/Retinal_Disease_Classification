import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import re

# ==================== GLOBAL VARIABLES ====================
# We use these to store the state after a folder is selected
global_folder_path = None
global_patient_id = None
global_predictions = []

# ==================== NUMPY CONVERTER ====================
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj

# ==================== MODEL SETUP ====================
try:
    model_vgg = load_model('model_vgg.h5')
    class_labels = ['Glaucoma', 'Normal']
except Exception as e:
    messagebox.showerror("Model Error", f"Failed to load 'model_vgg.h5'. Make sure the file is in the same directory.\nError: {e}")
    exit()


def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0

def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model_vgg.predict(preprocessed_image, verbose=0)[0]
    # Assuming last two are 'Glaucoma' and 'Normal'
    last_two_predictions = predictions[-2:] 
    percentages = [p * 100 for p in last_two_predictions]
    predicted_label = class_labels[np.argmax(last_two_predictions)]
    return predicted_label, percentages


# ==================== OPENROUTER CALL ====================
def get_medical_advice(patient_info):
    # --- IMPORTANT ---
    # Never hardcode API keys in real applications. 
    # Use environment variables.
    api_key = "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" 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Replace with your app's domain if deployed
        "X-Title": "EyePredict" # Optional
    }
    
    patient_info_serializable = convert_numpy(patient_info)
    
    message = {
        "role": "user",
        "content": (
            "You are a medical assistant specialized in ophthalmology. "
            "Given the patient information below, summarize their likely eye health, "
            "possible disease progression risks, and general lifestyle or medication recommendations. "
            "Do not provide prescriptions. Be concise and clear. Use bullet points for recommendations.\n\n"
            f"Patient Info:\n{json.dumps(patient_info_serializable, indent=2)}"
        )
    }

    data = {
        "model": "alibaba/tongyi-deepresearch-30b-a3b:free", # Using a free model
        "messages": [message]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=45 # Increased timeout
        )
        response.raise_for_status() # Will raise an error for bad responses (4xx or 5xx)
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: Unexpected API response format.\n{result}"
            
    except requests.exceptions.RequestException as e:
        return f"Error fetching medical advice: {e}"
    except Exception as e:
        return f"An unknown error occurred: {e}"


# ==================== SCROLL WHEEL BINDING ====================
def _on_mouse_wheel(event, canvas):
    canvas.yview_scroll(-1 * (event.delta // 120), "units")

# ==================== PATIENT QUESTIONNAIRE ====================
def open_questionnaire():
    """
    Opens the Toplevel window for the questionnaire.
    Reads global_patient_id, global_folder_path, and global_predictions.
    """
    if not global_folder_path:
        messagebox.showerror("Error", "Please select a patient folder first.")
        return

    form = tk.Toplevel(root)
    form.title("Patient Health Questionnaire")
    form.geometry("750x900")
    form.configure(bg="#EDEDED")
    form.grab_set() # Make the questionnaire modal

    tk.Label(form, text="Comprehensive Patient Health Questionnaire",
             font=("Helvetica", 18, "bold"), bg="#007ACC", fg="white", pady=10).pack(fill=tk.X)

    canvas = tk.Canvas(form, bg="#EDEDED")
    scrollbar = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#EDEDED")
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Bind scroll wheel for questionnaire canvas
    canvas.bind_all("<MouseWheel>", lambda event: _on_mouse_wheel(event, canvas))

    # Question categories
    questions = {
        "Demographics": {
            "Age": tk.StringVar(), "Gender": tk.StringVar(), "Occupation": tk.StringVar(),
        },
        "Medical History": {
            "Has diabetes?": tk.StringVar(), "Has hypertension?": tk.StringVar(), "Has heart disease?": tk.StringVar(),
            "Has thyroid disorder?": tk.StringVar(), "Has migraine issues?": tk.StringVar(), "High cholesterol?": tk.StringVar(),
            "Autoimmune disease (e.g. lupus, rheumatoid arthritis)?": tk.StringVar(), "History of eye trauma?": tk.StringVar(),
        },
        "Ocular History": {
            "Previous eye surgery?": tk.StringVar(), "Family history of glaucoma?": tk.StringVar(), "Family history of cataract?": tk.StringVar(),
            "Recent eye infection?": tk.StringVar(), "Do you use glasses or contact lenses?": tk.StringVar(), "Dry or itchy eyes?": tk.StringVar(),
            "Double vision?": tk.StringVar(), "Floaters or flashes of light?": tk.StringVar(), "Blurred or cloudy vision?": tk.StringVar(),
            "Difficulty seeing at night?": tk.StringVar(),
        },
        "Lifestyle": {
            "Smoking habit?": tk.StringVar(), "Alcohol consumption?": tk.StringVar(), "Regular steroid medication use?": tk.StringVar(),
            "Average screen time (hours/day)": tk.StringVar(), "Average outdoor exposure (hours/day)": tk.StringVar(),
            "Use of sunglasses/UV protection outdoors?": tk.StringVar(),
        },
        "General Wellbeing": {
            "Diet quality": tk.StringVar(), "Exercise frequency": tk.StringVar(), "Sleep quality": tk.StringVar(), "Stress level": tk.StringVar(),
        }
    }
    
    # Set default values if patient_info.json exists
    json_path = os.path.join(global_folder_path, "patient_info.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            saved_answers = saved_data.get("Questionnaire", {})
            for section, q_dict in questions.items():
                for q, var in q_dict.items():
                    if section in saved_answers and q in saved_answers[section]:
                        var.set(saved_answers[section][q])
        except Exception as e:
            messagebox.showwarning("Load Error", f"Could not load previous answers: {e}", parent=form)
            
    # Set other defaults if no saved data or specific fields are empty
    for section in ["Medical History", "Ocular History", "Lifestyle"]:
        for q, var in questions[section].items():
            if "hours" not in q and "Occupation" not in q and not var.get():
                 var.set("No")
                 
    if not questions["Demographics"]["Gender"].get(): questions["Demographics"]["Gender"].set("Male")
    if not questions["General Wellbeing"]["Diet quality"].get(): questions["General Wellbeing"]["Diet quality"].set("Balanced")
    if not questions["General Wellbeing"]["Exercise frequency"].get(): questions["General Wellbeing"]["Exercise frequency"].set("Occasionally")
    if not questions["General Wellbeing"]["Sleep quality"].get(): questions["General Wellbeing"]["Sleep quality"].set("Average")
    if not questions["General Wellbeing"]["Stress level"].get(): questions["General Wellbeing"]["Stress level"].set("Moderate")


    # Helper to add dropdowns / entries
    def add_question_section(title, section_dict):
        tk.Label(scroll_frame, text=title, font=("Helvetica", 15, "bold"),
                 bg="#007ACC", fg="white", pady=5).pack(fill=tk.X, pady=(10,5))
        for q, var in section_dict.items():
            frame = tk.Frame(scroll_frame, bg="#EDEDED", pady=5)
            frame.pack(fill=tk.X, padx=15)
            tk.Label(frame, text=q, font=("Helvetica", 12), bg="#EDEDED", anchor="w", width=45).pack(side=tk.LEFT)
            if "Age" in q or "hours" in q or "Occupation" in q:
                tk.Entry(frame, textvariable=var, font=("Helvetica", 12), width=15).pack(side=tk.LEFT, padx=5)
            elif "Diet quality" in q:
                tk.OptionMenu(frame, var, "Balanced", "Irregular", "High sugar", "High fat").pack(side=tk.LEFT, padx=5)
            elif "Exercise frequency" in q:
                tk.OptionMenu(frame, var, "Daily", "3-5 times/week", "Occasionally", "Rarely").pack(side=tk.LEFT, padx=5)
            elif "Sleep quality" in q:
                tk.OptionMenu(frame, var, "Good", "Average", "Poor").pack(side=tk.LEFT, padx=5)
            elif "Stress level" in q:
                tk.OptionMenu(frame, var, "Low", "Moderate", "High").pack(side=tk.LEFT, padx=5)
            elif "Gender" in q:
                tk.OptionMenu(frame, var, "Male", "Female", "Other").pack(side=tk.LEFT, padx=5)
            else:
                tk.OptionMenu(frame, var, "Yes", "No").pack(side=tk.LEFT, padx=5)

    for section, q_dict in questions.items():
        add_question_section(section, q_dict)

    def submit_info():
        """Collect answers, call OpenRouter, save JSON, and refresh main window."""
        answers = {section: {q: v.get() for q, v in q_dict.items()} for section, q_dict in questions.items()}

        patient_info = {
            "Patient_ID": global_patient_id,
            "Predictions": global_predictions,
            "Questionnaire": answers
        }
        
        # Show loading message
        form.title("Submitting... Please wait...")
        # Use a temporary label instead of messagebox for less interruption if form remains
        status_label = tk.Label(form, text="Fetching AI advice... This may take up to 45 seconds.",
                                font=("Helvetica", 12, "italic"), fg="blue", bg="#EDEDED")
        status_label.pack(pady=5)
        form.update_idletasks() # Refresh GUI to show status_label

        advice = get_medical_advice(patient_info)
        
        # Add AI advice to the dict before saving
        patient_info["AI_Advice"] = advice

        json_path = os.path.join(global_folder_path, "patient_info.json")
        try:
            with open(json_path, "w") as f:
                patient_info_serializable = convert_numpy(patient_info)
                json.dump(patient_info_serializable, f, indent=2)
            
            messagebox.showinfo("Saved", f"Information saved at:\n{json_path}", parent=form)
            form.destroy()
            
            # Refresh the main window to show the new data
            populate_main_display() 
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save JSON file: {e}", parent=form)
        finally:
            status_label.destroy() # Remove status message
            form.title("Patient Health Questionnaire") # Reset title


    tk.Button(scroll_frame, text="Submit Information",
              font=("Helvetica", 14), bg="#007ACC", fg="white", command=submit_info).pack(pady=20)
    
    # On close, ungrab
    form.protocol("WM_DELETE_WINDOW", lambda: [form.grab_release(), form.destroy()])


# ==================== MAIN DISPLAY LOGIC ====================
def select_and_process_folder():
    """
    Asks user to select a patient folder, runs predictions, 
    and populates the main display.
    """
    global global_folder_path, global_patient_id, global_predictions
    
    folder_path = filedialog.askdirectory(title="Select Patient Folder")
    if not folder_path:
        return

    global_folder_path = folder_path
    global_patient_id = os.path.basename(folder_path)
    global_predictions = [] # Clear previous predictions

    images = []
    try:
        for file_name in os.listdir(global_folder_path):
            file_path = os.path.join(global_folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                label, percentages = predict_image(file_path)
                global_predictions.append({"image": file_name, "label": label, "confidence": percentages})
                images.append(file_path)
                
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Failed to predict images: {e}")
        return

    if len(images) == 0:
        messagebox.showerror("Error", "No images found in the selected folder.")
        return
    if len(images) != 2:
        messagebox.showwarning("Warning", f"Found {len(images)} images. Expected 2 (Left & Right eyes). Displaying all found.")

    # Sort images to try and get Left/Right consistently (basic-level sort)
    global_predictions.sort(key=lambda x: x['image'])
    images.sort()

    # Populate the main display with all info
    populate_main_display()


def format_ai_text(text):
    """Removes common markdown formatting and cleans up text."""
    text = text.replace('**', '') # Remove bold
    text = text.replace('##', '') # Remove H2 headers
    text = text.replace('#', '')  # Remove H1 headers
    text = text.replace('* ', '• ') # Replace markdown list with bullet point
    text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE) # Replace other list formats
    return text.strip()


def populate_main_display():
    """
    Clears and redraws the entire 'content_frame' (the scrollable area).
    Shows predictions, 'Take Questionnaire' button, and (if available)
    questionnaire results and AI advice from the JSON file.
    """
    # Clear the existing content frame
    for widget in content_frame.winfo_children():
        widget.destroy()
        
    if not global_patient_id:
        tk.Label(content_frame, text="Please select a patient folder to begin.",
                 font=("Helvetica", 16), bg="#EDEDED").pack(pady=50)
        return

    # === Main Two-Column Layout ===
    main_columns_frame = tk.Frame(content_frame, bg="#EDEDED")
    main_columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left Column for Predictions
    left_col_frame = tk.Frame(main_columns_frame, bg="#EDEDED")
    left_col_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))

    # Right Column for Patient Info & AI Advice
    right_col_frame = tk.Frame(main_columns_frame, bg="#EDEDED")
    right_col_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10,0))


    # --- Left Column: Predictions ---
# --- Left Column: Predictions ---
    tk.Label(left_col_frame, text=f"Patient ID: {global_patient_id}",
             font=("Helvetica", 18, "bold"), bg="#EDEDED", fg="#4A4A4A").pack(pady=10, fill=tk.X)

    # Create a container frame for the horizontal predictions
    pred_container = tk.Frame(left_col_frame, bg="#EDEDED")
    pred_container.pack(fill=tk.X, pady=10) # This container holds the two eye frames

    for i, pred in enumerate(global_predictions):
        image_path = os.path.join(global_folder_path, pred['image'])
        try:
            img = Image.open(image_path)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        eye_side = f"Eye {i+1} ({pred['image']})"

        # Create the frame INSIDE the pred_container
        frame = tk.Frame(pred_container, bg="#FFFFFF", relief=tk.RAISED, bd=2, padx=15, pady=15)
        # Pack this frame to the LEFT, making it horizontal
        frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=5) 
        
        tk.Label(frame, text=eye_side, font=("Helvetica", 14, "bold"),
                 fg="#007ACC", bg="#FFFFFF").pack(pady=5)
        
        img_label = tk.Label(frame, image=img_tk, bg="#FFFFFF")
        img_label.pack(pady=5)
        img_label.image = img_tk # Keep a reference!
        
        tk.Label(frame, text=f"Prediction: {pred['label']}",
                 font=("Helvetica", 12, "bold" if pred['label'] == 'Glaucoma' else 'normal'), 
                 fg="red" if pred['label'] == 'Glaucoma' else 'green',
                 bg="#FFFFFF").pack(pady=2)
                 
        tk.Label(frame, text=f"Confidence (G/N): {pred['confidence'][0]:.2f}% / {pred['confidence'][1]:.2f}%",
                 font=("Helvetica", 12), bg="#FFFFFF").pack(pady=2)
    # --- Right Column: Questionnaire & AI Advice ---
    
    # Questionnaire Button
    tk.Button(right_col_frame, text="Take / Retake Questionnaire",
              font=("Helvetica", 14), bg="#007ACC", fg="white",
              command=open_questionnaire).pack(pady=20, fill=tk.X, padx=10)

    json_path = os.path.join(global_folder_path, "patient_info.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # --- Display Questionnaire Answers ---
            q_frame = tk.Frame(right_col_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
            q_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(q_frame, text="Saved Questionnaire Answers", font=("Helvetica", 16, "bold"),
                     bg="#4A4A4A", fg="white", pady=5).pack(fill=tk.X)
            
            q_text = tk.Text(q_frame, wrap=tk.WORD, font=("Helvetica", 11), height=15, bg="#FAFAFA")
            q_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            for section, answers in data.get("Questionnaire", {}).items():
                q_text.insert(tk.END, f"--- {section} ---\n", ("bold_underline",))
                for q, a in answers.items():
                    if a: # Only show filled answers
                        q_text.insert(tk.END, f"{q}: \t{a}\n")
                q_text.insert(tk.END, "\n")
            
            q_text.tag_config("bold_underline", font=("Helvetica", 12, "bold", "underline"))
            q_text.config(state=tk.DISABLED)

            # --- Display AI Advice ---
            ai_frame = tk.Frame(right_col_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
            ai_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(ai_frame, text="AI-Generated Summary & Recommendations", font=("Helvetica", 16, "bold"),
                     bg="#007ACC", fg="white", pady=5).pack(fill=tk.X)
            
            ai_text_content = data.get("AI_Advice", "No AI advice found in saved file.")
            ai_text_content = format_ai_text(ai_text_content) # Format the text
            
            ai_text_widget = tk.Text(ai_frame, wrap=tk.WORD, font=("Helvetica", 12), height=15, bg="#FDFDFD")
            ai_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            ai_text_widget.insert(tk.END, ai_text_content)
            ai_text_widget.config(state=tk.DISABLED)

        except Exception as e:
            tk.Label(right_col_frame, text=f"Error reading patient_info.json: {e}",
                     font=("Helvetica", 12), bg="#EDEDED", fg="red").pack(pady=10)
    else:
        tk.Label(right_col_frame, text="No questionnaire data found for this patient.",
                 font=("Helvetica", 12), bg="#EDEDED", fg="#555").pack(pady=10)


# ==================== TKINTER GUI SETUP ====================
root = tk.Tk()
root.title("Patient Eye Prediction Dashboard")
root.geometry("1200x800") # Wider window for two columns
root.configure(bg="#EDEDED")

# --- Title Bar ---
tk.Label(root, text="Patient Eye Prediction System",
         font=("Helvetica", 20, "bold"), bg="#007ACC", fg="white", pady=10).pack(fill=tk.X)

# --- Input Frame ---
input_frame = tk.Frame(root, bg="#EDEDED", pady=20)
input_frame.pack(fill=tk.X)

tk.Button(input_frame, text="Select Patient Folder & Process", font=("Helvetica", 14),
          bg="#007ACC", fg="white", command=select_and_process_folder).pack(pady=10)

# --- Scrollable Result Frame ---
# Create a Canvas with a Scrollbar
main_canvas = tk.Canvas(root, bg="#EDEDED")
scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
main_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# This frame will contain all the content and be scrolled by the canvas
content_frame = tk.Frame(main_canvas, bg="#EDEDED")
main_canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Update scrollregion when content_frame size changes
def on_frame_configure(event):
    # This is important to ensure the scrollbar adjusts to content size
    main_canvas.update_idletasks() # Ensure all widgets are rendered
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

content_frame.bind("<Configure>", on_frame_configure)

# Bind scroll wheel for main canvas
main_canvas.bind_all("<MouseWheel>", lambda event: _on_mouse_wheel(event, main_canvas))

# Initial population of the display
populate_main_display()

root.mainloop()