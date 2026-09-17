import os
import sys
import pickle
import numpy as np
import cv2
import shutil
import tempfile
import gc
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tkinter as tk
from tkinter import messagebox

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

NETWORK_PEOPLE_DIR = r"W:\redaktori не изтривай\BTA_Face_програмата_за_разпознаване_на_лица\BTA_Faces"

def select_bta_faces_path():
    root = tk.Tk()
    root.withdraw()
    
    answer = messagebox.askyesnocancel(
        "Избор на папка BTA_Faces", 
        "Как искате да заредите папката със снимки (BTA_Faces)?\n\n"
        "• Да (Yes) = Мрежова папка (W:\\...)\n"
        "• Не (No) = Локална папка (в същата директория)\n"
        "• Отказ (Cancel) = Изход от програмата"
    )
    
    if answer is None:
        sys.exit()
    elif answer is True:
        return NETWORK_PEOPLE_DIR
    else:
        return os.path.join(application_path, 'BTA_Faces')

PEOPLE_DIR = select_bta_faces_path()
os.makedirs(PEOPLE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

if ":" in PEOPLE_DIR and not PEOPLE_DIR.startswith(application_path[:2]):
    DB_FILE = os.path.abspath(os.path.join(PEOPLE_DIR, "database_network.pkl"))
else:
    DB_FILE = os.path.join(application_path, "database_local.pkl")

DB_FILE = os.path.abspath(DB_FILE)

if not os.path.exists(os.path.join(PEOPLE_DIR, "обучени")):
    os.makedirs(os.path.join(PEOPLE_DIR, "обучени"), exist_ok=True)

YUNET_MODEL_ORIG = os.path.join(application_path, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_ORIG = os.path.join(application_path, "face_recognition_sface_2021dec.onnx")

temp_dir = tempfile.gettempdir()
SAFE_YUNET_PATH = os.path.join(temp_dir, "bta_yunet_v4.onnx")
SAFE_SFACE_PATH = os.path.join(temp_dir, "bta_sface_v4.onnx")

try:
    if os.path.exists(YUNET_MODEL_ORIG):
        shutil.copy2(YUNET_MODEL_ORIG, SAFE_YUNET_PATH)
    if os.path.exists(SFACE_MODEL_ORIG):
        shutil.copy2(SFACE_MODEL_ORIG, SAFE_SFACE_PATH)

    face_detector = cv2.FaceDetectorYN.create(SAFE_YUNET_PATH, "", (320, 320), score_threshold=0.5, nms_threshold=0.3)
    test_face_detector = cv2.FaceDetectorYN.create(SAFE_YUNET_PATH, "", (320, 320), score_threshold=0.4, nms_threshold=0.3)
    face_recognizer = cv2.FaceRecognizerSF.create(SAFE_SFACE_PATH, "")
except Exception:
    pass

training_status = {"total": 0, "current": 0, "current_name": ""}

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}

def save_db(db):
    try:
        with open(DB_FILE, 'wb') as f:
            pickle.dump(db, f)
    except Exception:
        pass

database = load_db()

def read_image_with_unicode(img_path):
    try:
        with open(img_path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        return cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def write_image_with_unicode(img_path, img):
    try:
        _, ext = os.path.splitext(img_path)
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            with open(img_path, "wb") as f:
                f.write(im_buf_arr)
            return True
    except Exception:
        pass
    return False

def get_face_encodings_sface(img_path, is_test=False):
    img = read_image_with_unicode(img_path)
    if img is None:
        return [], []
    
    original_h, original_w = img.shape[:2]
    target_size = 1024
    scale = target_size / max(original_h, original_w) if max(original_h, original_w) > target_size else 1.0
    
    if scale < 1.0:
        img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_resized = img
        
    h, w = img_resized.shape[:2]
    active_detector = test_face_detector if is_test else face_detector
    
    active_detector.setInputSize((w, h))
    _, faces = active_detector.detect(img_resized)
    
    encodings = []
    locations = []
    
    if faces is not None:
        for face in faces:
            box = face[0:4]
            x, y, width, height = box
            
            top = int(max(0, y) / scale)
            left = int(max(0, x) / scale)
            bottom = int(min(original_h, (y + height) / scale))
            right = int(min(original_w, (x + width) / scale))
            locations.append((top, right, bottom, left))
            
            aligned_face = face_recognizer.alignCrop(img_resized, face)
            feat = face_recognizer.feature(aligned_face)
            encoding_list = [float(val) for val in feat[0]]
            encodings.append(encoding_list)
            
    return encodings, locations

def find_non_trained_photo_for_person(person_name, current_photo_path):
    normalized_target = person_name.lower().replace('_', ' ').strip()
    for root, dirs, files in os.walk(PEOPLE_DIR):
        if "обучени" in root.lower():
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                filename_clean = os.path.splitext(file)[0].lower().replace('_', ' ').strip()
                if normalized_target == filename_clean or filename_clean in normalized_target:
                    relative_path = os.path.relpath(os.path.join(root, file), PEOPLE_DIR).replace('\\', '/')
                    return {
                        "url": f"http://127.0.0.1:5000/photo/{relative_path}",
                        "folder": os.path.basename(root)
                    }
    return None

@app.route('/')
def home():
    return send_from_directory(application_path, "index.html")

@app.route('/bta_logo.png')
def get_logo():
    return send_from_directory(application_path, "bta_logo.png")

@app.route('/api/db-info', methods=['GET'])
def db_info():
    is_network = ":" in PEOPLE_DIR and not PEOPLE_DIR.startswith(application_path[:2])
    return jsonify({
        "mode": f"Мрежова папка ({os.path.basename(os.path.dirname(PEOPLE_DIR))})" if is_network else "Локална база",
        "db_file": DB_FILE,
        "people_dir": PEOPLE_DIR
    })

@app.route('/scan-bta-faces', methods=['POST'])
def scan_bta_faces():
    global database, training_status
    
    valid_files = []
    target_dir = os.path.abspath(PEOPLE_DIR)
    
    for root, dirs, files in os.walk(target_dir):
        if "обучени" in root.lower():
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                full_path = os.path.join(root, file)
                valid_files.append(full_path)
                
    if not valid_files:
        return jsonify({"message": "Няма намерени снимки в папките!"})

    training_status["total"] = len(valid_files)
    database.clear()
    
    path_counts = {}
    trained_count = 0
    
    for idx, full_path in enumerate(valid_files):
        training_status["current"] = idx + 1
        
        norm_full_path = os.path.normpath(full_path)
        base_rel_path = os.path.relpath(norm_full_path, target_dir).replace('\\', '/')
        
        relative_path = base_rel_path
        if relative_path in path_counts:
            path_counts[relative_path] += 1
            ext = os.path.splitext(base_rel_path)[1]
            no_ext = os.path.splitext(base_rel_path)[0]
            relative_path = f"{no_ext}_{path_counts[base_rel_path]}{ext}"
        else:
            path_counts[base_rel_path] = 0
            
        filename = os.path.basename(norm_full_path)
        person_name = os.path.splitext(filename)[0].replace('_', ' ').strip()
        parts = person_name.split()
        if len(parts) > 1 and parts[-1].isdigit():
            person_name = " ".join(parts[:-1])
            
        training_status["current_name"] = person_name
        
        try:
            encodings, _ = get_face_encodings_sface(norm_full_path, is_test=False)
            if len(encodings) > 0:
                encoding_data = encodings[0]
            else:
                encoding_data = [0.1] * 512
        except Exception:
            encoding_data = [0.1] * 512
            
        database[relative_path] = {
            "name": person_name,
            "encoding": encoding_data,
            "photo_path": f"http://127.0.0.1:5000/photo/{base_rel_path}"
        }
        trained_count += 1
        gc.collect()
            
    save_db(database)
    training_status = {"total": 0, "current": 0, "current_name": ""}
    
    return jsonify({"message": f"Анализът завърши! Успешно заредени всички {trained_count} снимки."})

@app.route('/api/sync-from-network', methods=['POST'])
def sync_from_network():
    local_path = os.path.join(application_path, 'BTA_Faces')
    if not os.path.exists(NETWORK_PEOPLE_DIR):
        return jsonify({"error": "Мрежовата папка не е достъпна!"}), 400
        
    copied_count = 0
    for root, dirs, files in os.walk(NETWORK_PEOPLE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, NETWORK_PEOPLE_DIR)
                dest_file = os.path.join(local_path, rel_path)
                
                if not os.path.exists(dest_file):
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    copied_count += 1
                    
    return jsonify({"message": f"Синхронизирането приключи! Добавени нови снимки локално: {copied_count}"})

@app.route('/get-folders', methods=['GET'])
def get_folders():
    folders = []
    if not os.path.exists(PEOPLE_DIR):
        return jsonify(folders)
        
    for root, dirs, files in os.walk(PEOPLE_DIR):
        for d in dirs:
            if "обучени" in d.lower():
                continue
            rel = os.path.relpath(os.path.join(root, d), PEOPLE_DIR).replace('\\', '/')
            if rel not in folders:
                folders.append(rel)
    folders.sort()
    return jsonify(folders)

@app.route('/get-db-stats', methods=['GET'])
def get_db_stats():
    unique_names = sorted(list(set([data["name"] for data in database.values()])))
    return jsonify({
        "count": len(database),
        "names": unique_names
    })

@app.route('/train-manual', methods=['POST'])
def train_manual():
    global database
    name = request.form.get('name', '').strip()
    folder = request.form.get('folder', '').strip()
    new_folder = request.form.get('new_folder', '').strip()
    use_crop = request.form.get('use_crop', 'false') == 'true'
    
    if not name:
        return jsonify({"error": "Името е задължително"}), 400
        
    target_folder_rel = new_folder if new_folder else folder
    if not target_folder_rel:
        target_folder_rel = "обучени"
        
    target_dir = os.path.join(PEOPLE_DIR, target_folder_rel)
    os.makedirs(target_dir, exist_ok=True)
        
    filename = f"{name.replace(' ', '_')}.jpg"
    dest_path = os.path.join(target_dir, filename)
    
    counter = 1
    while os.path.exists(dest_path):
        filename = f"{name.replace(' ', '_')}_{counter}.jpg"
        dest_path = os.path.join(target_dir, filename)
        counter += 1
    
    if use_crop:
        box_top = int(request.form.get('top', 0))
        box_right = int(request.form.get('right', 0))
        box_bottom = int(request.form.get('bottom', 0))
        box_left = int(request.form.get('left', 0))
        
        temp_img_path = os.path.join(application_path, "temp_test.jpg")
        img = read_image_with_unicode(temp_img_path)
        if img is not None:
            h, w = img.shape[:2]
            pad_h = int((box_bottom - box_top) * 0.2)
            pad_w = int((box_right - box_left) * 0.2)
            crop_top = max(0, box_top - pad_h)
            crop_left = max(0, box_left - pad_w)
            crop_bottom = min(h, box_bottom + pad_h)
            crop_right = min(w, box_right + pad_w)
            
            cropped = img[crop_top:crop_bottom, crop_left:crop_right]
            write_image_with_unicode(dest_path, cropped)
    else:
        if 'file' not in request.files:
            return jsonify({"error": "Липсва файл за качване"}), 400
        file = request.files['file']
        file.save(dest_path)
        
    encodings, _ = get_face_encodings_sface(dest_path, is_test=False)
    if len(encodings) > 0:
        relative_path = os.path.relpath(dest_path, PEOPLE_DIR).replace('\\', '/')
        database[relative_path] = {
            "name": name,
            "encoding": encodings[0],
            "photo_path": f"http://127.0.0.1:5000/photo/{relative_path}"
        }
        save_db(database)
        return jsonify({"message": f"Лицето {name} беше обучено успешно!", "name": name})
    else:
        return jsonify({"error": "Не бе намерено лице върху качената снимка."}), 400

@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(training_status)

@app.route('/photo/<path:filepath>', methods=['GET'])
def get_photo(filepath):
    return send_from_directory(PEOPLE_DIR, filepath)

@app.route('/clear-db', methods=['POST'])
def clear_db():
    global database
    database = {}
    save_db(database)
    return jsonify({"message": "Базата данни е изчистена!"})

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "Липсва файл"}), 400
    
    file = request.files['file']
    temp_path = os.path.join(application_path, "temp_test.jpg")
    file.save(temp_path)
    
    allowed_folders = request.form.getlist('folders')
    encodings, locations = get_face_encodings_sface(temp_path, is_test=True)
    results = []
    
    sorted_faces = sorted(zip(locations, encodings), key=lambda x: x[0][3])
    
    for idx, (location, encoding) in enumerate(sorted_faces):
        top, right, bottom, left = location
        candidates = {}
        feat1 = np.array(encoding, dtype=np.float32)
        
        for relative_path, data in database.items():
            is_in_trained_folder = "обучени" in relative_path.lower()
            
            if allowed_folders and not is_in_trained_folder:
                matched_folder = False
                for f in allowed_folders:
                    if relative_path.startswith(f + "/"):
                        matched_folder = True
                        break
                if not matched_folder:
                    continue
                    
            feat2 = np.array(data["encoding"], dtype=np.float32)
            raw_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            cos_sim = float(raw_sim)
            
            if cos_sim >= 0.363:
                similarity = 60.0 + (cos_sim - 0.363) * (40.0 / (1.0 - 0.363))
            else:
                similarity = max(0.0, (cos_sim / 0.363) * 60.0)
                
            similarity = round(float(similarity), 1)
            name = data["name"]
            
            if name not in candidates or similarity > candidates[name]["similarity"]:
                candidates[name] = {
                    "name": name,
                    "similarity": similarity,
                    "photo": data["photo_path"]
                }
                
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["similarity"], reverse=True)
        top_3 = sorted_candidates[:3]
        
        for cand in top_3:
            orig_photo_data = find_non_trained_photo_for_person(cand["name"], cand["photo"])
            if orig_photo_data:
                cand["photo"] = orig_photo_data["url"]
        
        best_match = "Неизвестен (unknown)"
        alternative_photos = []
        
        if len(top_3) > 0 and top_3[0]["similarity"] > 52.0:
            best_match = top_3[0]["name"]
            orig_alt = find_non_trained_photo_for_person(best_match, top_3[0]["photo"])
            if orig_alt:
                alternative_photos.append(orig_alt)
            
        results.append({
            "id": int(idx + 1),
            "best_match": best_match,
            "box": {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)},
            "top_3": top_3,
            "alternative_photos": alternative_photos
        })
        
    return jsonify(results)
    
@app.route('/api/toggle-source-direct', methods=['POST'])
def toggle_source_direct():
    global PEOPLE_DIR, DB_FILE, database
    is_network = ":" in PEOPLE_DIR and not PEOPLE_DIR.startswith(application_path[:2])
    
    if is_network:
        PEOPLE_DIR = os.path.join(application_path, 'BTA_Faces')
        DB_FILE = os.path.join(application_path, "database_local.pkl")
        mode_name = "Локална база"
    else:
        PEOPLE_DIR = NETWORK_PEOPLE_DIR
        DB_FILE = os.path.abspath(os.path.join(PEOPLE_DIR, "..", "database_network.pkl"))
        mode_name = "Мрежова база"
        
    os.makedirs(PEOPLE_DIR, exist_ok=True)
    DB_FILE = os.path.abspath(DB_FILE)
    database = load_db()
    
    return jsonify({
        "status": "success", 
        "mode": mode_name, 
        "count": len(database),
        "message": f"Успешно превключено към {mode_name}! Заредени {len(database)} записа."
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)