import base64
import io
import json
import os
from datetime import datetime
from functools import lru_cache

import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from werkzeug.exceptions import RequestEntityTooLarge

IMG_SIZE = 224
HISTORY_LIMIT = 10
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.45
MAX_FILE_SIZE_MB = 8
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
EXAMPLE_SIGNS = [
    {"title": "STOP", "family": "Restriction", "shape": "Octagon"},
    {"title": "SPEED LIMIT 40", "family": "Restriction", "shape": "Circle"},
    {"title": "GIVE WAY", "family": "Priority", "shape": "Triangle"},
    {"title": "SCHOOL AHEAD", "family": "Warning", "shape": "Triangle"},
    {"title": "NO ENTRY", "family": "Restriction", "shape": "Circle"},
    {"title": "ROUNDABOUT", "family": "Mandatory", "shape": "Circle"},
]

TOKEN_TRANSLATIONS = {
    "AHEAD": "Aage",
    "ALL": "Sabhi",
    "AND": "Aur",
    "ASCENT": "Chadhai",
    "AXLE": "Axle",
    "BARRIER": "Badhak",
    "BEND": "Mod",
    "BRIDGE": "Pul",
    "BULLOCK": "Bailgadi",
    "CATTLE": "Pashu",
    "COMPULSARY": "Anivarya",
    "CROSSING": "Par",
    "CROSS": "Chauraha",
    "CURVE": "Vakra Mod",
    "CYCLE": "Cycle",
    "DANGEROUS": "Khatarnak",
    "DESCENT": "Dhalan",
    "DIP": "Gaddha",
    "DIRECTION": "Disha",
    "ENTRY": "Pravesh",
    "FALLING": "Girne Wale",
    "FERRY": "Ferry",
    "GAP": "Gap",
    "GIVE": "De",
    "GRAVEL": "Kankad",
    "GUARDED": "Surakshit",
    "HAIR": "Teekha",
    "HANDCART": "Haathgadi",
    "HEIGHT": "Uchchai",
    "HORN": "Horn",
    "HUMP": "Speed Breaker",
    "INTERSECTION": "Junction",
    "KEEP": "Bane Raho",
    "LEFT": "Baen",
    "LENGTH": "Lambai",
    "LEVEL": "Level",
    "LIMIT": "Seema",
    "LOAD": "Load",
    "LOOSE": "Dhila",
    "MEDIAN": "Median",
    "MEN": "Kaam Chal Raha Hai",
    "MINIMUM": "Nyuntam",
    "MOTOR": "Motor",
    "NARROW": "Sankri",
    "NO": "Nahi",
    "ONCOMING": "Saamne Se Aane Wale",
    "OVERTAKING": "Overtaking",
    "PARKING": "Parking",
    "PASS": "Nikal Sakte Hain",
    "PEDESTRIAN": "Paidal Yatri",
    "PRIORITY": "Prathmikta",
    "PROHIBITED": "Mana Hai",
    "QUAY": "Kinara",
    "RESTRICTION": "Pratibandh",
    "REVERSE": "Ulta",
    "RIGHT": "Dahine",
    "RIVER": "Nadi",
    "ROAD": "Sadak",
    "ROCKS": "Patthar",
    "ROUNDABOUT": "Gol Chakkar",
    "ROUGH": "Uneven",
    "SCHOOL": "School",
    "SIDE": "Side",
    "SIGNAL": "Signal",
    "SLIPPERY": "Phisalne Wali",
    "SOUND": "Bajaiye",
    "SPEED": "Gati",
    "STAGGERED": "Shifted",
    "STANDING": "Khade Hona",
    "STEEP": "Tez",
    "STOP": "Rukiye",
    "STRAIGHT": "Seedha",
    "TONGA": "Tanga",
    "TRACK": "Track",
    "TRAFFIC": "Traffic",
    "TRUCK": "Truck",
    "TURN": "Mudna",
    "U": "U",
    "UNGUARDED": "Asurakshit",
    "VEHICLE": "Vehicle",
    "VEHICLES": "Vehicles",
    "WAY": "Rasta",
    "WIDTH": "Chaudai",
    "WORK": "Kaam",
    "Y": "Y",
}

SPECIAL_HINDI = {
    "NO ENTRY": "Pravesh Nishedh",
    "STOP": "Rukiye",
    "GIVE WAY": "Rasta Dijiye",
    "SCHOOL AHEAD": "Aage School",
    "TRAFFIC SIGNAL": "Traffic Signal",
    "PEDESTRIAN CROSSING": "Paidal Yatri Par Path",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "indian-traffic-sign-demo-secret")


@lru_cache(maxsize=1)
def load_artifacts():
    model = load_model("indian_traffic_sign_model.h5", compile=False)
    with open("class_labels.json", "r", encoding="utf-8") as file:
        raw_labels = json.load(file)
    id_to_key = {int(value): key for key, value in raw_labels.items()}
    return model, id_to_key


def beautify_label(label_key: str) -> str:
    return label_key.replace("_", " ").title()


def label_to_hindi(label_text: str) -> str:
    if label_text in SPECIAL_HINDI:
        return SPECIAL_HINDI[label_text]
    return " ".join(TOKEN_TRANSLATIONS.get(token, token.title()) for token in label_text.split())


def localize_label(label_text: str, language: str) -> str:
    if language == "hindi":
        return f"{label_text} / {label_to_hindi(label_text)}"
    return label_text


def infer_sign_family(label_key: str) -> str:
    restriction_terms = ("PROHIBITED", "NO_", "STOP", "GIVE_WAY", "LIMIT", "ENTRY", "PARKING", "STANDING")
    mandatory_terms = ("COMPULSARY", "KEEP", "PASS_EITHER_SIDE", "ROUNDABOUT", "TURN_RIGHT", "TURN_LEFT", "DIRECTION")
    warning_terms = (
        "AHEAD",
        "BEND",
        "CURVE",
        "CROSSING",
        "INTERSECTION",
        "ROAD",
        "ASCENT",
        "DESCENT",
        "DIP",
        "SIGNAL",
        "BRIDGE",
        "ROCKS",
        "GRAVEL",
        "SCHOOL",
        "WORK",
        "HUMP",
        "RIVER",
        "FERRY",
    )

    if label_key.startswith("SPEED_LIMIT_") or any(term in label_key for term in restriction_terms):
        return "Restriction"
    if any(term in label_key for term in mandatory_terms):
        return "Mandatory"
    if any(term in label_key for term in warning_terms):
        return "Warning"
    return "Information"


def infer_sign_shape(label_key: str) -> str:
    if label_key == "STOP":
        return "Octagon"
    if label_key.startswith("SPEED_LIMIT_") or "PROHIBITED" in label_key or label_key.startswith("NO_"):
        return "Circle"
    if "GIVE_WAY" in label_key or "SCHOOL_AHEAD" in label_key:
        return "Triangle"
    return "Mixed"


def describe_sign(label_key: str, label_text: str) -> str:
    if label_key.startswith("SPEED_LIMIT_"):
        speed = label_key.split("_")[-1]
        return f"Vehicles should not exceed {speed} km/h in this zone."
    if label_key == "STOP":
        return "Drivers must come to a complete halt before moving ahead."
    if label_key == "GIVE_WAY":
        return "Yield to traffic on the major road before proceeding."
    if "PROHIBITED" in label_key or label_key.startswith("NO_"):
        return f"{label_text} indicates a restricted action or restricted vehicle class."
    if "CROSSING" in label_key:
        return f"{label_text} warns that crossing activity may be present ahead."
    if "AHEAD" in label_key:
        return f"{label_text} alerts drivers to prepare early and reduce uncertainty."
    if "TURN" in label_key or "KEEP" in label_key or "ROUNDABOUT" in label_key:
        return f"{label_text} gives direction guidance that should be followed at the junction."
    return f"{label_text} is a road sign category recognized by the model."


def safety_tip(label_key: str) -> str:
    if label_key.startswith("SPEED_LIMIT_"):
        return "Match your speed to road, weather, and traffic conditions even when the limit allows more."
    if label_key in {"STOP", "GIVE_WAY", "NO_ENTRY"}:
        return "Scan both directions, watch for pedestrians, and move only when the path is clear."
    if "SCHOOL" in label_key or "PEDESTRIAN" in label_key:
        return "Expect sudden pedestrian movement and keep braking distance longer than usual."
    if "CROSSING" in label_key or "INTERSECTION" in label_key:
        return "Slow down early and be ready for merging or crossing traffic."
    if "SLIPPERY" in label_key or "GRAVEL" in label_key or "ROUGH" in label_key:
        return "Reduce speed, avoid harsh steering inputs, and leave extra stopping distance."
    if "BEND" in label_key or "CURVE" in label_key or "HAIR_PIN" in label_key:
        return "Brake before the bend and keep a stable line through the turn."
    return "Treat the sign as an early warning and adjust speed, lane position, and attention level."


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(image: Image.Image):
    model, id_to_key = load_artifacts()
    processed = preprocess_image(image)
    probabilities = model.predict(processed, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-5:][::-1]

    top_predictions = []
    for index in top_indices:
        key = id_to_key[int(index)]
        top_predictions.append(
            {
                "index": int(index),
                "key": key,
                "label": beautify_label(key),
                "score": round(float(probabilities[index]) * 100, 2),
            }
        )

    best = top_predictions[0]
    return best, top_predictions


def confidence_state(score_percent: float):
    score = score_percent / 100.0
    if score >= CONFIDENCE_HIGH:
        return "High confidence", "good"
    if score >= CONFIDENCE_MEDIUM:
        return "Medium confidence", "warn"
    return "Low confidence", "low"


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def decode_camera_image(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Camera image data is missing or malformed.")
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_result_payload(image: Image.Image, filename: str, language: str):
    result_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_prediction, top_predictions = predict_image(image)
    status_text, status_kind = confidence_state(best_prediction["score"])
    result = {
        "filename": filename,
        "preview_url": image_to_data_url(image),
        "time": result_time,
        "best": {
            "label": best_prediction["label"],
            "localized_label": localize_label(best_prediction["label"], language),
            "confidence": best_prediction["score"],
            "family": infer_sign_family(best_prediction["key"]),
            "shape": infer_sign_shape(best_prediction["key"]),
            "description": describe_sign(best_prediction["key"], best_prediction["label"]),
            "safety_tip": safety_tip(best_prediction["key"]),
            "status_text": status_text,
            "status_kind": status_kind,
        },
        "top_predictions": [
            {
                "label": item["label"],
                "localized_label": localize_label(item["label"], language),
                "score": item["score"],
            }
            for item in top_predictions
        ],
    }
    push_history(result["best"]["localized_label"], result["best"]["confidence"], filename, timestamp=result_time)
    return result


def push_history(label: str, confidence: float, source: str, timestamp: str | None = None):
    history = session.get("history", [])
    history.insert(
        0,
        {
            "label": label,
            "confidence": confidence,
            "source": source,
            "time": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    session["history"] = history[:HISTORY_LIMIT]


def page_context(**kwargs):
    return {
        "examples": EXAMPLE_SIGNS,
        "history": session.get("history", []),
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        **kwargs,
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **page_context(result=None, error=None, language="english"))


@app.route("/predict", methods=["POST"])
def predict():
    language = request.form.get("language", "english")
    uploaded_files = request.files.getlist("images")
    camera_image = request.form.get("camera_image", "").strip()

    has_uploads = any(file and file.filename != "" for file in uploaded_files)
    has_camera_image = bool(camera_image)

    if not has_uploads and not has_camera_image:
        return render_template(
            "index.html",
            **page_context(
                result=None,
                error="Please upload at least one JPG or PNG image, or capture a photo with the live camera.",
                language=language,
            ),
        )

    results = []
    errors = []

    for uploaded_file in uploaded_files:
        if not uploaded_file or uploaded_file.filename == "":
            continue
        if not allowed_file(uploaded_file.filename):
            errors.append(f"{uploaded_file.filename}: unsupported file type.")
            continue

        try:
            image = Image.open(uploaded_file.stream).convert("RGB")
            results.append(build_result_payload(image, uploaded_file.filename, language))
        except UnidentifiedImageError:
            errors.append(f"{uploaded_file.filename}: file could not be read as an image.")
        except Exception as error:
            errors.append(f"{uploaded_file.filename}: prediction failed with error: {error}")

    if has_camera_image:
        try:
            image = decode_camera_image(camera_image)
            results.append(build_result_payload(image, "Live camera capture", language))
        except UnidentifiedImageError:
            errors.append("Live camera capture: image could not be decoded.")
        except ValueError as error:
            errors.append(f"Live camera capture: {error}")
        except Exception as error:
            errors.append(f"Live camera capture: prediction failed with error: {error}")

    if not results:
        return render_template(
            "index.html",
            **page_context(result=None, error=" ".join(errors) if errors else "Prediction failed.", language=language),
        )

    return render_template(
        "index.html",
        **page_context(result=results, error=" ".join(errors) if errors else None, language=language),
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    session["history"] = []
    return redirect(url_for("home"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return (
        render_template(
            "index.html",
            **page_context(
                result=None,
                error=f"Upload too large. Please keep the total upload under {MAX_FILE_SIZE_MB} MB.",
                language="english",
            ),
        ),
        413,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), debug=False)
