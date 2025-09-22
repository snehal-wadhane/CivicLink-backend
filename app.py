from fastapi import FastAPI, Body, HTTPException,Query,File, UploadFile
from database import supabase  
from typing import Optional
import math
import requests
import postgrest
from ultralytics import YOLO
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
from fastapi import Form
from  URL_Generator import upload_image_to_supabase,download_image_from_url
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------
# all functions
# ----------------------
@app.get("/")
def root():
    return {"message": "CivicLink backend running!"}

def get_osm_nearby_count(lat, lon, amenity_type, radius=500):
    """
    Fetch count of nearby OSM amenities using Overpass API
    :param lat: Latitude
    :param lon: Longitude
    :param amenity_type: 'hospital', 'school', 'bank', etc.
    :param radius: in meters
    :return: total count of nearby places
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
      way["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
      relation["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
    );
    out count;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Count number of elements returned
    count = 0
    if "elements" in data:
        count = sum(1 for el in data["elements"] if el["type"] in ["node", "way", "relation"])
    return count


# Your existing prediction function
def predict_pothole(image, model_path="pothole_yolov8_best.pt", min_conf=0.1):
    """
    Args:
        image (numpy array): OpenCV image.
        model_path (str): Path to the trained YOLO model.
        min_conf (float): Minimum confidence threshold to consider a detection.
    
    Returns:
        dict: {"severity": str, "confidence": float}
    """
    import numpy as np

    class_names = ["minor_pothole", "medium_pothole", "major_pothole"]
    
    model = YOLO(model_path)
    results = model(image)
    res = results[0]
    
    if len(res.boxes) == 0 or res.boxes.conf.max() < min_conf:
        return {"severity": "No pothole detected", "confidence": 0.0}
    
    cls_indices = res.boxes.cls.cpu().numpy()
    conf_scores = res.boxes.conf.cpu().numpy()
    
    valid_indices = conf_scores >= min_conf
    cls_indices = cls_indices[valid_indices]
    conf_scores = conf_scores[valid_indices]
    
    if len(cls_indices) == 0:
        return {"severity": "No pothole detected", "confidence": 0.0}
    
    cls_counts = {name: 0 for name in class_names}
    for c in cls_indices:
        cls_counts[class_names[int(c)]] += 1
    
    image_category = max(cls_counts, key=cls_counts.get)
    indices_of_category = [i for i, c in enumerate(cls_indices) if class_names[int(c)] == image_category]
    avg_confidence = float(np.mean(conf_scores[indices_of_category]))
    
    return {"severity": image_category, "confidence": avg_confidence}
@app.post("/add_problem")
async def add_problem(
    background_tasks: BackgroundTasks,
    uid: int = Form(...),
    email: str = Form(...),
    IssueType: Optional[str] = Form(None),
    Description: Optional[str] = Form(None),
    Latitude: Optional[str] = Form(None),
    Longitude: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    try:
        # Read image bytes
        img_bytes = await image.read()
        file_ext = image.filename.split(".")[-1]

        url = upload_image_to_supabase(img_bytes, file_ext)
        # Insert basic problem row in DB (fast)
        data = {
            "uid": uid,
            "email": email,
            "photo": url,
            "IssueType": IssueType,
            "Description": Description,
            "Latitude": Latitude,
            "Longitude": Longitude,
            "status": "Issue created",
            "severity_status": None,
            "overall_severity": None,
        }
        response = supabase.table("problems").insert(data).execute()
        pid = response.data[0]["pid"]  # Assuming table has `id` primary key

        # Add background task for heavy processing
        background_tasks.add_task(process_problem, pid, img_bytes, Latitude, Longitude)

        return {"status": "success", "pid": pid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_problem(pid: int, img_bytes: bytes, Latitude: str, Longitude: str):
    # Run YOLO model
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = predict_pothole(img)

    # Get OSM amenities
    nearby_counts = {}
    if Latitude and Longitude:
        lat, lon = float(Latitude), float(Longitude)
        for amenity in ["hospital", "school", "bank", "university"]:
            nearby_counts[amenity] = get_osm_nearby_count(lat, lon, amenity, radius=500)

    # Calculate severity score
    pothole_weights = {"minor_pothole": 1, "medium_pothole": 2, "major_pothole": 3}
    amenity_weights = {"hospital": 3, "school": 2, "bank": 1, "university": 2}
    pothole_factor = pothole_weights.get(result["severity"], 0) * result["confidence"]
    amenity_factor = sum(nearby_counts[a] * w for a, w in amenity_weights.items()) / 10
    final_score = pothole_factor + amenity_factor
    overall_severity = (
        "Low" if final_score < 1.5 else "Moderate" if final_score < 3 else "High"
    )

    # Update Supabase record
    supabase.table("problems").update({
        "severity_status": final_score,
        "overall_severity": overall_severity,
    }).eq("pid", pid).execute()

# ----------------------
# Example route: Fetch all users
# ----------------------   
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c 

@app.get("/fetch_by_location/")
async def get_users(
    longitude: float = Query(..., description="Longitude of current location"),
    latitude: float = Query(..., description="Latitude of current location")
):
    # Fetch all problems
    response = supabase.table("problems").select("*").execute()
    all_data = response.data

    nearby = []
    for item in all_data:
        try:
            item_lat = float(item.get("Latitude", 0))
            item_lon = float(item.get("Longitude", 0))
            distance = haversine(latitude, longitude, item_lat, item_lon)
            if distance <= 0.5:  # within 0.5 km
                nearby.append(item)
        except Exception:
            continue
    return {"status": "success", "data": nearby}

# ✅ Fetch problem by ID (primary key)

@app.get("/fetch_by_location_with_img/")
async def get_user(
    longitude: float = Query(..., description="Longitude of current location"),
    latitude: float = Query(..., description="Latitude of current location")
):
    # Fetch all problems
    response = supabase.table("problems").select("*").execute()
    all_data = response.data

    nearby = []
    for item in all_data:
        try:
            item_lat = float(item.get("Latitude", 0))
            item_lon = float(item.get("Longitude", 0))
            distance = haversine(latitude, longitude, item_lat, item_lon)
            if distance <= 0.5:  # within 0.5 km
                # Download image and convert to base64
                image_url = item.get("photo")
                image_base64 = None
                if image_url:
                    try:
                        r = requests.get(image_url, timeout=5)
                        if r.status_code == 200:
                            image_base64 = base64.b64encode(r.content).decode("utf-8")
                    except:
                        pass
                item_copy = item.copy()
                item_copy["photo_base64"] = image_base64
                nearby.append(item_copy)
        except Exception:
            continue

    return {"status": "success", "data": nearby}

@app.get("/get_image/{pid}")
def get_image(pid: int):
    # fetch problem from DB
    response = supabase.table("problems").select("photo").eq("pid", pid).execute()
    if not response.data:
        return {"error": "Problem not found"}
    image_url = response.data[0]["photo"]
    r = requests.get(image_url, stream=True)
    return StreamingResponse(r.raw, media_type="image/png")

@app.get("/fetchall")
async def get_users():
    response = supabase.table("problems").select("*").execute()
    return {"status": "success", "data": response.data}

@app.get("/by_id/{uid}")
async def get_by_id(uid: int):
    try:
        response = supabase.table("problems").select("*").eq("uid", uid).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Problem not found")
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# ✅ Update problem by ID
@app.put("/update_id/{pid}")
async def by_update_id(pid: int, updated_data: dict = Body(...)):
    try:
        response = supabase.table("problems").update(updated_data).eq("pid", pid).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Problem not found for update")
        return {"status": "success", "updated": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ✅ Delete problem by ID
@app.delete("/delete_id/{pid}")
async def by_delete_id(pid: int):
    try:
        response = supabase.table("problems").delete().eq("pid", pid).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Problem not found for delete")
        return {"status": "success", "deleted": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    
@app.post("/add_user")
async def add_user(email: str = Body(...), password: str = Body(...)):
    try:
        # Insert user into Supabase "Users" table
        response = supabase.table("users").insert({
            "email": email,
            "password": password  # ⚠️ Plaintext for now (better: hash it with bcrypt)
        }).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="User not created")

        # Assuming "uid" is the primary key column
        uid = response.data[0].get("uid")

        return {"uid": uid}

    except postgrest.exceptions.APIError as e:
        raise HTTPException(status_code=400, detail=f"Supabase API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
@app.post("/login")
async def login(email: str = Body(...), password: str = Body(...)):
    try:
        # Fetch user from Supabase by email
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user = response.data[0]
        
        # Compare password
        if user.get("password") != password:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return {"uid": user.get("uid")}

    except postgrest.exceptions.APIError as e:
        raise HTTPException(status_code=400, detail=f"Supabase API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    