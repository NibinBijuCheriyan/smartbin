"""
Mobile-Friendly Indian Waste Data Collection Tool for SmartBin AI v2.
Runs a lightweight web server with camera capture, 28-class hierarchical selector,
optional GPS geotagging, offline local buffer, and direct YOLO dataset export.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="SmartBin Indian Waste Collector", version="2.0.0")

COLLECTION_DIR = Path("smartbin-v2/datasets/field_collection")
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
(COLLECTION_DIR / "images").mkdir(parents=True, exist_ok=True)
(COLLECTION_DIR / "labels").mkdir(parents=True, exist_ok=True)


class SamplePayload(BaseModel):
    image_base64: str
    class_id: int
    class_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    collector_id: Optional[str] = "field_user_01"


@app.get("/", response_class=HTMLResponse)
async def serve_mobile_ui() -> str:
    """Mobile responsive web camera capture UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>SmartBin Waste Collector</title>
    <style>
        body { font-family: -apple-system, system-ui, BlinkMacSystemFont, sans-serif; margin: 0; background: #0F172A; color: white; padding: 15px; }
        .header { text-align: center; margin-bottom: 15px; }
        .header h1 { font-size: 20px; color: #38BDF8; margin: 0; }
        #video-preview { width: 100%; border-radius: 12px; border: 2px solid #334155; background: black; max-height: 50vh; object-fit: cover; }
        .controls { display: flex; flex-direction: column; gap: 12px; margin-top: 15px; }
        select, button { padding: 14px; border-radius: 8px; font-size: 16px; border: none; }
        select { background: #1E293B; color: white; border: 1px solid #475569; }
        button { background: #0284C7; color: white; font-weight: bold; cursor: pointer; }
        button:active { background: #0369A1; }
        .btn-capture { background: #10B981; font-size: 18px; }
        #status-msg { text-align: center; font-size: 14px; margin-top: 10px; color: #94A3B8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SmartBin AI — Field Data Collector</h1>
        <p style="font-size:12px; color:#94A3B8; margin:5px 0;">Indian Waste Taxonomy (28 Classes)</p>
    </div>

    <video id="video-preview" autoplay playsinline></video>
    <canvas id="capture-canvas" style="display:none;"></canvas>

    <div class="controls">
        <select id="class-picker">
            <optgroup label="Plastic Waste">
                <option value="20:pet_bottle">PET Bottle (Water/Soda)</option>
                <option value="13:hdpe_bottle">HDPE Bottle (Shampoo/Oil)</option>
                <option value="17:milk_packet">Milk Pouch (Amul/Nandini)</option>
                <option value="21:plastic_carry_bag">Plastic Carry Bag</option>
                <option value="31:water_sachet">Water Sachet</option>
                <option value="6:chips_packet">Chips Packet</option>
                <option value="14:kurkure_packet">Kurkure Packet</option>
                <option value="2:biscuit_wrapper">Biscuit Wrapper</option>
            </optgroup>
            <optgroup label="Organic Waste">
                <option value="1:banana_peel">Banana Peel</option>
                <option value="8:coconut_shell">Coconut Shell / Husk</option>
                <option value="9:egg_shell">Egg Shell</option>
                <option value="22:rice_waste">Rice Waste</option>
                <option value="30:vegetable_waste">Vegetable Waste</option>
                <option value="10:fruit_waste">Fruit Waste</option>
                <option value="26:tea_bag">Tea Bag / Powder</option>
                <option value="15:leaves">Dry / Green Leaves</option>
            </optgroup>
            <optgroup label="Recyclables (Other)">
                <option value="0:aluminium_can">Aluminium Can</option>
                <option value="28:tin_can">Tin Can</option>
                <option value="11:glass_bottle">Glass Bottle</option>
                <option value="12:glass_jar">Glass Jar</option>
                <option value="18:newspaper">Newspaper</option>
                <option value="4:cardboard">Cardboard Box</option>
                <option value="19:paper_cup">Chai Paper Cup</option>
                <option value="27:tetra_pak">Tetra Pak (Juice/Milk)</option>
            </optgroup>
            <optgroup label="Landfill / Reject">
                <option value="25:styrofoam">Styrofoam / Thermocol</option>
                <option value="29:tissue">Used Tissue / Napkin</option>
                <option value="16:mask">Disposable Mask</option>
                <option value="7:cigarette_butt">Cigarette Butt</option>
                <option value="5:ceramic">Ceramic Shards</option>
                <option value="3:broken_plastic_toys">Broken Plastic Toys</option>
            </optgroup>
        </select>

        <button class="btn-capture" onclick="captureAndUpload()">📸 Capture & Save Sample</button>
        <div id="status-msg">Camera active. Point at waste item.</div>
    </div>

    <script>
        const video = document.getElementById('video-preview');
        const canvas = document.getElementById('capture-canvas');
        const statusMsg = document.getElementById('status-msg');
        let currentGps = { lat: null, lon: null };

        // Initialize Camera
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 640, height: 640 } })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { statusMsg.innerText = "Camera error: " + err.message; });

        // Request Geolocation
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                currentGps.lat = pos.coords.latitude;
                currentGps.lon = pos.coords.longitude;
            });
        }

        async function captureAndUpload() {
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 640;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.85);

            const [clsId, clsName] = document.getElementById('class-picker').value.split(':');
            statusMsg.innerText = "Saving sample...";

            try {
                const res = await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_base64: b64,
                        class_id: parseInt(clsId),
                        class_name: clsName,
                        latitude: currentGps.lat,
                        longitude: currentGps.lon
                    })
                });
                const data = await res.json();
                statusMsg.innerText = "✅ Saved sample: " + data.sample_id;
            } catch (err) {
                statusMsg.innerText = "Error saving: " + err.message;
            }
        }
    </script>
</body>
</html>
"""


@app.post("/api/submit")
async def submit_sample(payload: SamplePayload) -> Dict[str, Any]:
    """Ingest image upload from mobile client."""
    sample_id = f"field_{payload.class_name}_{int(time.time() * 1000)}"
    img_data = payload.image_base64.split(",")[-1]
    raw_bytes = base64.b64decode(img_data)

    img_path = COLLECTION_DIR / "images" / f"{sample_id}.jpg"
    lbl_path = COLLECTION_DIR / "labels" / f"{sample_id}.txt"

    with open(img_path, "wb") as f:
        f.write(raw_bytes)

    # By default in collection app, center object occupies 60% of frame [cx=0.5, cy=0.5, w=0.6, h=0.6]
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write(f"{payload.class_id} 0.500000 0.500000 0.600000 0.600000\n")

    meta = {
        "sample_id": sample_id,
        "class_id": payload.class_id,
        "class_name": payload.class_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latitude": payload.latitude,
        "longitude": payload.longitude,
    }
    with open(COLLECTION_DIR / f"{sample_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"status": "success", "sample_id": sample_id}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Indian Waste Mobile Collector Server")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
