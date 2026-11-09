import cv2, re, numpy as np
from ultralytics import YOLO
import easyocr
from .config import CONFIG

class ALPR:
    """Vehicle detection + plate-region proposal + OCR.

    YOLO detects vehicles. Since the supplied recording does not expose a
    dedicated plate detector checkpoint, plate proposals use image processing
    within vehicle crops; OCR reads candidate regions.
    """
    def __init__(self):
        self.detector=YOLO(CONFIG["detection"]["model"])
        self.reader=easyocr.Reader(CONFIG["ocr"]["languages"],gpu=CONFIG["ocr"]["gpu"])
        self.vehicle_classes=set(CONFIG["detection"]["vehicle_classes"])

    def detect_vehicles(self,frame):
        result=self.detector.predict(frame,conf=CONFIG["detection"]["confidence"],verbose=False)[0]
        vehicles=[]
        for box in result.boxes:
            cls=int(box.cls[0])
            if cls not in self.vehicle_classes: continue
            x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
            vehicles.append((max(0,x1),max(0,y1),min(frame.shape[1],x2),min(frame.shape[0],y2)))
        return vehicles

    def plate_candidates(self, frame, vehicle_box):
        x1,y1,x2,y2=vehicle_box
        crop=frame[y1:y2,x1:x2]
        if crop.size==0: return []
        h,w=crop.shape[:2]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        gray=cv2.bilateralFilter(gray,9,75,75)
        edges=cv2.Canny(gray,60,180)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(17,3))
        closed=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        minw=CONFIG["detection"]["plate_min_width"]
        minh=CONFIG["detection"]["plate_min_height"]
        for c in contours:
            rx,ry,rw,rh=cv2.boundingRect(c)
            ratio=rw/max(1,rh)
            if rw>=minw and rh>=minh and 2.0<=ratio<=7.5 and rw*rh < w*h*.35:
                # Restrict proposal to lower/middle vehicle area to reduce false OCR.
                if ry+rh < h*.85:
                    candidates.append((x1+rx,y1+ry,x1+rx+rw,y1+ry+rh))
        # OCR full vehicle crop as fallback; candidate boxes can be poor on low-res footage.
        if not candidates:
            candidates=[(x1,y1+int((y2-y1)*.45),x2,y2)]
        return candidates[:8]

    def read_plate(self,frame,box):
        x1,y1,x2,y2=box
        crop=frame[y1:y2,x1:x2]
        if crop.size==0: return "",0.0
        enlarged=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        gray=cv2.cvtColor(enlarged,cv2.COLOR_BGR2GRAY)
        results=self.reader.readtext(gray,detail=1,paragraph=False,
            allowlist=CONFIG["ocr"]["allowlist"])
        if not results: return "",0.0
        best=max(results,key=lambda x:float(x[2]))
        text=re.sub(r"[^A-Z0-9-]","",best[1].upper())
        return text,float(best[2])

    def analyse(self,frame):
        detections=[]
        for vehicle in self.detect_vehicles(frame):
            best_text,best_conf,best_box="",0.0,None
            for plate_box in self.plate_candidates(frame,vehicle):
                text,conf=self.read_plate(frame,plate_box)
                if text and conf>best_conf:
                    best_text,best_conf,best_box=text,conf,plate_box
            detections.append({"vehicle_box":vehicle,"plate_box":best_box,
                               "plate":best_text or "UNREADABLE","ocr_confidence":best_conf})
        return detections
