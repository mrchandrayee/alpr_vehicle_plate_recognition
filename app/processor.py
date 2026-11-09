import cv2, csv, json
from collections import Counter
from pathlib import Path
from .config import CONFIG,OUTPUT_DIR
from .tracker import CentroidTracker

def process_video(path, alpr, output_name):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError("Could not open video.")
    fps=cap.get(cv2.CAP_PROP_FPS) or CONFIG["video"]["fallback_fps"]
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outpath=OUTPUT_DIR/output_name
    writer=cv2.VideoWriter(str(outpath),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    tracker=CentroidTracker(CONFIG["tracking"]["max_centroid_distance"],CONFIG["tracking"]["max_disappeared"])
    frame_no=0; events=[]; seen=Counter(); frame_step=max(1,int(CONFIG["video"]["process_every_n_frames"]))
    last={}
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_no%frame_step==0:
            detections=alpr.analyse(frame)
            boxes=[d["vehicle_box"] for d in detections]
            ids=tracker.update(boxes)
            for i,d in enumerate(detections):
                tid=ids.get(i,-1); x1,y1,x2,y2=d["vehicle_box"]
                plate=d["plate"]; conf=d["ocr_confidence"]
                color=(30,190,80) if plate!="UNREADABLE" else (20,150,230)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"ID {tid} | {plate} | {conf:.2f}"
                cv2.rectangle(frame,(x1,max(0,y1-24)),(min(width,x1+min(330,len(label)*9+12)),y1),color,-1)
                cv2.putText(frame,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                if d["plate_box"]:
                    px1,py1,px2,py2=d["plate_box"]
                    cv2.rectangle(frame,(px1,py1),(px2,py2),(255,190,20),1)
                key=(tid,plate)
                if plate!="UNREADABLE" and (key not in last or frame_no-last[key]>fps*2):
                    events.append({"time_seconds":round(frame_no/fps,2),"track_id":tid,
                                   "plate_text":plate,"ocr_confidence":round(conf,3)})
                    last[key]=frame_no
                seen[plate]+=1
        writer.write(frame); frame_no+=1
    cap.release(); writer.release()
    report={"frames_processed":frame_no,"fps":fps,"duration_seconds":round(frame_no/fps,2),
            "unique_plate_reads":len({e["plate_text"] for e in events}),
            "plate_read_counts":dict(seen),"events":events,"output":str(outpath)}
    (OUTPUT_DIR/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (OUTPUT_DIR/"plate_events.csv").open("w",newline="",encoding="utf-8") as f:
        writer_csv=csv.DictWriter(f,fieldnames=["time_seconds","track_id","plate_text","ocr_confidence"])
        writer_csv.writeheader(); writer_csv.writerows(events)
    return report
