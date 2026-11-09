from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
from .config import CONFIG,UPLOAD_DIR,OUTPUT_DIR
from .alpr import ALPR
from .processor import process_video

app=FastAPI(title=CONFIG["app"]["title"])
app.mount("/static",StaticFiles(directory=Path(__file__).parent/"static"),name="static")
alpr=None

@app.on_event("startup")
def startup():
    global alpr
    alpr=ALPR()

@app.get("/",response_class=HTMLResponse)
def home():
    return (Path(__file__).parent/"static"/"index.html").read_text(encoding="utf-8")

@app.post("/api/process")
async def process(video:UploadFile=File(...)):
    max_size=CONFIG["app"]["max_upload_mb"]*1024*1024
    suffix=Path(video.filename or "video.mp4").suffix or ".mp4"
    path=UPLOAD_DIR/f"{uuid.uuid4().hex}{suffix}"
    size=0
    with path.open("wb") as f:
        while chunk:=await video.read(1024*1024):
            size+=len(chunk)
            if size>max_size:
                path.unlink(missing_ok=True)
                raise HTTPException(413,"Video is larger than configured upload limit.")
            f.write(chunk)
    try:
        report=process_video(path,alpr,f"{uuid.uuid4().hex}.mp4")
        report["download_url"]="/api/output/"+Path(report["output"]).name
        report["csv_url"]="/api/report/plate_events.csv"
        return report
    except Exception as e:
        raise HTTPException(400,f"Processing failed: {e}")
    finally:
        path.unlink(missing_ok=True)

@app.get("/api/output/{filename}")
def output(filename:str):
    p=OUTPUT_DIR/Path(filename).name
    if not p.exists(): raise HTTPException(404,"Output not found")
    return FileResponse(p,media_type="video/mp4",filename=p.name)

@app.get("/api/report/plate_events.csv")
def csv_report():
    p=OUTPUT_DIR/"plate_events.csv"
    if not p.exists(): raise HTTPException(404,"No report yet")
    return FileResponse(p,media_type="text/csv",filename="plate_events.csv")
