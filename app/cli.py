import argparse,json
from .alpr import ALPR
from .processor import process_video

def main():
    p=argparse.ArgumentParser(description="Local ALPR video processor")
    p.add_argument("video"); p.add_argument("--output",default="alpr_annotated.mp4")
    a=p.parse_args()
    print(json.dumps(process_video(a.video,ALPR(),a.output),indent=2))
if __name__=="__main__": main()
