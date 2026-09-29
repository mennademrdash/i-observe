from pathlib import Path
import cv2

def extract_clip(video,start,end,output):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_MSEC,max(0,start)*1000); ok,frame=cap.read()
    if not ok: raise RuntimeError('Cannot seek/decode evidence clip')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
    t=start
    while ok and t<=end:
        writer.write(frame); ok,frame=cap.read(); t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    writer.release(); cap.release(); return str(Path(output).resolve())
