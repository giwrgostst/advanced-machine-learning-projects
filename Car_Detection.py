#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vehicle_counter_yolo_no_roi.py

Ανίχνευση & Μέτρηση Αυτοκινήτων σε Ολόκληρο το Κάδρο με YOLOv8 (Ultralytics)

Dataset (για documentation/future fine‐tuning):
https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset

Παράδειγμα εκτέλεσης:
$ python3 vehicle_counter_yolo_no_roi.py \
    --video /home/tom/ML/archive/Sample_Video_HighQuality.mp4 \
    --output /home/tom/ML/archive/output_no_roi.mp4 \
    --threshold 0.25 \
    --imgsz 1024 \
    --classes 2
"""
import argparse
import os
import cv2
import torch
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(
        description="Count vehicles in whole frame using YOLOv8 (no ROI filter)"
    )
    p.add_argument('--video',     required=True,  help="path to input video")
    p.add_argument('--output',    required=True,  help="path to output annotated video")
    p.add_argument('--threshold', type=float, default=0.3,
                     help="confidence threshold (0–1)")
    p.add_argument('--imgsz',     type=int,   default=640,
                     help="YOLO input image size (e.g. 640, 1024)")
    p.add_argument('--classes',   nargs='+', type=int, default=[2],
                     help="COCO class IDs to count (2=car,5=bus,7=truck,3=motorbike)")
    return p.parse_args()

def configure_torch():
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARN] CUDA όχι διαθέσιμη, τρέχουμε σε CPU.")

def main():
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Δεν βρέθηκε βίντεο: {args.video}")

    configure_torch()
    # Φόρτωσε YOLOv8n (θα κατεβάσει αυτόματα το yolov8n.pt αν δεν υπάρχει)
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    frame_idx = 0
    print("[INFO] Ξεκινά η επεξεργασία…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO inference
        results = model(frame,
                        imgsz=args.imgsz,
                        conf=args.threshold,
                        device=device)[0]

        # Μέτρησε & σχεδίασε όλα τα boxes των επιλεγμένων classes
        count = 0
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(),
                            results.boxes.cls.cpu().numpy().astype(int)):
            if cls not in args.classes:
                continue
            count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Σχεδίασε αριθμό αυτοκινήτων
        text = f"Frame {frame_idx}: Count = {count}"
        cv2.putText(frame, text, (10, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print(text)
        out.write(frame)

    cap.release()
    out.release()
    print(f"[DONE] Αποθήκευση: {args.output}")

if __name__ == "__main__":
    main()
