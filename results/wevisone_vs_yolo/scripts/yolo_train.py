"""Train an Ultralytics YOLO-seg model (COCO-pretrained) on an exported dataset."""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26m-seg.pt")
    ap.add_argument("--out", required=True, help="run directory (project/name)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()
    from ultralytics import YOLO

    out = Path(args.out).resolve()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                batch=args.batch, seed=args.seed, deterministic=True, project=str(out.parent), name=out.name,
                exist_ok=True, workers=4, fraction=args.fraction, plots=False, verbose=False)
    print("best:", out / "weights" / "best.pt", flush=True)


if __name__ == "__main__":
    main()
