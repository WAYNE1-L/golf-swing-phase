import pandas as pd

def detect_phase(row):
    try:
        lw_y = row['y15']  # LEFT_WRIST
        ls_y = row['y11']  # LEFT_SHOULDER
        le_x, le_y = row['x13'], row['y13']
        lw_x, lw_y2 = row['x15'], row['y15']
        ls_x, ls_y2 = row['x11'], row['y11']
        re_y = row['y14']
        rs_y = row['y12']
        
        # 计算角度
        import numpy as np
        def angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        elbow_angle = angle([ls_x, ls_y2], [le_x, le_y], [lw_x, lw_y2])
        
        if lw_y > ls_y + 50 and elbow_angle < 70:
            return "Preparation"
        elif lw_y < ls_y - 50 and elbow_angle > 90:
            return "Backswing"
        elif elbow_angle > 110:
            return "Downswing"
        else:
            return "Impact / Follow-through"
    except:
        return "Unknown"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if 'phase' not in df.columns:
        df['phase'] = df.apply(detect_phase, axis=1)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved: {args.output}")
