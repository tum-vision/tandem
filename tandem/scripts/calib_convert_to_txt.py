import json
import sys


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Call like python calib_convert_to_txt TANDEM_CALIB_DIR"
    path = sys.argv[1]
    print("Path = ", path)

    with open(path+"/calibration.json", "r") as fp:
        d = json.load(fp)['value0']
    assert d['intrinsics'][0]['camera_type'] == 'kb4'
    intr = d['intrinsics'][0]['intrinsics']

    w_in = 1280
    w = 512

    h_in = 800
    h = 320

    scale_x = w/w_in
    scale_y = h/h_in

    fx = intr['fx']*scale_x
    fy = intr['fy']*scale_y
    cx = (intr['cx']+.5)*scale_x - .5
    cy = (intr['cy']+.5)*scale_y - .5

    ks = ' '.join(str(intr[f"k{i}"]) for i in range(1, 5))

    with open(path+"/camera.txt", "w") as fp:
        fp.write(f"EquiDistant {fx} {fy} {cx} {cy} {ks}\n")
        fp.write(f"{w} {h}\n")
        fp.write(f"crop\n")
        fp.write(f"{w} {h}\n")
