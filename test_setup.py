import numpy, cv2, pygame, matplotlib, scipy, shapely, networkx
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("pygame", pygame.version.ver)
for name in ["mediapipe", "ortools", "ultralytics", "filterpy"]:
    try:
        __import__(name)
        print(name, "OK")
    except Exception as e:
        print(name, "problem:", e)
print("ALL GOOD")