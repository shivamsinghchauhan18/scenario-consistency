import math, scenic
scene, _ = scenic.scenarioFromFile("probe.scenic").generate(maxIterations=1)
names = ["ego", "P", "leftNeg", "leftPos", "rightPos", "plus90", "minus90"]
for name, o in zip(["ego", "leftNeg", "leftPos", "rightPos", "plus90", "minus90"], scene.objects):
    fwd = o.orientation.yaw
    print(f"{name:8s} pos=({o.position.x:6.2f},{o.position.y:6.2f})  faces=({-math.sin(fwd):5.2f},{math.cos(fwd):5.2f})")
