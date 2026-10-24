"""Assemble ChatScene snippets (single-lane road geometry G9 + pedestrian crossing behavior B0 + a spawn
entry) into Scenic 3 syntax, sample scenes, and measure where the adversary is relative to the ego."""
import json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); import safe_pickle
import scenic

db = safe_pickle.load(ROOT.parent / "ChatScene/retrieve/database_v1.pkl")
HEAD = ("param map = localPath('Town03.xodr')\nparam carla_map = 'Town03'\n"
        "model scenic.simulators.carla.model\nEGO_MODEL = 'vehicle.lincoln.mkz_2017'\n")

def to_scenic3(code):
    code = re.sub(r"^Town = .*\n", "", code, flags=re.M)
    code = re.sub(r"=\s*(Car|BusStop|VendingMachine|Pedestrian|OrientedPoint)\s+(at|left of|right of|following|in)", r"= new \1 \2", code)
    return code

def run(spawn_id, n=40):
    code = HEAD + db["behavior"]["snippet"][0] + "\n" + to_scenic3(db["geometry"]["snippet"][9]) + "\n" + \
        to_scenic3(db["spawn"]["snippet"][spawn_id]).replace("{AdvObject}", "new Pedestrian")
    code = code.replace("egoTrajectory", "egoLaneSec.lane")   # G9 defines no trajectory; only used by the behavior
    ref = re.search(r"^(\w+) = new OrientedPoint following", code, flags=re.M).group(1)
    code += f"\nparam refX = {ref}.position.x\nparam refY = {ref}.position.y\nparam refH = {ref}.heading\n"
    Path(f"spawn{spawn_id}.scenic").write_text(code)
    scenario = scenic.scenarioFromFile(f"spawn{spawn_id}.scenic", mode2D=True)
    lat, face = [], []
    for _ in range(n):
        scene, _ = scenario.generate(maxIterations=2000)
        adv = scene.objects[-1]
        h = scene.params["refH"]              # road direction at the snippet's reference point
        right = (math.cos(h), math.sin(h))    # unit vector pointing to the right of the road
        d = (adv.position.x - scene.params["refX"], adv.position.y - scene.params["refY"])
        lat.append(d[0] * right[0] + d[1] * right[1])           # > 0: adversary is right of the ego
        rel = (adv.heading - h + math.pi) % (2 * math.pi) - math.pi
        face.append("left" if rel > 0 else "right")        # which way the adversary faces
    return dict(spawn=spawn_id, description=db["spawn"]["description"][spawn_id].strip(),
                share_right_of_road_point=sum(x > 0 for x in lat) / n,
                lateral_m=[round(min(lat), 2), round(max(lat), 2)],
                faces_left=sum(f == "left" for f in face) / n)

if __name__ == "__main__":
    res = [run(i) for i in (1, 2, 6, 7, 8, 9, 10, 11)]
    for r in res: print(json.dumps(r))
    (ROOT / "results/execcheck.json").write_text(json.dumps(res, indent=1))
