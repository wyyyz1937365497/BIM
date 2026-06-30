import json, numpy as np
d = json.load(open("output/virtual_scan_h1.5m.json"))
labels = np.array(d["semantic_labels"])
u, c = np.unique(labels, return_counts=True)
names = ["wall","floor","ceiling","door","window","column","beam","stairs","furniture"]
print("Semantic labels present:", d["semantic_labels"] is not None)
print(f"Total points: {len(labels)}")
print("Label distribution:")
for i in range(len(u)):
    n = names[u[i]] if u[i] < len(names) else f"class_{u[i]}"
    print(f"  {n}: {c[i]} pts ({100*c[i]/len(labels):.1f}%)")
