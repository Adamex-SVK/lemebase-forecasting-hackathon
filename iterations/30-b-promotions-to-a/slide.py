from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path(__file__).resolve().parent / "results"
rows = pd.read_csv(out / "scores.csv").set_index("arm")
own, control, promo = rows.loc[["A only","A+B without promotions","A+B with promotions"], "score"]
fig = plt.figure(figsize=(13,7.3125))
def label(x,y,s,size,bold=False,color="#19374a"):
    fig.text(x,y,s,fontsize=size,color=color,weight="bold" if bold else "normal")

label(.055,.898,"Did Set B’s promotions help forecast Set A?",27,True)
label(.055,.811,"Extra variable from B",16,True,"#07796d")
label(.055,.752,"For each product and store, the share of units sold on promotion in the preceding four weeks.",15)
label(.055,.689,"A had no promotion records. Both models learned from the same A and B histories.",15)
for x,title,value in [(.055,"Without B’s promotion feature",control),(.523,"With B’s promotion feature",promo)]:
    ax=fig.add_axes([x,.39,.425,.207],facecolor="#eaf3f2")
    ax.spines[:].set_visible(False); ax.set_xticks([]); ax.set_yticks([])
    ax.text(.04,.69,title,fontsize=16,transform=ax.transAxes)
    ax.text(.04,.17,f"{value:.3f} points",fontsize=28,weight="bold",color="#076c62",transform=ax.transAxes)
delta=promo-control
direction="raised" if delta>0 else "lowered"
label(.055,.289,f"B’s promotion feature {direction} A’s score by {abs(delta):.3f} points.",21,True)
label(.055,.217,f"A trained alone scored {own:.3f}. Higher scores are better.",14)
label(.055,.114,"Full five-model blend evaluated on 599 Set A product–store pairs over 13 weeks. Only LightGBM was refitted.",12)
label(.055,.064,"Retrospective test: synthetic Set B records from 2025 were used to study transfer to Set A’s 2024 holdout.",12)
fig.savefig(out/"set-b-promotions-to-set-a.png",dpi=170)
