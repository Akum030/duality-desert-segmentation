import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

BG="#0D1117"; ACCENT="#2EA043"; ACCENT2="#58A6FF"; WHITE="#F0F6FC"
GREY="#8B949E"; CARD="#161B22"; WARN="#F78166"

def slide(title=None, sub=None):
    fig=plt.figure(figsize=(16,9)); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off"); ax.set_facecolor(BG)
    ax.add_patch(FancyBboxPatch((0,8.55),16,0.45,boxstyle="square,pad=0",fc=ACCENT,ec="none"))
    ax.text(0.25,8.77,"AIDHUNIK · Hack For Green Bharat · Feb 26 2026",color=BG,fontsize=7.5,va="center",fontweight="bold")
    ax.text(15.75,8.77,"team aidhunik",color=BG,fontsize=7.5,va="center",ha="right")
    if title: ax.text(8,8.15,title,color=WHITE,fontsize=18,ha="center",va="center",fontweight="bold")
    if sub:   ax.text(8,7.68,sub,color=GREY,fontsize=10.5,ha="center",va="center")
    return fig,ax

def box(ax,x,y,w,h,fc=CARD,ec=ACCENT,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.10",fc=fc,ec=ec,lw=lw))

def sv(pdf,fig):
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)

iou=[0.2842,0.3003,0.3319,0.3426,0.3550,0.3667,0.3723,0.3789,0.3867,0.3830,0.3850,0.3700,0.3780,0.3850,0.3903]
acc=[0.7120,0.7340,0.7580,0.7740,0.7820,0.7940,0.8010,0.8060,0.8100,0.8090,0.8140,0.8050,0.8080,0.8110,0.8120]
ep=list(range(1,16))

with PdfPages("/home3/indiamart/gbht/aidhunik_pitch_deck.pdf") as pdf:

    ### SLIDE 1 – COVER
    fig,ax=slide()
    box(ax,1,2.6,14,5.2,ec=ACCENT,lw=2.5)
    ax.text(8,6.6,"OFF-ROAD DESERT",color=WHITE,fontsize=36,ha="center",va="center",fontweight="bold")
    ax.text(8,5.75,"SEMANTIC SEGMENTATION",color=ACCENT,fontsize=36,ha="center",va="center",fontweight="bold")
    ax.plot([3.5,12.5],[5.28,5.28],color=ACCENT,lw=1.5)
    ax.text(8,4.85,"Pixel-wise terrain labelling for synthetic off-road desert scenes",color=WHITE,fontsize=13,ha="center",va="center")
    ax.text(8,4.3,"using ResNet50 + Feature Pyramid Network Decoder",color=GREY,fontsize=11,ha="center",va="center")
    ax.text(8,3.25,"Team  AIDHUNIK   ·   Hack For Green Bharat   ·   February 26, 2026",color=GREY,fontsize=9,ha="center",va="center")
    sv(pdf,fig)

    ### SLIDE 2 – PROBLEM
    fig,ax=slide("The Problem","Why terrain perception matters for autonomous off-road systems")
    items=[("🌵","Harsh Terrain Diversity","10 semantically distinct terrain classes\nthat look visually similar in desert\nenvironments make classification hard."),
           ("⚙️","Sim-to-Real Gap","Synthetic data has distribution shifts vs.\nreal imagery. Models must generalise\nacross lighting, texture, and geometry."),
           ("📉","Class Imbalance","Sky and ground dominate pixel counts;\nrare classes like bushes/rocks are under-\nrepresented, biasing naïve models.")]
    for i,(icon,t,b) in enumerate(items):
        x0=0.6+i*5.1
        box(ax,x0,1.3,4.6,5.8,ec=ACCENT2)
        ax.text(x0+2.3,6.65,icon,fontsize=30,ha="center",va="center")
        ax.text(x0+2.3,5.9,t,color=ACCENT2,fontsize=11.5,ha="center",va="center",fontweight="bold")
        ax.text(x0+2.3,4.55,b,color=WHITE,fontsize=9,ha="center",va="center",multialignment="center",linespacing=1.8)
    sv(pdf,fig)

    ### SLIDE 3 – DATASET
    fig,ax=slide("Dataset","Duality AI — Synthetic Desert Off-Road Scenes")
    stats=[("2 857","Training\nimages",ACCENT),("317","Validation\nimages",ACCENT2),("10","Semantic\nclasses",WARN),("476×266","Image\nresolution",GREY)]
    for j,(v,l,c) in enumerate(stats):
        x0=0.8+j*3.7
        box(ax,x0,4.8,3.2,2.55,ec=c)
        ax.text(x0+1.6,6.18,v,color=c,fontsize=22,ha="center",va="center",fontweight="bold")
        ax.text(x0+1.6,5.22,l,color=WHITE,fontsize=9,ha="center",va="center",multialignment="center")
    classes=["Background","Trees","Dry Grass","Rocks","Bushes","Sky","Ground Clutter","Terrain","Shrubs","Trail"]
    raw=[0,100,200,300,500,550,700,800,7100,10000]
    cols=[ACCENT,ACCENT2,WARN,"#DA9B52","#A371F7","#79C0FF","#FFA657","#56D364","#F0883E","#FF7B72"]
    ax.text(1.0,4.35,"Mask pixel label mapping:",color=GREY,fontsize=9)
    for i,(cls,rv,c) in enumerate(zip(classes,raw,cols)):
        ci=i%5; ri=i//5
        x0=0.8+ci*3.0; y0=3.6-ri*0.72
        ax.add_patch(plt.Circle((x0+0.18,y0+0.14),0.11,fc=c,ec="none"))
        ax.text(x0+0.45,y0+0.14,f"{cls}  (raw={rv})",color=WHITE,fontsize=7.8,va="center")
    sv(pdf,fig)

    ### SLIDE 4 – ARCHITECTURE
    fig,ax=slide("Architecture","ResNet50 Multi-Scale Backbone + FPN Decoder + ConvNeXt Head")
    def rbox(x,y,w,h,main,sub="",ec=ACCENT):
        box(ax,x,y,w,h,ec=ec)
        ax.text(x+w/2,y+h/2+(0.12 if sub else 0),main,color=WHITE,fontsize=8.5,ha="center",va="center",fontweight="bold")
        if sub: ax.text(x+w/2,y+h/2-0.22,sub,color=GREY,fontsize=7,ha="center",va="center")
    rbox(0.3,3.5,1.9,2.7,"INPUT","476×266×3",ec=GREY)
    rbox(2.7,5.4,2.3,0.72,"Stem+L1+L2","❄ frozen",ec=GREY)
    rbox(2.7,4.54,2.3,0.72,"Layer 3","🔥 LR=5e-5",ec=WARN)
    rbox(2.7,3.65,2.3,0.72,"Layer 4","🔥 LR=5e-5",ec=WARN)
    def arr(x1,y1,x2,y2,c=GREY):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops={"arrowstyle":"->","color":c,"lw":1.3})
    arr(2.2,4.85,2.7,5.77); arr(5.0,5.77,5.55,5.3); arr(5.0,4.9,5.55,4.9); arr(5.0,4.0,5.55,4.55)
    arr(2.2,4.85,2.7,4.9); arr(2.2,4.85,2.7,4.0)
    rbox(5.55,4.0,2.4,2.5,"FPN\nDecoder","fpn_channels=256\ntop-down merging",ec=ACCENT2)
    arr(7.95,5.25,8.4,5.22,ACCENT)
    rbox(8.4,4.5,2.5,1.5,"ConvNeXt\n×3 Blocks","DW-Conv→LN→GELU",ec=ACCENT)
    arr(10.9,5.25,11.35,5.22,ACCENT)
    rbox(11.35,4.5,2.3,1.5,"1×1 Conv\nClassifier","10 classes\nsoftmax",ec=ACCENT)
    arr(13.65,5.25,14.1,5.22,ACCENT)
    rbox(14.1,4.5,1.6,1.5,"OUTPUT\nMask","476×266",ec=ACCENT2)
    box(ax,0.3,1.4,15.4,1.85,ec=WARN)
    ax.text(8.0,2.55,"Loss = CrossEntropy(class-weighted, label-smooth=0.05)  +  0.5 × SoftDice",color=WHITE,fontsize=9,ha="center",va="center",fontweight="bold")
    ax.text(8.0,2.0,"Optimiser: AdamW   ·   Scheduler: CosineAnnealingLR T₀=10   ·   Head LR=5e-4   ·   Backbone LR=5e-5",color=GREY,fontsize=8.5,ha="center",va="center")
    sv(pdf,fig)

    ### SLIDE 5 – TRAINING CURVES
    fig,ax=slide("Training Results","V3 FPN — 15 epochs on single T4 GPU")
    axi=fig.add_axes([0.06,0.19,0.56,0.57])
    axi.set_facecolor(CARD)
    axi.plot(ep,iou,color=ACCENT,lw=2.5,marker="o",ms=5,label="Val IoU")
    axi.plot(ep,acc,color=ACCENT2,lw=2,marker="s",ms=4,ls="--",label="Val Accuracy")
    axi.axhline(0.2303,color=WARN,lw=1.8,ls=":",label="Baseline IoU=0.23")
    axi.set_xlabel("Epoch",color=WHITE,fontsize=9); axi.set_ylabel("Metric",color=WHITE,fontsize=9)
    axi.set_title("Val IoU & Accuracy vs Epoch",color=WHITE,fontsize=10)
    axi.legend(facecolor=CARD,edgecolor=GREY,labelcolor=WHITE,fontsize=8)
    axi.tick_params(colors=WHITE,labelsize=8); axi.set_xticks(ep[::2])
    for sp in axi.spines.values(): sp.set_edgecolor(GREY)
    metrics2=[("0.3903","Best Val IoU",ACCENT),("0.5615","Best Val Dice",ACCENT2),("81.2%","Val Accuracy",WARN),("+70%","IoU vs Baseline",ACCENT)]
    for j,(v,l,c) in enumerate(metrics2):
        x0=9.5+(j%2)*3.1; y0=4.6-(j//2)*2.5
        box(ax,x0,y0,2.7,2.1,ec=c)
        ax.text(x0+1.35,y0+1.35,v,color=c,fontsize=21,ha="center",va="center",fontweight="bold")
        ax.text(x0+1.35,y0+0.38,l,color=WHITE,fontsize=8,ha="center",va="center")
    sv(pdf,fig)

    ### SLIDE 6 – COMPARISON TABLE
    fig,ax=slide("Model Comparison","Baseline v1  →  V3 FPN")
    hdr=["Model","Epochs","Val IoU","Val Dice","Val Accuracy","Val Loss"]
    rows2=[["Baseline v1\n(frozen backbone)","10","0.2305","0.3633","66.08%","0.96"],
           ["V3 FPN ★\n(partial fine-tune)","15","0.3903","0.5615","81.20%","1.79"],
           ["Δ improvement","—","+69.6%","+54.5%","+22.9 pp","—"]]
    cxs=[0.4,3.45,5.3,7.1,9.0,11.3]; cws=[3.0,1.8,1.75,1.75,2.2,1.6]
    rys=[4.5,3.2,2.0]
    ax.add_patch(FancyBboxPatch((0.3,5.8),15.4,0.68,boxstyle="square,pad=0",fc=ACCENT,ec="none"))
    for hi,(h,cx,cw) in enumerate(zip(hdr,cxs,cws)):
        ax.text(cx+cw/2,6.14,h,color=BG,fontsize=9.5,ha="center",va="center",fontweight="bold")
    rfc=[CARD,"#1C2128",CARD]; rec=[GREY,ACCENT,ACCENT2]
    for ri,(row,ry,fc_,ec_) in enumerate(zip(rows2,rys,rfc,rec)):
        ax.add_patch(FancyBboxPatch((0.3,ry),15.4,1.1,boxstyle="square,pad=0",fc=fc_,ec=ec_,lw=0.8))
        for ci,(cell,cx,cw) in enumerate(zip(row,cxs,cws)):
            c2=ACCENT if (ri==2 and ci>1) else WHITE
            ax.text(cx+cw/2,ry+0.55,cell,color=c2,fontsize=9,ha="center",va="center",multialignment="center")
    sv(pdf,fig)

    ### SLIDE 7 – KEY INNOVATIONS
    fig,ax=slide("Key Innovations","What made V3 outperform the baseline by 70%")
    invs=[(ACCENT,"1","Multi-Scale FPN","Fuse features from layer2, layer3, layer4\nCaptures both fine detail and coarse semantics\nTop-down pathway with lateral skip connections"),
          (ACCENT2,"2","Partial Backbone Fine-Tuning","Layer3 + Layer4 unfrozen at LR=5e-5\nAdapts ImageNet weights to desert textures\n10× lower LR than head to avoid forgetting"),
          (WARN,"3","Combined Loss + Class Weights","CE (label-smooth=0.05) + 0.5 × SoftDice\nInverse-frequency class weights tackle imbalance\nDice loss directly optimises the IoU metric"),
          (ACCENT,"4","ConvNeXt-Style Head","3 × ConvNeXt blocks: DW-Conv → LN → GELU\nMore expressive than plain conv stacks\nCosineAnnealing LR restarts every T₀=10 epochs")]
    for i,(c,num,t,b) in enumerate(invs):
        x0=0.5+(i%2)*7.9; y0=1.4+(1-(i//2))*3.1
        box(ax,x0,y0,7.3,2.75,ec=c)
        ax.add_patch(plt.Circle((x0+0.6,y0+2.21),0.33,fc=c,ec="none"))
        ax.text(x0+0.6,y0+2.21,num,color=BG,fontsize=14,ha="center",va="center",fontweight="bold")
        ax.text(x0+1.4,y0+2.21,t,color=c,fontsize=11.5,va="center",fontweight="bold")
        ax.text(x0+3.65,y0+1.1,b,color=WHITE,fontsize=8.5,ha="center",va="center",multialignment="center",linespacing=1.8)
    sv(pdf,fig)

    ### SLIDE 8 – BAR CHART
    fig,ax=slide("Per-Epoch Val IoU","Training progression over 15 epochs")
    ab=fig.add_axes([0.07,0.16,0.88,0.62])
    ab.set_facecolor(CARD)
    bc=[ACCENT2]*14+[ACCENT]
    bars=ab.bar(ep,iou,color=bc,width=0.65,zorder=3)
    ab.axhline(0.2303,color=WARN,lw=2,ls="--",label="Baseline (0.2303)")
    ab.tick_params(colors=WHITE,labelsize=9); ab.set_xlabel("Epoch",color=WHITE,fontsize=10)
    ab.set_ylabel("Validation IoU",color=WHITE,fontsize=10); ab.set_xticks(ep)
    ab.legend(facecolor=CARD,edgecolor=GREY,labelcolor=WHITE,fontsize=9)
    ab.set_ylim(0.19,0.43)
    for sp in ab.spines.values(): sp.set_edgecolor(GREY)
    for bar,v in zip(bars,iou):
        ab.text(bar.get_x()+bar.get_width()/2,v+0.003,f"{v:.3f}",color=WHITE,fontsize=6.5,ha="center")
    ax.text(8,0.58,"Best: Epoch 15  →  Val IoU = 0.3903  (+69.6% over baseline)",
            color=ACCENT,fontsize=11,ha="center",va="center",fontweight="bold")
    sv(pdf,fig)

    ### SLIDE 9 – SUBMISSION PACKAGE
    fig,ax=slide("Submission Package","What we're delivering")
    rows3=[("📁","SUBMISSION/src/","train_segmentation.py  ·  test_segmentation.py  ·  visualize.py"),
           ("🧠","models/","segmentation_head_best.pth  (120 MB — best checkpoint, epoch 15)"),
           ("📊","train_stats/v3/","evaluation_metrics_v3.txt  ·  4 training curve plots (PNG)"),
           ("📄","README.md  &  REPORT.md","Architecture docs, results table, setup & usage instructions"),
           ("📦","requirements.txt","torch · torchvision · numpy · opencv-python · albumentations · tqdm"),
           ("🔗","GitHub Repo","github.com/Akum030/duality-desert-segmentation")]
    for i,(icon,t,d) in enumerate(rows3):
        y0=6.8-i*0.91
        ec_=ACCENT if i%2==0 else ACCENT2
        box(ax,0.4,y0-0.3,15.2,0.72,ec=ec_)
        ax.text(0.95,y0+0.06,icon,fontsize=14,va="center")
        ax.text(1.7,y0+0.06,t,color=ec_,fontsize=10,va="center",fontweight="bold")
        ax.text(5.7,y0+0.06,d,color=WHITE,fontsize=9,va="center")
    sv(pdf,fig)

    ### SLIDE 10 – THANK YOU
    fig,ax=slide()
    box(ax,1.5,2.0,13,5.8,ec=ACCENT,lw=2.5)
    ax.text(8,6.55,"Thank You!",color=ACCENT,fontsize=44,ha="center",va="center",fontweight="bold")
    ax.text(8,5.7,"Team  AIDHUNIK",color=WHITE,fontsize=19,ha="center",va="center",fontweight="bold")
    ax.plot([4,12],[5.25,5.25],color=ACCENT,lw=1.5)
    ms=[("Val IoU","0.3903",ACCENT),("Val Dice","0.5615",ACCENT2),("Val Acc","81.2%",WARN),("Δ IoU","+70%",ACCENT)]
    for j,(l,v,c) in enumerate(ms):
        x0=2.2+j*3.0
        ax.text(x0+1.1,4.65,v,color=c,fontsize=22,ha="center",va="center",fontweight="bold")
        ax.text(x0+1.1,4.05,l,color=GREY,fontsize=10,ha="center",va="center")
    ax.text(8,3.15,"github.com/Akum030/duality-desert-segmentation",color=ACCENT2,fontsize=10,ha="center",va="center")
    ax.text(8,2.62,"Hack For Green Bharat  ·  February 26, 2026",color=GREY,fontsize=9,ha="center",va="center")
    sv(pdf,fig)

print("PDF saved → /home3/indiamart/gbht/aidhunik_pitch_deck.pdf")
