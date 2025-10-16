import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path
SETTING = "ir_classification_mTAN_P12_S2IPLLM_mTAN_encoder_P12_phy12_split3_ftM_sl96_ll1_pl0_dm768_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0"
ROOT    = Path("/data/gpfs/projects/punim2341/feixiangz/S2IP-LLM/Irregular_Classification/test_results") / SETTING
MATRIX  = ROOT / "sim_matrix.npy"

if not MATRIX.exists():
    raise FileNotFoundError(f"Cannot find {MATRIX}. Run testing first!")

sim = np.load(MATRIX)[:10, :]
print("Loaded similarity matrix:", sim.shape)

words = ["Trend", "seasonality", "cyclicity", "rise", "peak", "pattern", "shift", "position",
                     "irregular", "missing", "inconsistent", "discontinuous", "heart", "period", "echo",
                     "arm", "key", "mint"]

plt.figure(figsize=(8, 6))
im = plt.imshow(sim, aspect='auto')            # default colour map
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(np.arange(sim.shape[1]), words, rotation=45, ha='right') # words
plt.yticks(np.arange(10), [f"{i+1}" for i in range(10)]) # instances

plt.title("Cosine similarity between words\nand 10 test instances")
plt.tight_layout()
plt.savefig(str(ROOT / "sim_heatmap.png"), dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved to {str(ROOT / 'sim_heatmap.png')}")