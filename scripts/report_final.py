"""Build the report from the saved official predictions without inference."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_paths import DEFAULT_DATA_ROOT, resolve_data_root

METRICS = ("image_auroc", "pixel_auroc", "aupro")
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_cases(cases, maps, masks, data_root, path, name):
    """Use a shared color scale for the heatmaps and overlays in each figure."""
    norm = Normalize(0, max(float(maps[row["index"]].max()) for row in cases))
    fig, axes = plt.subplots(len(cases), 4, squeeze=False, figsize=(12, 3.2*len(cases)), layout="constrained")
    fig.suptitle(f"MVTec AD · {name} · modelo completo")
    for row_number, (row_axes, case) in enumerate(zip(axes, cases)):
        index = case["index"]
        with Image.open(data_root / case["image"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((maps.shape[2], maps.shape[1]), Image.Resampling.BILINEAR))
        row_axes[0].imshow(rgb)
        row_axes[1].imshow(masks[index], cmap="gray", vmin=0, vmax=1)
        heat = row_axes[2].imshow(maps[index], cmap="magma", norm=norm)
        row_axes[3].imshow(rgb)
        row_axes[3].imshow(maps[index], cmap="magma", norm=norm, alpha=.52)
        if masks[index].any():
            row_axes[3].contour(masks[index], levels=[.5], colors=["cyan"], linewidths=.75)
        for axis, title in zip(row_axes, ("Imagem", "Máscara de referência", "Mapa de calor", "Sobreposição")):
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        row_axes[0].set_ylabel(f"{case['case']}\n{case['category']} · {case['score']:.2f}", fontsize=10)
    fig.colorbar(heat, ax=axes, shrink=.65, label="Discrepância L2 por pixel")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("data_root", type=Path, nargs="?", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    out = args.results
    data_root = resolve_data_root(args.data_root)
    manifest = json.loads((out / "final_test_run.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise RuntimeError("Evaluation is incomplete")
    metric_rows = csv_rows(out / "final_metrics.csv")
    metric = {(row["model"], row["category"]): row for row in metric_rows}
    with np.load(out / "complete_predictions.npz", allow_pickle=False) as saved:
        scores, labels, cats, paths = (saved[key] for key in
            ("image_scores", "image_labels", "categories", "image_paths"))
        maps = saved["anomaly_maps"]
        packed = saved["packed_masks"]
    masks = np.unpackbits(packed, axis=1, count=maps.shape[1]*maps.shape[2]).reshape(maps.shape).view(bool)
    names = sorted(set(cats))
    threshold = float(np.quantile(scores[labels == 0], .95))
    normal_pixels = maps[labels == 0, ::8, ::8]
    pixel_threshold = float(np.quantile(normal_pixels, .99))
    del normal_pixels
    decisions = []
    for i, (score, label, category, path) in enumerate(zip(scores, labels, cats, paths)):
        mask = masks[i]
        prediction = maps[i] >= pixel_threshold
        pred_anomaly = score >= threshold
        case = ("VP" if pred_anomaly else "FN") if label else ("FP" if pred_anomaly else "VN")
        union = int(np.count_nonzero(mask | prediction))
        border = np.ones(mask.shape, dtype=bool)
        border[32:-32,32:-32] = False
        decisions.append({"index": i, "image": str(path), "category": str(category),
             "group": "texturas" if category in TEXTURES else "objetos", "defect": Path(path).parent.name,
             "case": case, "score": float(score), "mask_fraction": float(mask.mean()),
             "iou": np.count_nonzero(mask & prediction) / union if union else 0.0,
             "hot_border_fraction": float(prediction[border].sum()/max(1,prediction.sum())),
             "mask_border_fraction": float(mask[border].sum()/max(1,mask.sum()))})
    write_csv(out / "diagnostico_por_imagem.csv", decisions)

    examples = []
    group_thresholds = {}
    for group in ("objetos", "texturas"):
        cases = []
        for case, field, reverse in (("VP", "iou", True), ("FN", "score", False), ("FP", "score", True)):
            pool = [row for row in decisions if row["group"] == group and row["case"] == case]
            if not pool and case == "FP":
                normal = [row for row in decisions if row["group"] == group and row["case"] == "VN"]
                if normal:
                    cutoff = float(np.quantile([row["score"] for row in normal], .95))
                    group_thresholds[group] = cutoff
                    pool = [dict(row, case="FP (P95 do grupo)") for row in normal if row["score"] >= cutoff]
            if not pool:
                continue
            best = sorted(pool, key=lambda row: ((-1 if reverse else 1)*row[field],row["image"]))[0]
            cases.append(best)
        plot_cases(cases, maps, masks, data_root, out/f"casos_{group}.png", group.capitalize())
        examples.extend(cases)
    write_csv(out / "casos_selecionados.csv", examples)

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(13, 8), layout="constrained")
    for ax, model, title in zip(axes, ("baseline", "complete"), ("Baseline", "Modelo completo")):
        with np.load(out / f"{model}_predictions.npz", allow_pickle=False) as saved:
            s, l, c = (saved[k] for k in ("image_scores", "image_labels", "categories"))
        groups = [s[(l == 0) & (c == category)] for category in names]
        ax.boxplot(groups, tick_labels=names, vert=False, patch_artist=True,
                   boxprops={"facecolor": "#cce4f2"}, medianprops={"color": "#154b73"})
        ax.set(title=title, xlabel="Escore de imagem · média do 1% superior")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Distribuição dos escores das imagens normais por categoria")
    fig.savefig(out / "escores_normais.png", dpi=300)
    plt.close(fig)

    def line(category):
        values = [float(metric[model, category][key])*100 for model in ("baseline", "complete") for key in METRICS]
        return f"| {category} | " + " | ".join(f"{value:.2f}".replace(".",",") for value in values) + " |"

    lines = ["# Avaliação final no MVTec AD", "",
        "| Categoria | Baseline AUROC imagem | Baseline AUROC pixel | Baseline AUPRO | Completo AUROC imagem | Completo AUROC pixel | Completo AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|", *map(line, names), line("macro"), "",
        "Resultados em porcentagem. A média macro atribui o mesmo peso a cada uma das 15 categorias. "
        "AUROC de imagem por postos, AUROC de pixel e AUPRO por histogramas monotônicos de 2.048 bins. "
        "A AUPRO é normalizada até FPR de 0,30, com regiões de conectividade 4.", "",
        "## Protocolo e rastreabilidade", "",
        f"Commit do avaliador: {manifest['evaluation_code_commit']}.",
        "Entrada RGB 512 × 512, normalização ImageNet, WideResNet50-2 congelada, discrepância L2, interpolação bilinear e média do 1% superior para o escore de imagem.", ""]
    for model, details in manifest["models"].items():
        lines.append(f"- **{model}**, época {details['epoch']}, SHA-256 `{details['checkpoint_sha256']}`, commit de treino `{details['training_commit']}`.")
    lines.extend(["", "## Seleção dos casos e legendas", "",
        f"Limiar diagnóstico de imagem: P95 dos escores das normais do modelo completo ({threshold:.4f}). "
        f"Limiar de mapa: P99 dos pixels das normais, amostrados a cada 8 pixels em cada dimensão ({pixel_threshold:.4f}). "
        "Esses limiares descrevem os erros nesta rodada e devem ser calibrados em um conjunto independente para uso operacional.", "",
        "Em cada grupo, o VP é o defeito detectado com maior IoU, o FN é o defeito não detectado com menor escore e "
        "o FP é a imagem normal com maior escore. Em empates, vale a ordem do caminho do arquivo. "
        "Cada figura usa uma escala L2 comum e a sobreposição tem o contorno ciano da anotação.", ""])
    for group, cutoff in group_thresholds.items():
        lines.append(f"No grupo {group}, o limiar global não produziu FP. "
            f"Para ilustrar um alarme relativo ao grupo, a figura usa a normal de maior escore acima do P95 "
            f"das normais desse grupo ({cutoff:.4f}). As contagens abaixo continuam usando o limiar global.")
        lines.append("")
    for row in examples:
        lines.append(f"- **{row['group']} · {row['case']}** — `{row['image']}`. Escore {row['score']:.3f}, IoU {row['iou']:.3f}, máscara em {100*row['mask_fraction']:.3f}% da imagem.")
    lines.extend(["", "## Erros e limitações", ""])
    for key, label in zip(METRICS, ("AUROC de imagem", "AUROC de pixel", "AUPRO")):
        lowest = sorted(names, key=lambda c: float(metric["complete",c][key]))[:3]
        lines.append(f"- Menores valores de {label} no modelo completo: " +
                     ", ".join(f"{c} ({100*float(metric['complete',c][key]):.2f}%)" for c in lowest) + ".")
    for group in ("objetos", "texturas"):
        g = [row for row in decisions if row["group"] == group]
        fn, fp = ([row for row in g if row["case"] == case] for case in ("FN","FP"))
        anomalies = [row for row in g if row["case"] in ("VP","FN")]
        normal_count = len(g) - len(anomalies)
        tiny = [row for row in anomalies if row["mask_fraction"] < .01]
        larger = [row for row in anomalies if row["mask_fraction"] >= .01]
        lines.append(f"- **{group.capitalize()}** — {len(fn)}/{len(anomalies)} FN e {len(fp)}/{normal_count} FP. "
            f"Defeitos com máscara <1%: {sum(row['case']=='FN' for row in tiny)}/{len(tiny)} FN; "
            f"máscaras ≥1%: {sum(row['case']=='FN' for row in larger)}/{len(larger)} FN. "
            f"Em {sum(row['hot_border_fraction'] >= .5 for row in fp)}/{len(fp)} FP, ao menos metade dos pixels acima do limiar fica nos 32 pixels próximos à borda da imagem.")
    lines.extend(["- Os rótulos disponíveis descrevem defeito e máscara. Variações de pose, bordas do objeto e reflexos são hipóteses a verificar visualmente nos mapas; a contagem de borda acima se refere à borda da imagem.",
        "- A grade de features é 32 × 32 e os mapas são interpolados a 512 × 512. Isso limita a localização fina. "
        "O extrator foi pré-treinado em ImageNet. Ambos os treinos usam uma única semente. "
        "Os checkpoints diferem em duração e critério de seleção, de modo que as métricas comparam as versões treinadas e não isolam o efeito da máscara e do jitter.",
        "- A validação de treino tinha somente imagens normais. Houve inspeção visual prévia de amostras do teste e ela deve ser registrada ao interpretar a avaliação oficial.",
        "- Compatibilidade corrigida antes da rodada: integração trapezoidal via SciPy para funcionar também com NumPy recente. "
        "Os mapas foram arquivados em precisão float32 para preservar a mesma precisão usada nas métricas.", "",
        "## Discussão", ""])
    medians = {category:float(np.median(scores[(cats==category)&(labels==0)])) for category in names}
    lo,hi = min(medians,key=medians.get),max(medians,key=medians.get)
    deltas = [(float(metric["complete","macro"][k])-float(metric["baseline","macro"][k]))*100 for k in METRICS]
    lines.append(f"Em comparação com o baseline, o modelo completo variou {deltas[0]:+.2f} p.p. em AUROC de imagem, "
        f"{deltas[1]:+.4f} p.p. em AUROC de pixel e {deltas[2]:+.2f} p.p. em AUPRO macro. "
        f"As medianas dos escores normais foram de {medians[lo]:.2f} ({lo}) a {medians[hi]:.2f} ({hi}), "
        "o que quantifica o deslocamento de escala entre categorias e ajuda a interpretar o efeito de aplicar um limiar global. "
        "Os mapas e a análise por tamanho de máscara mostram por que a ordenação de imagens e a localização espacial devem ser avaliadas separadamente.")
    (out/"relatorio_avaliacao.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
