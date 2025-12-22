"""
advanced_visualizer.py - Công cụ visualization nâng cao (advanced plots) cho NSGA-II results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # thư viện vẽ biểu đồ nâng cao
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Plotly cho tương tác (interactive 3D plot)
import plotly.graph_objects as go
import plotly.express as px


class AdvancedVisualizer:
    """
    Visualization nâng cao cho Pareto front & design space
    """

    def __init__(self):
        self.style = 'seaborn-v0_8-darkgrid'
        try:
            plt.style.use(self.style)
        except Exception:
            pass
        self.colors = plt.cm.tab10.colors

    # ------------------------------------------------------------------
    # 1) 3D Pareto front: Cost - Strength - CO2
    # ------------------------------------------------------------------
    def plot_3d_pareto_front(
        self,
        optimization_results: Dict,
        output_dir: Optional[str] = None
    ):
        """
        Vẽ 3D Pareto front: Cost vs Strength vs CO2

        Args:
            optimization_results: dict {cement_type: {'pareto_front': (X, F), ...}}
            output_dir: nếu không None thì lưu file .png
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        for idx, (cement_type, result) in enumerate(optimization_results.items()):
            X, F = result['pareto_front']

            cost = F[:, 0] / 1000.0          # cost (k VND/m3)
            strength = -F[:, 1]              # f28 (MPa)
            co2 = F[:, 3]                    # kgCO2/m3

            ax.scatter(
                cost, strength, co2,
                c=[self.colors[idx % len(self.colors)]],
                marker='o', s=40, alpha=0.6,
                edgecolors='black', linewidths=0.3,
                label=cement_type
            )

        ax.set_xlabel('Cost (k VNĐ/m³)', fontsize=12, labelpad=10)
        ax.set_ylabel('Strength f28 (MPa)', fontsize=12, labelpad=10)
        ax.set_zlabel('CO₂ (kgCO₂/m³)', fontsize=12, labelpad=10)
        ax.set_title('3D Pareto Front: Cost – Strength – CO₂', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out / f"pareto_3d_{ts}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved 3D Pareto to: {save_path}")

        plt.show()

    # ------------------------------------------------------------------
    # 2) Parallel coordinates – trade-off giữa objectives & features
    # ------------------------------------------------------------------
    def plot_parallel_coordinates(
        self,
        optimization_results: Dict,
        n_designs: int = 60,
        output_dir: Optional[str] = None
    ):
        """
        Vẽ parallel coordinates (trục song song) để xem trade-off

        Args:
            optimization_results: dict {cement_type: {'pareto_front': (X, F)}}
            n_designs: số điểm random mỗi cement_type
        """
        all_rows = []

        for cement_type, result in optimization_results.items():
            X, F = result['pareto_front']
            n = min(n_designs, len(X))
            if n <= 0:
                continue

            idx = np.random.choice(len(X), n, replace=False)
            for i in idx:
                x = X[i]
                f = F[i]
                binder = x[0] + x[2] + x[3] + x[4]
                w_b = x[1] / binder if binder > 0 else 0.0
                scm = x[2] + x[3] + x[4]
                scm_frac = scm / binder * 100 if binder > 0 else 0.0

                all_rows.append({
                    "Cement": cement_type,
                    "Cost": f[0] / 1000.0,
                    "Strength": -f[1],
                    "Slump Dev": f[2],
                    "CO₂": f[3],
                    "w/b": w_b,
                    "SCM %": scm_frac
                })

        if not all_rows:
            print("⚠️ Không có dữ liệu để vẽ parallel coordinates.")
            return

        df = pd.DataFrame(all_rows)

        from pandas.plotting import parallel_coordinates

        plt.figure(figsize=(16, 8))
        parallel_coordinates(
            df,
            "Cement",
            cols=["Cost", "Strength", "CO₂", "Slump Dev", "w/b", "SCM %"],
            color=self.colors[:len(optimization_results)],
            alpha=0.25
        )
        plt.title("Parallel Coordinates – Multi-objective Trade-offs", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out / f"parallel_coords_{ts}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved parallel coords to: {save_path}")

        plt.show()

    # ------------------------------------------------------------------
    # 3) 2D heatmap Cost vs Strength, màu = CO2
    # ------------------------------------------------------------------
    def plot_objective_heatmap(
        self,
        optimization_results: Dict,
        output_dir: Optional[str] = None
    ):
        """
        Heatmap (bản đồ màu) Cost–Strength, màu thể hiện CO₂
        """
        rows = []
        for cement_type, result in optimization_results.items():
            _, F = result['pareto_front']
            for f in F:
                rows.append({
                    "Cement": cement_type,
                    "Cost": f[0] / 1000.0,
                    "Strength": -f[1],
                    "CO₂": f[3],
                })

        if not rows:
            print("⚠️ Không có dữ liệu để vẽ heatmap.")
            return

        df = pd.DataFrame(rows)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df, x="Cost", y="Strength", hue="CO₂", palette="viridis",
            alpha=0.7
        )
        plt.title("Cost–Strength trade-off (màu = CO₂)", fontsize=14, fontweight='bold')
        plt.xlabel("Cost (k VNĐ/m³)")
        plt.ylabel("Strength f28 (MPa)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out / f"cost_strength_co2_{ts}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved Cost–Strength–CO₂ scatter to: {save_path}")

        plt.show()

    # ------------------------------------------------------------------
    # 4) Interactive 3D Pareto với Plotly (vẽ trong notebook/UI)
    # ------------------------------------------------------------------
    def plot_interactive_pareto_3d(self, optimization_results: Dict):
        """
        3D Pareto interactive (Plotly) – dùng tốt cho Streamlit / notebook
        """
        traces = []

        for cement_type, result in optimization_results.items():
            _, F = result['pareto_front']
            cost = F[:, 0] / 1000.0
            strength = -F[:, 1]
            co2 = F[:, 3]

            traces.append(go.Scatter3d(
                x=cost, y=strength, z=co2,
                mode="markers",
                name=cement_type,
                marker=dict(
                    size=5,
                    opacity=0.7
                ),
                hovertemplate=(
                    "Cement: %{text}<br>"
                    "Cost: %{x:.1f} kVND/m³<br>"
                    "Strength: %{y:.1f} MPa<br>"
                    "CO₂: %{z:.0f} kg/m³"
                ),
                text=[cement_type] * len(cost)
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="3D Pareto Front (Interactive)",
            scene=dict(
                xaxis_title="Cost (k VNĐ/m³)",
                yaxis_title="Strength f28 (MPa)",
                zaxis_title="CO₂ (kgCO₂/m³)"
            ),
            legend_title="Cement Type",
            height=650
        )
        return fig  # để Streamlit hiển thị bằng st.plotly_chart(fig)


if __name__ == "__main__":
    print("✅ AdvancedVisualizer ready. Dùng với optimization_results từ NSGA-II.")
