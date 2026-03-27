
import numpy as np
from gymnasium import Env, spaces
import pandas as pd
# NO matplotlib.use('Agg') aquí — dejar que Jupyter controle el backend
import matplotlib.pyplot as plt


class motorEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        super(motorEnv, self).__init__()
        self.rot = pd.read_csv('../../Datos/Originales/02_Reinforcement_learning/Datos_v1.csv')
        self.rot = self.rot.drop('index', axis=1)
        self.rot['col'] = '0'

        self.num_states = len(self.rot)
        self.w_min   = self.rot['w'].min()
        self.w_max   = self.rot['w'].max()
        self.w_range = self.w_max - self.w_min   # ≈ 1.928

        self.var1      = None
        self.var2      = None
        self.new_state = None
        self.old_state = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _try_move(self, v1, v2):
        result = self.rot[(self.rot['var1'] == v1) & (self.rot['var2'] == v2)]
        if len(result):
            return result.index[0], True
        return self.old_state, False

    # ── step (SIN matplotlib) ─────────────────────────────────────────────────────
    def step(self, a):
        done   = False
        reward = 0.0

        v1, v2 = self.var1, self.var2
        if   a == 0: v1 += 1
        elif a == 1: v1 -= 1
        elif a == 2: v2 += 1
        elif a == 3: v2 -= 1

        self.new_state, moved = self._try_move(v1, v2)

        if not moved:
            reward = -1.0
        else:
            self.var1 = v1
            self.var2 = v2

            w_old = self.rot.iloc[self.old_state, 2]
            w_new = self.rot.iloc[self.new_state, 2]

            delta_w = w_old - w_new
            reward  = (delta_w / self.w_range) * 1000.0
            reward -= 0.3

            if w_new == self.w_min:
                reward += 1000.0
                done    = True

        self.rot.iloc[self.new_state, 3] = '1'
        self.old_state = self.new_state

        return self.new_state, reward, done, {}

    # ── reset — SIEMPRE index 0 ───────────────────────────────────────────────────
    def reset(self):
        self.var1      = self.rot.iloc[0, 0]   # 0.0
        self.var2      = self.rot.iloc[0, 1]   # 172.0
        self.old_state = 0
        self.rot.iloc[0, 3] = '1'
        return self.old_state

    # ── render — solo cuando se llama explícitamente ─────────────────────────────
    def render(self, camino_v1=None, camino_v2=None, camino_w=None,
               titulo="Ruta del Agente", save_path=None):
        """
        Dibuja el mapa de w con la ruta superpuesta.
        Pasar camino_v1/v2/w para mostrar la ruta; si no, solo el mapa.
        """
        fig, ax = plt.subplots(figsize=(11, 8))
        sc = ax.scatter(self.rot['var1'], self.rot['var2'],
                        c=self.rot['w'], cmap='viridis', alpha=0.4, s=15)
        plt.colorbar(sc, label='Consumo de Energía (w)', ax=ax)

        min_rows = self.rot[self.rot['w'] == self.w_min]
        ax.scatter(min_rows['var1'], min_rows['var2'],
                   color='lime', s=5, alpha=0.7, label='Zona mínimo global')

        if camino_v1 is not None:
            ax.plot(camino_v1, camino_v2, color='red', linewidth=2,
                    marker='.', markersize=4,
                    label=f'Ruta agente ({len(camino_v1)-1} pasos)')
            ax.scatter(camino_v1[0],  camino_v2[0],
                       color='blue', s=200, zorder=6, label='Inicio')
            ax.scatter(camino_v1[-1], camino_v2[-1], color='gold',
                       marker='*', s=400, edgecolors='black', zorder=6,
                       label=f'Final  w={camino_w[-1]:.4f}')

        ax.set_xlabel('var1'); ax.set_ylabel('var2')
        ax.set_title(titulo)
        ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)