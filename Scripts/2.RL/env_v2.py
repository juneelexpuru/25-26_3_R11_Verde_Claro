#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_con_reward_potencia.py

Reward original preservado:
  • Paso normal : -1 + factor * delta_w   (factor=5, delta_w no normalizado)
  • Muro        : -100
  • Terminal    : +10000 al alcanzar w_min

Estilo env_final.py:
  • numpy arrays pre-extraídos (sin pandas iloc en el loop)
  • state_dict para lookup O(1)
  • Sin matplotlib en step() — solo en render()
  • Parámetro malla=1|2 para seleccionar el CSV
"""

import numpy as np
from gymnasium import Env, spaces
import pandas as pd
import matplotlib.pyplot as plt


class motorEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, malla=1):
        super(motorEnv, self).__init__()

        if malla == 1:
            df = pd.read_csv('../../Datos/Originales/02_Reinforcement_learning/Datos_v1.csv')
        elif malla == 2:
            df = pd.read_csv('../../Datos/Originales/02_Reinforcement_learning/Datos_v2.csv')
        else:
            raise ValueError("El parámetro 'malla' debe ser 1 o 2")

        df = df.drop('index', axis=1)   # elimina columna índice si existe

        self.num_states = len(df)
        self.w_min      = df['w'].min()

        # ── Arrays numpy pre-extraídos: acceso O(1) en el loop ───────────────
        self.VAR1 = df['var1'].to_numpy()
        self.VAR2 = df['var2'].to_numpy()
        self.W    = df['w'].to_numpy()

        # ── Diccionario (var1, var2) → índice para lookup O(1) ──────────────
        self.state_dict = {
            (self.VAR1[i], self.VAR2[i]): i
            for i in range(self.num_states)
        }

        self.old_state  = None
        self.new_state  = None
        self.step_count = 0

        # Registro de caminos (para gráficos y métricas)
        self.last_path     = []   # camino del episodio actual
        self.shortest_path = []   # mejor camino que llegó al objetivo

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        print(f"Entorno cargado: malla={malla} | {self.num_states} estados")
        print(f"Mínimo global: w={self.w_min:.6f}")
        print(f"Inicio: var1={self.VAR1[0]}, var2={self.VAR2[0]}, w={self.W[0]:.6f}")

    # ── step (SIN matplotlib) ─────────────────────────────────────────────────
    def step(self, a, verbose=False):
        self.step_count += 1
        done      = False
        ha_salido = False

        v1 = self.VAR1[self.old_state]
        v2 = self.VAR2[self.old_state]
        w_old = self.W[self.old_state]

        if   a == 0: v1 += 1
        elif a == 1: v1 -= 1
        elif a == 2: v2 += 1
        elif a == 3: v2 -= 1

        if (v1, v2) in self.state_dict:
            self.new_state = self.state_dict[(v1, v2)]
            self.last_path.append((v1, v2))
        else:
            self.new_state = self.old_state
            reward    = -100
            ha_salido = True

        w_new = self.W[self.new_state]

        if w_new == self.w_min:
            reward = 10000
            done   = True
            if verbose:
                print(f"  *** MÍNIMO ALCANZADO *** Steps: {self.step_count}")

        if not ha_salido and not done:
            delta_w = w_old - w_new
            factor  = 5
            reward  = -1 + factor * delta_w   # idéntico al original

        self.old_state = self.new_state
        return self.new_state, reward, done, {}

    # ── reset — SIEMPRE index 0 ───────────────────────────────────────────────
    def reset(self,estado_inicial=None):
        # Guarda el camino si el último episodio llegó al objetivo
        if self.new_state is not None:
            w_final = self.W[self.new_state]
            if self.last_path and w_final == self.w_min:
                if not self.shortest_path or \
                        len(self.last_path) < len(self.shortest_path):
                    self.shortest_path = list(self.last_path)

        # --- AQUI ESTA EL CAMBIO ---
        if estado_inicial is not None:
            self.old_state = estado_inicial
            self.new_state = estado_inicial
        else:
            self.old_state = 0
            self.new_state = 0
        # ---------------------------

        self.step_count = 0
        self.last_path  = []
        return self.old_state

    # ── render — solo cuando se llama explícitamente ──────────────────────────
    def render(self, test=False, camino_v1=None, camino_v2=None,
               camino_w=None, titulo="Agent path", save_path=None):
        fig, ax = plt.subplots(figsize=(11, 8))
        sc = ax.scatter(self.VAR1, self.VAR2,
                        c=self.W, cmap='viridis', alpha=0.4, s=15)
        plt.colorbar(sc, label='Energy consumption (w))', ax=ax)

        # Mínimo global
        min_mask = self.W == self.w_min
        ax.scatter(self.VAR1[min_mask], self.VAR2[min_mask],
                   color='lime', s=10, alpha=0.9, zorder=3,
                   label='Global minima zone')

        # Último camino (rojo)
        if self.last_path:
            px, py = zip(*self.last_path)
            ax.plot(px, py, 'r-', linewidth=2, label='Last path')

        # Mejor camino (azul) — solo si no estamos en modo test
        if self.shortest_path and not test:
            bx, by = zip(*self.shortest_path)
            ax.plot(bx, by, 'b-', linewidth=2, label='Best path')

        # Ruta externa (opcional, para gráficos del notebook)
        if camino_v1 is not None:
            ax.plot(camino_v1, camino_v2, color='red', linewidth=2,
                    marker='.', markersize=4,
                    label=f'Ruta agente ({len(camino_v1)-1} pasos)')
            ax.scatter(camino_v1[0], camino_v2[0],
                       color='blue', s=200, zorder=6, label='Inicio')
            ax.scatter(camino_v1[-1], camino_v2[-1],
                       color='gold', marker='*', s=400,
                       edgecolors='black', zorder=6,
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
