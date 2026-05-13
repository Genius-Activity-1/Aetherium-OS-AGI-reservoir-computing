import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# ============================================================
# 0. CONFIG DE BASE
# ============================================================

class AetheriumConfig:
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        H: int = 64,
        W: int = 64,
        latent_channels_phi: int = 16,
        latent_channels_psi: int = 32,
        latent_channels_omega: int = 16,
        steps_pred: int = 8,
        device: str = "cuda"
    ):
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.H = H
        self.W = W
        self.latent_channels_phi = latent_channels_phi
        self.latent_channels_psi = latent_channels_psi
        self.latent_channels_omega = latent_channels_omega
        self.steps_pred = steps_pred
        self.device = device

    @property
    def latent_channels_total(self):
        return (
            self.latent_channels_phi
            + self.latent_channels_psi
            + self.latent_channels_omega
        )

# ============================================================
# 1. ENCODEUR CNN
# ============================================================

class Encoder(nn.Module):
    """
    Encode une image (B, C, H, W) en un latent Z (B, C_lat, H', W').
    H'/W' ~ H/4, W/4 par exemple.
    """
    def __init__(self, cfg: AetheriumConfig):
        super().__init__()
        C = cfg.base_channels
        self.conv1 = nn.Conv2d(cfg.in_channels, C, 4, 2, 1)   # H/2
        self.conv2 = nn.Conv2d(C, C * 2, 4, 2, 1)             # H/4
        self.conv3 = nn.Conv2d(C * 2, cfg.latent_channels_total, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x), 0.1)
        h = F.leaky_relu(self.conv2(h), 0.1)
        z = self.conv3(h)
        return z


# ============================================================
# 2. FACTORISATION Z → (Φ, Ψ, Ω)
# ============================================================

def split_latent_phi_psi_omega(
    z: torch.Tensor, cfg: AetheriumConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    z: (B, C_total, H', W')
    retourne: (phi, psi, omega) avec découpe sur la dimension canal.
    """
    B, C, H, W = z.shape
    c_phi = cfg.latent_channels_phi
    c_psi = cfg.latent_channels_psi
    c_omega = cfg.latent_channels_omega

    phi = z[:, :c_phi]
    psi = z[:, c_phi:c_phi + c_psi]
    omega = z[:, c_phi + c_psi:c_phi + c_psi + c_omega]
    return phi, psi, omega


def merge_phi_psi_omega(
    phi: torch.Tensor, psi: torch.Tensor, omega: torch.Tensor
) -> torch.Tensor:
    """
    Construit A(x,t) = Φ + Ψ + Ω au niveau latent (concat → conv ou simple somme).
    Ici on choisit concat+conv pour laisser le modèle apprendre la combinaison.
    """
    z_cat = torch.cat([phi, psi, omega], dim=1)
    return z_cat


# ============================================================
# 3. NOYAUX TEMPORELS (LSTM Φ + ConvLSTM Ψ/Ω)
# ============================================================

class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM 2D standard (pour Ψ et Ω).
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            B, C, H, W = x.shape
            h = x.new_zeros(B, self.hidden_channels, H, W)
            c = x.new_zeros(B, self.hidden_channels, H, W)
        else:
            h, c = state

        xc = torch.cat([x, h], dim=1)
        gates = self.conv(xc)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class PhiLSTM(nn.Module):
    """
    LSTM "lent" pour Φ : on ne garde qu'un vecteur global comprimé.
    """
    def __init__(self, cfg: AetheriumConfig, latent_spatial: Tuple[int, int]):
        super().__init__()
        Hs, Ws = latent_spatial
        c_phi = cfg.latent_channels_phi
        self.flatten_dim = c_phi * Hs * Ws
        self.lstm = nn.LSTM(input_size=self.flatten_dim, hidden_size=self.flatten_dim)
        self.Hs = Hs
        self.Ws = Ws
        self.c_phi = c_phi

    def forward(
        self,
        phi_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, H, W = phi_t.shape
        x = phi_t.view(B, 1, -1)  # (B, 1, flatten_dim)
        x = x.transpose(0, 1)     # (1, B, D)

        if state is None:
            h0 = phi_t.new_zeros(1, B, self.flatten_dim)
            c0 = phi_t.new_zeros(1, B, self.flatten_dim)
            state = (h0, c0)

        out, state_next = self.lstm(x, state)  # out: (1,B,D)
        phi_next_flat = out.transpose(0, 1).contiguous().view(B, self.c_phi, self.Hs, self.Ws)
        return phi_next_flat, state_next


# ============================================================
# 4. DÉCODEUR CNN
# ============================================================

class Decoder(nn.Module):
    """
    Décodage latent → image (B, C, H, W).
    On part de z_cat = concat(phi, psi, omega).
    """
    def __init__(self, cfg: AetheriumConfig):
        super().__init__()
        C = cfg.base_channels
        c_lat = cfg.latent_channels_total
        self.deconv1 = nn.ConvTranspose2d(c_lat, C * 2, 4, 2, 1)  # H*2
        self.deconv2 = nn.ConvTranspose2d(C * 2, C, 4, 2, 1)      # H*4
        self.conv_out = nn.Conv2d(C, cfg.in_channels, 3, 1, 1)

    def forward(self, z_cat: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.deconv1(z_cat), 0.1)
        h = F.leaky_relu(self.deconv2(h), 0.1)
        x_rec = torch.sigmoid(self.conv_out(h))
        return x_rec


# ============================================================
# 5. WORLD MODEL Φ–Ψ–Ω
# ============================================================

class AetheriumWorldModel(nn.Module):
    """
    World Model complet :
      - encode frame_t
      - factorise en (Φ, Ψ, Ω)
      - fait évoluer Φ (LSTM global), Ψ/Ω (ConvLSTM locaux)
      - reconstruit A_t = concat(Φ,Ψ,Ω) → image_t+1
    """
    def __init__(self, cfg: AetheriumConfig, latent_spatial: Tuple[int, int]):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        Hs, Ws = latent_spatial
        c_phi = cfg.latent_channels_phi
        c_psi = cfg.latent_channels_psi
        c_omega = cfg.latent_channels_omega

        # Noyaux temporels
        self.phi_core = PhiLSTM(cfg, latent_spatial=(Hs, Ws))
        self.psi_core = ConvLSTMCell(c_psi + c_phi + c_omega, c_psi)
        self.omega_core = ConvLSTMCell(c_omega + c_psi, c_omega)

        # Tête physique (C_t, Δφ_t)
        self.physics_head = PhysicsHead(cfg, latent_spatial=(Hs, Ws))

    def forward(
        self,
        frames_init: torch.Tensor,
        steps_pred: Optional[int] = None,
        gw_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        frames_init : (B, T_in, C, H, W)
        gw_context  : (B, Dg) optionnel (paramètres GW / gravité globale)
        retourne :
          - pred_frames: (B, T_pred, C, H, W)
          - champs latents : phi_seq, psi_seq, omega_seq
        """
        cfg = self.cfg
        if steps_pred is None:
            steps_pred = cfg.steps_pred

        B, T_in, C, H, W = frames_init.shape

        # Encode dernier frame comme point de départ
        x0 = frames_init[:, -1]
        z0 = self.encoder(x0)  # (B, C_lat, H', W')
        Hs, Ws = z0.shape[-2:]

        phi_t, psi_t, omega_t = split_latent_phi_psi_omega(z0, cfg)

        phi_state = None
        psi_state = None
        omega_state = None

        pred_frames = []
        phi_seq = []
        psi_seq = []
        omega_seq = []
        phys_seq = []

        for t in range(steps_pred):
            # 1) Mise à jour Φ (lent)
            phi_t, phi_state = self.phi_core(phi_t, phi_state)

            # 2) Mise à jour Ψ (ConvLSTM, couplé à Φ/Ω)
            psi_input = torch.cat([psi_t, phi_t, omega_t], dim=1)
            psi_t, psi_state = self.psi_core(psi_input, psi_state)

            # 3) Mise à jour Ω (ConvLSTM, couplé à Ψ)
            omega_input = torch.cat([omega_t, psi_t], dim=1)
            omega_t, omega_state = self.omega_core(omega_input, omega_state)

            # 3bis) Prédiction physique (C_t, Δφ_t)
            y_t = self.physics_head(phi_t, psi_t, omega_t)

            # (Optionnel) injection gw_context ici pour moduler Φ/Ψ/Ω

            # 4) Reconstruction A_t latent + image
            z_cat = merge_phi_psi_omega(phi_t, psi_t, omega_t)
            x_pred = self.decoder(z_cat)

            pred_frames.append(x_pred)
            phi_seq.append(phi_t)
            psi_seq.append(psi_t)
            omega_seq.append(omega_t)

        pred_frames = torch.stack(pred_frames, dim=1)  # (B, T_pred, C, H, W)
        phi_seq = torch.stack(phi_seq, dim=1)
        psi_seq = torch.stack(psi_seq, dim=1)
        omega_seq = torch.stack(omega_seq, dim=1)
        phys_seq = torch.stack(phys_seq, dim=1)  # (B, T_pred, 2)

        return {
            "pred_frames": pred_frames,
            "phi_seq": phi_seq,
            "psi_seq": psi_seq,
            "omega_seq": omega_seq,
            "phys_seq": phys_seq,
        }


# ============================================================
# 6bis. PHYSICS HEAD : prédiction C_t et Δφ_t
# ============================================================

class PhysicsHead(nn.Module):
    """\
    Tête physique qui lit les champs (Φ, Ψ, Ω) à un instant t
    et prédit deux scalaires :
      - C_t : cohérence fluide (proxy de ton C dans le qubit fluide)
      - dphi_t : déphasage Δφ_t (proxy GW / lentille)
    """

    def __init__(self, cfg: AetheriumConfig, latent_spatial: Tuple[int, int]):
        super().__init__()
        Hs, Ws = latent_spatial

        in_channels = (
            cfg.latent_channels_phi
            + cfg.latent_channels_psi
            + cfg.latent_channels_omega
        )

        # petit CNN → global pooling → MLP (2 scalaires)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 2)  # [C_t, Δφ_t]

    def forward(
        self,
        phi_t: torch.Tensor,
        psi_t: torch.Tensor,
        omega_t: torch.Tensor,
    ) -> torch.Tensor:
        """\
        Retourne y_t: (B, 2) avec
          y_t[...,0] ~ C_t
          y_t[...,1] ~ Δφ_t
        """
        z_cat = torch.cat([phi_t, psi_t, omega_t], dim=1)  # (B, Cin, Hs, Ws)
        h = F.leaky_relu(self.conv1(z_cat), 0.1)
        h = F.leaky_relu(self.conv2(h), 0.1)
        h = self.pool(h).squeeze(-1).squeeze(-1)  # (B, 32)
        y = self.fc(h)  # (B, 2)
        return y


# ============================================================
# 6. FONCTIONS DE PERTE (CANEVAS)
# ============================================================

def loss_reconstruction(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    mode: str = "l1"
) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred_frames, target_frames)
    elif mode == "l2":
        return F.mse_loss(pred_frames, target_frames)
    else:
        raise ValueError(f"Unknown recon mode: {mode}")


def loss_spectral_coherence(psi_seq: torch.Tensor) -> torch.Tensor:
    """
    Pénalise les variations temporelles du spectre de Ψ (FFT 2D simple).
    psi_seq: (B, T, C, Hs, Ws)
    -> approximatif mais suffisant comme hook "ζ(s) fluide".
    """
    B, T, C, Hs, Ws = psi_seq.shape
    # On prend un canal moyen comme proxy
    psi_mean = psi_seq.mean(dim=2)  # (B, T, Hs, Ws)
    fft = torch.fft.rfft2(psi_mean, norm="ortho")  # (B, T, Hr, Wr_fft)
    power = (fft.abs() ** 2)  # spectre de puissance

    # Variance temporelle du spectre (on veut la garder faible)
    var_t = power.var(dim=1).mean()
    return var_t


def loss_phase_threshold(psi_seq: torch.Tensor, target_sigma: float = 0.10) -> torch.Tensor:
    """
    Approximation du seuil σ_Δφ ≈ 0.10 : on mesure la variance de la différence
    de "phase" temporelle et on la pousse vers target_sigma.
    """
    B, T, C, Hs, Ws = psi_seq.shape
    # Proxy phase : on traite un canal comme "phase"
    psi_one = psi_seq[:, :, 0]  # (B, T, Hs, Ws)
    # Différences temporelles
    dpsi = psi_one[:, 1:] - psi_one[:, :-1]  # (B, T-1, Hs, Ws)
    sigma = dpsi.std(dim=(1, 2, 3))  # (B,)
    sigma_mean = sigma.mean()
    # Pénalise |sigma_mean - target_sigma|
    return (sigma_mean - target_sigma).abs()


def total_loss(
    outputs: Dict[str, torch.Tensor],
    target_frames: torch.Tensor,
    lambda_spec: float = 1e-3,
    lambda_phase: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    L_rec = loss_reconstruction(outputs["pred_frames"], target_frames)
    L_spec = loss_spectral_coherence(outputs["psi_seq"])
    L_phase = loss_phase_threshold(outputs["psi_seq"])

    L_total = L_rec + lambda_spec * L_spec + lambda_phase * L_phase

    return {
        "L_total": L_total,
        "L_rec": L_rec,
        "L_spec": L_spec,
        "L_phase": L_phase,
    }


# ============================================================
# 7. BOUCLE D’ENTRAÎNEMENT (SQUELETTE)
# ============================================================

def train_step(
    model: AetheriumWorldModel,
    batch_frames: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: AetheriumConfig,
) -> Dict[str, float]:
    """
    batch_frames : (B, T_total, C, H, W)
    On utilise les T_in premières images comme conditionnement, le reste comme target.
    """
    model.train()
    optimizer.zero_grad()

    B, T_total, C, H, W = batch_frames.shape
    T_in = max(1, T_total // 2)
    T_pred = T_total - T_in

    frames_in = batch_frames[:, :T_in]
    frames_target = batch_frames[:, T_in:]  # (B, T_pred, C, H, W)

    outputs = model(frames_in, steps_pred=T_pred)
    losses = total_loss(outputs, frames_target)

    losses["L_total"].backward()
    optimizer.step()

    return {k: float(v.detach().cpu().item()) for k, v in losses.items()}
