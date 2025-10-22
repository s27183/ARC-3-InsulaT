import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellPositionEncoder(nn.Module):
    """Encodes cell positions as learnable smooth function of normalized coordinates.

    Instead of lookup table, learns a continuous function that maps (x, y) → position vector.
    This allows generalization and smooth interpolation for spatial positions.

    Args:
        pos_dim: Dimension of position encoding (default: 32)
        hidden_dim: Hidden layer dimension (default: pos_dim * 2)
        grid_size: Grid dimension (default: 64 for 64×64 grids)
    """

    def __init__(self, pos_dim: int = 32, hidden_dim: int = None, grid_size: int = 64):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = pos_dim * 2

        self.pos_dim = pos_dim
        self.grid_size = grid_size

        # Learnable smooth function: normalized (x, y) → position encoding
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2D coordinates → hidden
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pos_dim),  # hidden → position encoding
        )

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        """Encode positions for all cells in batch of grids.

        Args:
            grid_states: [batch, 64, 64] - Integer grids (only used for shape)

        Returns:
            pos_encodings: [batch, 64, 64, pos_dim] - Position encodings for each cell
        """
        batch_size = grid_states.shape[0]
        device = grid_states.device

        # Create normalized 2D coordinate grid: [0, 1] × [0, 1]
        # Shape: [64, 64, 2]
        y_coords = torch.linspace(0, 1, self.grid_size, device=device)
        x_coords = torch.linspace(0, 1, self.grid_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)  # [64, 64, 2]

        # Flatten coordinates: [4096, 2]
        coords_flat = coords.reshape(-1, 2)

        # Encode all positions: [4096, pos_dim]
        pos_encodings_flat = self.encoder(coords_flat)

        # Reshape to grid: [64, 64, pos_dim]
        pos_encodings = pos_encodings_flat.reshape(
            self.grid_size, self.grid_size, self.pos_dim
        )

        # Expand for batch: [batch, 64, 64, pos_dim]
        pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return pos_encodings


class InsularCellIntegration(nn.Module):
    """Integrates cell color and position using insular cortex-inspired mechanism.

    Biological Inspiration:
        Insular cortex integrates multimodal information (taste + temperature → flavor).
        Mechanism: Concatenate modalities → Learn fusion → Output unified representation.

    Architecture:
        - Concatenate color embedding + position encoding
        - Linear projection learns optimal fusion
        - LayerNorm stabilizes
        - GELU activation for smooth nonlinearity
        - Output dimension = color_dim (compression forces integration)

    Args:
        color_dim: Dimension of color embeddings (e.g., 64)
        pos_dim: Dimension of position encodings (e.g., 32)
    """

    def __init__(self, color_dim: int = 64, pos_dim: int = 32):
        super().__init__()
        self.color_dim = color_dim
        self.pos_dim = pos_dim
        # Output dimension hardcoded to color_dim for parameter efficiency
        # and to maintain patch projection dimensions unchanged
        output_dim = color_dim

        # Insular-inspired integration: concat → learn fusion → unified repr
        self.integration = nn.Sequential(
            nn.Linear(color_dim + pos_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self, color_embeddings: torch.Tensor, pos_encodings: torch.Tensor, patches: torch.Tensor
    ) -> torch.Tensor:
        """Integrate color and position into unified cell representations.

        Args:
            color_embeddings: [batch, 8, 8, 64, color_dim] - Embedded cell colors within patches
            pos_encodings: [batch, 64, 64, pos_dim] - Position encodings for each cell
            patches: [batch, 8, 8, 64] - Integer cell values (used to extract positions)

        Returns:
            integrated_cells: [batch, 8, 8, 64, color_dim] - Unified cell representations
        """
        batch_size = patches.shape[0]
        patch_size = 8  # 8×8 patches

        # Extract position encodings for cells in each patch
        # patches shape: [batch, 8, 8, 64]
        # pos_encodings shape: [batch, 64, 64, pos_dim]

        # Reshape patches to get cell coordinates: [batch, 8, 8, 8, 8]
        patches_2d = patches.view(batch_size, 8, 8, patch_size, patch_size)

        # Extract corresponding position encodings by unfolding pos_encodings
        # Unfold pos_encodings: [batch, 64, 64, pos_dim] → [batch, 8, 8, 8, 8, pos_dim]
        pos_unfolded = pos_encodings.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # pos_unfolded shape: [batch, 8, 8, pos_dim, 8, 8]
        # Permute to: [batch, 8, 8, 8, 8, pos_dim]
        pos_unfolded = pos_unfolded.permute(0, 1, 2, 4, 5, 3)

        # Flatten spatial dims within each patch: [batch, 8, 8, 64, pos_dim]
        pos_per_cell = pos_unfolded.reshape(batch_size, 8, 8, 64, -1)

        # Concatenate color + position: [batch, 8, 8, 64, color_dim + pos_dim]
        concat = torch.cat([color_embeddings, pos_per_cell], dim=-1)

        # Integrate: [batch, 8, 8, 64, color_dim]
        integrated = self.integration(concat)

        return integrated


class ViTStateEncoder(nn.Module):
    """Vision Transformer State Encoder with Learned Cell Embeddings.

    Part of InsulaAgent architecture (posterior insula analog).
    Encodes 64×64 grids into vector representations using patch-based
    self-attention with learned embeddings for each cell value (0-15)

    Args:
        num_colors: Number of possible cell values (default: 16 for colors 0-15)
        embed_dim: Transformer embedding dimension
        cell_embed_dim: Dimension for each cell embedding (0-15)
        patch_size: Size of each square patch (default: 8 for 8×8 patches)
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_cls_token: Whether to use CLS token (True) or global pooling (False)
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 256,
        cell_embed_dim: int = 64,
        patch_size: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        pos_dim_ratio: float = 0.5,  # Position encoding dimension as ratio of cell_embed_dim
        use_patch_pos_encoding: bool = False,  # Whether to use patch-level positional encoding
    ):
        super().__init__()

        self.num_colors = num_colors
        self.cell_embed_dim = cell_embed_dim
        self.patch_size = patch_size
        self.grid_size = 64
        num_patches_per_dim = self.grid_size // patch_size  # 8
        self.num_patches = num_patches_per_dim ** 2  # 64 patches for 8×8
        self.use_cls_token = use_cls_token
        self.use_patch_pos_encoding = use_patch_pos_encoding

        # Cell-level positional encoding (insular-inspired integration)
        pos_dim = int(cell_embed_dim * pos_dim_ratio)
        self.cell_position_encoder = CellPositionEncoder(
            pos_dim=pos_dim, grid_size=self.grid_size
        )
        self.insular_integration = InsularCellIntegration(
            color_dim=cell_embed_dim, pos_dim=pos_dim
        )

        # Learned cell embedding: each color (0-15) → vector
        self.cell_embedding = nn.Embedding(num_colors, cell_embed_dim)

        # Attention-based pooling components (pure attention, no alpha mixing)
        self.attention_head = nn.Linear(cell_embed_dim, 1)  # Compute attention scores

        # Patch projection: aggregated cell embeddings → transformer dimension
        self.patch_projection = nn.Linear(cell_embed_dim, embed_dim)

        # 2D learnable positional embeddings for patch grid (optional, now redundant with cell-level encoding)
        if use_patch_pos_encoding:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches_per_dim, num_patches_per_dim, embed_dim) * 0.02
            )
        else:
            self.pos_embed = None

        # CLS token for global representation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Post-norm (standard ViT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def _extract_patches(self, grid_states: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from integer grid.

        Args:
            grid_states: [batch, 64, 64] - Integer grid with values 0-15

        Returns:
            patches: [batch, num_patches_h, num_patches_w, patch_size*patch_size]
                     [batch, 8, 8, 64] - 64 cells per patch
        """
        batch_size = grid_states.shape[0]

        # Unfold to extract patches: [batch, 8, 8, 8, 8]
        patches = grid_states.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )

        # Flatten each patch: [batch, 8, 8, 64]
        num_patches_per_dim = self.grid_size // self.patch_size
        patches = patches.reshape(
            batch_size,
            num_patches_per_dim,
            num_patches_per_dim,
            self.patch_size * self.patch_size,  # 64 cells per patch
        )

        return patches


    def _embed_and_aggregate_patches(
        self, patches: torch.Tensor, pos_encodings: torch.Tensor
    ) -> torch.Tensor:
        """Embed each cell with insular integration and aggregate within patches using pure attention.

        Args:
            patches: [batch, 8, 8, 64] - Integer cell values 0-15
            pos_encodings: [batch, 64, 64, pos_dim] - Position encodings for each cell

        Returns:
            patch_embeddings: [batch, 8, 8, cell_embed_dim]
        """
        # Embed each cell color: [batch, 8, 8, 64, cell_embed_dim]
        # Ensure integer type for embedding lookup
        cell_color_embeddings = self.cell_embedding(patches.long())

        # Insular-inspired integration: color + position → unified cell representation
        # Shape: [batch, 8, 8, 64, cell_embed_dim]
        cell_embeddings = self.insular_integration(
            cell_color_embeddings, pos_encodings, patches
        )

        # Pure attention pooling (content-based aggregation, no spatial bias)
        # Compute attention scores: [batch, 8, 8, 64, 1]
        attention_scores = self.attention_head(cell_embeddings)

        # Softmax over cells within each patch (dim=3): [batch, 8, 8, 64, 1]
        attention_weights = F.softmax(attention_scores, dim=3)

        # Weighted sum: [batch, 8, 8, cell_embed_dim]
        patch_embeddings = (attention_weights * cell_embeddings).sum(dim=3)

        return patch_embeddings

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_states: [batch, 64, 64] - Integer grid with values 0-15

        Returns:
            state_repr: [batch, embed_dim] - State representations
        """
        batch_size = grid_states.shape[0]

        # Cell-level position encoding: [batch, 64, 64, pos_dim]
        pos_encodings = self.cell_position_encoder(grid_states)

        # Extract patches: [batch, 8, 8, 64]
        patches = self._extract_patches(grid_states)

        # Embed cells with insular integration and aggregate: [batch, 8, 8, cell_embed_dim]
        patch_repr = self._embed_and_aggregate_patches(patches, pos_encodings)

        # Project to transformer dimension: [batch, 8, 8, embed_dim]
        x = self.patch_projection(patch_repr)

        # Add 2D positional embeddings (optional, redundant with cell-level encoding)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Flatten to sequence: [batch, 64, embed_dim]
        x = x.reshape(batch_size, self.num_patches, -1)

        # Prepend CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, 65, embed_dim]

        # Transformer encoding
        x = self.transformer(x)

        # Extract global representation
        if self.use_cls_token:
            state_repr = x[:, 0]  # [batch, embed_dim] - CLS token
        else:
            state_repr = x.mean(dim=1)  # [batch, embed_dim] - Average pooling

        # Final normalization
        state_repr = self.norm(state_repr)

        return state_repr

class ActionEmbedding(nn.Module):
    """Action embedding for 4102 action vocabulary: ACTION1-5, ACTION7 + coordinates."""

    def __init__(self, embed_dim=256):
        super().__init__()
        # 4102 actions: ACTION1-5 (indices 0-4) + ACTION7 (index 5) + coordinates (indices 6-4101)
        self.action_embedding = nn.Embedding(4102, embed_dim)

    def forward(self, action_indices):
        """
        Args:
            action_indices: [batch, seq_len] with values in [0, 4101]
        Returns:
            action_embeddings: [batch, seq_len, embed_dim]
        """
        return self.action_embedding(action_indices)

class DecisionModel(nn.Module):
    """Insular cortex-inspired transformer for action prediction.

    Combines Vision Transformer (spatial processing, posterior insula) with
    Decision Transformer (temporal processing, anterior insula) for multi-level
    integration: cell → spatial → temporal → decision signals.

    Uses online supervised learning with self-generated labels from game outcomes.
    """

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        context_len=25,  # Number of past actions (k) - unified for all heads
        # ViT encoder parameters
        vit_cell_embed_dim=64,
        vit_patch_size=8,
        vit_num_layers=4,
        vit_num_heads=8,
        vit_dropout=0.1,
        vit_use_cls_token=True,
        vit_pos_dim_ratio=0.5,  # Cell position encoding dimension ratio
        vit_use_patch_pos_encoding=False,  # Whether to use patch-level positional encoding
        # Head configuration
        use_change_momentum_head=True,  # Whether to use change momentum prediction head
        use_completion_head=True,  # Whether to use completion prediction head
        use_gameover_head=True,    # Whether to use GAME_OVER prediction head
        # Learned decay configuration
        use_learned_decay=False,  # Whether to learn decay rates vs use fixed values
        change_decay_init=0.7,  # Initial decay rate for change head (init value if learned)
        change_momentum_decay_init=1.0,  # Initial decay rate for change_momentum head (action-level, no decay)
        completion_decay_init=0.8,  # Initial decay rate for completion head (init value if learned)
        gameover_decay_init=0.9,  # Initial decay rate for gameover head (init value if learned)
    ):
        super().__init__()

        self.use_change_momentum_head = use_change_momentum_head
        self.use_completion_head = use_completion_head
        self.use_gameover_head = use_gameover_head

        # Temporal decay rates (learned or fixed)
        # Store decay rates directly in (0, 1] - config validation ensures valid initialization
        # Properties will clamp to valid range to handle gradient descent edge cases
        self.use_learned_decay = use_learned_decay
        if use_learned_decay:
            # Learnable parameters (optimized by gradient descent)
            self.change_decay_param = nn.Parameter(torch.tensor(change_decay_init))
            if use_change_momentum_head:
                self.change_momentum_decay_param = nn.Parameter(torch.tensor(change_momentum_decay_init))
            if use_completion_head:
                self.completion_decay_param = nn.Parameter(torch.tensor(completion_decay_init))
            if use_gameover_head:
                self.gameover_decay_param = nn.Parameter(torch.tensor(gameover_decay_init))
        else:
            # Fixed buffers (saved with checkpoint but not optimized)
            self.register_buffer('change_decay_param', torch.tensor(change_decay_init))
            if use_change_momentum_head:
                self.register_buffer('change_momentum_decay_param', torch.tensor(change_momentum_decay_init))
            if use_completion_head:
                self.register_buffer('completion_decay_param', torch.tensor(completion_decay_init))
            if use_gameover_head:
                self.register_buffer('gameover_decay_param', torch.tensor(gameover_decay_init))

        # Component modules - Use ViT State Encoder with insular-inspired cell integration
        self.state_encoder = ViTStateEncoder(
            num_colors=16,
            embed_dim=embed_dim,
            cell_embed_dim=vit_cell_embed_dim,
            patch_size=vit_patch_size,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            dropout=vit_dropout,
            use_cls_token=vit_use_cls_token,
            pos_dim_ratio=vit_pos_dim_ratio,
            use_patch_pos_encoding=vit_use_patch_pos_encoding,
        )
        self.action_embedding = ActionEmbedding(embed_dim=embed_dim)

        # Positional encoding for temporal context
        # Sequence: state0, action0, state1, action1, ..., state_k (final state)
        # Total positions: context_len * 2 + 1
        # With context_len=25: (2*25 + 1) = 51 positions
        self.pos_embedding = nn.Parameter(
            torch.randn(context_len * 2 + 1, embed_dim) * 0.02
        )

        # Decoder-only transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Action head for predicting changes caused by discrete actions (ACTION1-5, ACTION7)
        self.change_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 6),  # ACTION1-5, ACTION7
        )

        # Coordinate head for predicting changes caused by spatial actions (64x64 coordinates)
        self.change_coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096),  # 64x64 coordinates
        )

        # Change momentum head (optional: predict change momentum)
        if self.use_change_momentum_head:
            # Predicts P(action causes MORE change than recent actions)
            # Learns to identify actions that build momentum toward impactful pattern completion
            # Reward semantics:
            #   - 1.0: Action changed MORE cells than previous action (momentum building)
            #   - 0.0: Action changed FEWER/EQUAL cells (momentum lost/maintained)
            self.change_momentum_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 6), # ACTION1-5, ACTION7
            )
            self.change_momentum_coord_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 4096),
            )

        # Completion head (optional: predict level completion)
        if self.use_completion_head:
            # Action head for predicting level completion caused by discrete actions (ACTION1-5, ACTION7)
            self.completion_action_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 6),  # ACTION1-5, ACTION7
            )

            # Coordinate head for predicting level completion caused by spatial actions (64x64 coordinates)
            self.completion_coord_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 4096),
            )

        # GAME_OVER head (optional: predict GAME_OVER avoidance)
        if self.use_gameover_head:
            # Action head for predicting GAME_OVER caused by discrete actions (ACTION1-5, ACTION7)
            self.gameover_action_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 6),  # ACTION1-5, ACTION7
            )

            # Coordinate head for predicting GAME_OVER caused by spatial actions (64x64 coordinates)
            self.gameover_coord_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 4096),  # 64x64 coordinates
            )

    def build_state_action_sequence(self, states, actions) -> torch.Tensor:
        """Build interleaved state-action sequence: [s₀, a₀, s₁, a₁, ..., s_{t-1}, a_{t-1}, s_t]

        Args:
            states: [batch, seq_len+1, 64, 64] - k+1 integer grids with cell values 0-15
            actions: [batch, seq_len] - k past actions (excluding current to predict)

        Returns:
            sequence: [batch, 2*seq_len+1, embed_dim] - Interleaved state-action sequence
        """
        batch_size = states.shape[0]
        seq_len = actions.shape[1]  # k past actions
        sequence_tokens = []

        # Build interleaved sequence: s₀, a₀, s₁, a₁, ..., s_{k-1}, a_{k-1}, s_k
        for t in range(seq_len):
            # State at time t
            state_repr = self.state_encoder(states[:, t])  # [batch, embed_dim]
            sequence_tokens.append(state_repr)

            # Action at time t
            action_embed = self.action_embedding(actions[:, t])  # [batch, embed_dim]
            sequence_tokens.append(action_embed)

        # Add final state (current state) - no action after this (we predict it)
        final_state_repr = self.state_encoder(states[:, -1])  # [batch, embed_dim]
        sequence_tokens.append(final_state_repr)

        # Stack sequence tokens: [batch, 2*seq_len+1, embed_dim]
        sequence = torch.stack(sequence_tokens, dim=1)

        # Add positional encoding
        seq_positions = min(sequence.shape[1], self.pos_embedding.shape[0])
        sequence = sequence + self.pos_embedding[:seq_positions].unsqueeze(0)

        return sequence

    def forward(self, states, actions, temporal_credit: bool = False) -> dict[str, torch.Tensor]:
        """Per-timestep forward modeling with mode-dependent output shapes.

        Training mode (replay): Predictions at ALL states (if temporal_credit=True) or FINAL state only (if False).
        Inference mode (real-time): Prediction at FINAL state only for action selection.

        Args:
            states: [batch, seq_len+1, 64, 64] - k+1 integer grids with cell values 0-15 (past + current)
            actions: [batch, seq_len] - k past actions (0-4101)
            temporal_credit: bool - If True, compute predictions at all timesteps for temporal credit assignment.
                                   If False, compute only final prediction (more memory efficient).
                                   Only used in training mode. Ignored during inference.

        Returns:
            dict[str, torch.Tensor]:
                - Training mode with temporal_credit=True: {"change_logits": [batch, seq_len+1, 4102], ...}
                - Training mode with temporal_credit=False: {"change_logits": [batch, 4102], ...}
                - Inference mode: {"change_logits": [batch, 4102], ...}
        """
        # Build interleaved state-action sequence
        sequence = self.build_state_action_sequence(states, actions)

        # Create causal attention mask for autoregressive modeling
        seq_len = sequence.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(sequence.device)

        # Transformer forward pass (autoregressive)
        transformer_output = self.transformer(
            sequence,
            tgt_mask=causal_mask,
            memory=sequence,  # Self-attention over the sequence
        )

        if self.training:
            # === TRAINING MODE ===
            if temporal_credit:
                # Predictions at ALL states for temporal credit assignment
                # Extract ALL state representations (states at even positions: 0, 2, 4, ...)
                state_reprs = transformer_output[:, ::2, :]  # [batch, seq_len+1, embed_dim]

                # Multi-head prediction - Change head (always present)
                change_action_logits = self.change_action_head(
                    state_reprs
                )  # [batch, seq_len+1, 6] - ACTION1-5 at each state
                change_coord_logits = self.change_coord_head(
                    state_reprs
                )  # [batch, seq_len+1, 4096] - coordinates at each state
                change_logits = torch.cat(
                    [change_action_logits, change_coord_logits], dim=2
                )  # [batch, seq_len+1, 4102] - concat on dim=2 for 3D tensors

                # Change momentum head (optional)
                if self.use_change_momentum_head:
                    change_momentum_action_logits = self.change_momentum_head(
                            state_reprs
                    ) # [batch, seq_len+1, 6]
                    change_momentum_coord_logits = self.change_momentum_coord_head(
                            state_reprs
                    ) # [batch, seq_len+1, 4096]
                    change_momentum_logits = torch.cat(
                        [change_momentum_action_logits, change_momentum_coord_logits], dim=2
                    )  # [batch, seq_len+1, 4102] - concat on dim=2 for 3D tensors
                else:
                    change_momentum_logits = None

                # Build output dict - always include change logits
                output = {
                        "change_logits": change_logits,
                        "change_momentum_logits": change_momentum_logits,
                }

                # Completion head (optional)
                if self.use_completion_head:
                    completion_action_logits = self.completion_action_head(
                        state_reprs
                    )  # [batch, seq_len+1, 6]
                    completion_coord_logits = self.completion_coord_head(
                        state_reprs
                    )  # [batch, seq_len+1, 4096]
                    completion_logits = torch.cat(
                        [completion_action_logits, completion_coord_logits], dim=2
                    )  # [batch, seq_len+1, 4102]
                    output["completion_logits"] = completion_logits

                # GAME_OVER head (optional)
                if self.use_gameover_head:
                    gameover_action_logits = self.gameover_action_head(
                        state_reprs
                    )  # [batch, seq_len+1, 6]
                    gameover_coord_logits = self.gameover_coord_head(
                        state_reprs
                    )  # [batch, seq_len+1, 4096]
                    gameover_logits = torch.cat(
                        [gameover_action_logits, gameover_coord_logits], dim=2
                    )  # [batch, seq_len+1, 4102]
                    output["gameover_logits"] = gameover_logits

                return output  # All [batch, seq_len+1, 4102]

            else:
                # Prediction at FINAL state only (memory efficient)
                # Extract final representation (current state)
                final_repr = transformer_output[:, -1]  # [batch, embed_dim]

                # Multi-head prediction - Change head (always present)
                change_action_logits = self.change_action_head(
                    final_repr
                )  # [batch, 6] - ACTION1-5
                change_coord_logits = self.change_coord_head(
                    final_repr
                )  # [batch, 4096] - coordinates
                change_logits = torch.cat(
                    [change_action_logits, change_coord_logits], dim=1
                )  # [batch, 4102] - concat on dim=1 for 2D tensors

                # Change momentum head (optional)
                if self.use_change_momentum_head:
                    change_momentum_action_logits = self.change_momentum_head(
                            final_repr
                    ) # [batch, 6]
                    change_momentum_coord_logits = self.change_momentum_coord_head(
                            final_repr
                    ) # [batch, 4096]
                    change_momentum_logits = torch.cat(
                        [change_momentum_action_logits, change_momentum_coord_logits], dim=1
                    )
                else:
                    change_momentum_logits = None

                # Build output dict - always include change logits
                output = {
                        "change_logits": change_logits,
                        "change_momentum_logits": change_momentum_logits
                }

                # Completion head (optional)
                if self.use_completion_head:
                    completion_action_logits = self.completion_action_head(
                        final_repr
                    )  # [batch, 6]
                    completion_coord_logits = self.completion_coord_head(
                        final_repr
                    )  # [batch, 4096]
                    completion_logits = torch.cat(
                        [completion_action_logits, completion_coord_logits], dim=1
                    )  # [batch, 4102]
                    output["completion_logits"] = completion_logits

                # GAME_OVER head (optional)
                if self.use_gameover_head:
                    gameover_action_logits = self.gameover_action_head(
                        final_repr
                    )  # [batch, 6]
                    gameover_coord_logits = self.gameover_coord_head(
                        final_repr
                    )  # [batch, 4096]
                    gameover_logits = torch.cat(
                        [gameover_action_logits, gameover_coord_logits], dim=1
                    )  # [batch, 4102]
                    output["gameover_logits"] = gameover_logits

                return output  # All [batch, 4102]

        else:
            # === INFERENCE MODE: Prediction at FINAL state only (real-time forward modeling) ===
            # Extract final representation (current state)
            final_repr = transformer_output[:, -1]  # [batch, embed_dim]

            # Multi-head prediction - Change head (always present)
            change_action_logits = self.change_action_head(
                final_repr
            )  # [batch, 6] - ACTION1-5, ACTION7
            change_coord_logits = self.change_coord_head(
                final_repr
            )  # [batch, 4096] - coordinates
            change_logits = torch.cat(
                [change_action_logits, change_coord_logits], dim=1
            )  # [batch, 4102] - concat on dim=1 for 2D tensors

            # Change momentum head (optional)
            if self.use_change_momentum_head:
                change_momentum_action_logits = self.change_momentum_head(
                        final_repr
                ) # [batch, 6] - ACTION1-5, ACTION7
                change_momentum_coord_logits = self.change_momentum_coord_head(
                        final_repr
                ) # [batch, 4096] - coordinates
                change_momentum_logits = torch.cat(
                    [change_momentum_action_logits, change_momentum_coord_logits], dim=1
                ) # [batch, 4102] - concat on dim=1 for 2D tensors
            else:
                change_momentum_logits = None

            # Build output dict - always include change logits
            output = {
                    "change_logits": change_logits,
                    "change_momentum_logits": change_momentum_logits
            }

            # Completion head (optional)
            if self.use_completion_head:
                completion_action_logits = self.completion_action_head(
                    final_repr
                )  # [batch, 6]
                completion_coord_logits = self.completion_coord_head(
                    final_repr
                )  # [batch, 4096]
                completion_logits = torch.cat(
                    [completion_action_logits, completion_coord_logits], dim=1
                )  # [batch, 4102]
                output["completion_logits"] = completion_logits

            # GAME_OVER head (optional)
            if self.use_gameover_head:
                gameover_action_logits = self.gameover_action_head(
                    final_repr
                )  # [batch, 6]
                gameover_coord_logits = self.gameover_coord_head(
                    final_repr
                )  # [batch, 4096]
                gameover_logits = torch.cat(
                    [gameover_action_logits, gameover_coord_logits], dim=1
                )  # [batch, 4102]
                output["gameover_logits"] = gameover_logits

            return output  # All [batch, 4102]

    @property
    def change_decay(self) -> float:
        """Current change head decay rate (learned or fixed).

        Returns:
            decay: float in (0, 1], clamped to valid range
        """
        return torch.clamp(self.change_decay_param, min=1e-7, max=1.0).item()

    @property
    def change_momentum_decay(self) -> float:
        """Current change_momentum head decay rate (learned or fixed).

        Returns:
            decay: float in (0, 1], clamped to valid range

        Raises:
            AttributeError: If change_momentum head is disabled
        """
        if not self.use_change_momentum_head:
            raise AttributeError(
                "change_momentum_decay_param not available when use_change_momentum_head=False"
            )
        return torch.clamp(self.change_momentum_decay_param, min=1e-7, max=1.0).item()

    @property
    def completion_decay(self) -> float:
        """Current completion head decay rate (learned or fixed).

        Returns:
            decay: float in (0, 1], clamped to valid range

        Raises:
            AttributeError: If completion head is disabled
        """
        if not self.use_completion_head:
            raise AttributeError(
                "completion_decay_param not available when use_completion_head=False"
            )
        return torch.clamp(self.completion_decay_param, min=1e-7, max=1.0).item()

    @property
    def gameover_decay(self) -> float:
        """Current gameover head decay rate (learned or fixed).

        Returns:
            decay: float in (0, 1], clamped to valid range

        Raises:
            AttributeError: If gameover head is disabled
        """
        if not self.use_gameover_head:
            raise AttributeError(
                "gameover_decay_param not available when use_gameover_head=False"
            )
        return torch.clamp(self.gameover_decay_param, min=1e-7, max=1.0).item()