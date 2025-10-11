import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTStateEncoder(nn.Module):
    """Vision Transformer State Encoder with Learned Cell Embeddings.

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
    ):
        super().__init__()

        self.num_colors = num_colors
        self.cell_embed_dim = cell_embed_dim
        self.patch_size = patch_size
        self.grid_size = 64
        num_patches_per_dim = self.grid_size // patch_size  # 8
        self.num_patches = num_patches_per_dim ** 2  # 64 patches for 8×8
        self.use_cls_token = use_cls_token

        # Learned cell embedding: each color (0-15) → vector
        self.cell_embedding = nn.Embedding(num_colors, cell_embed_dim)

        # Attention-based pooling components
        self.cell_norm = nn.LayerNorm(cell_embed_dim)  # Normalize before attention
        self.attention_head = nn.Linear(cell_embed_dim, 1)  # Compute attention scores

        # Per-patch learnable alpha: [8, 8] grid of mixing coefficients
        # Each spatial patch position learns its own mean/attention balance
        self.alpha = nn.Parameter(torch.randn(num_patches_per_dim, num_patches_per_dim) * 0.02)

        # Patch projection: aggregated cell embeddings → transformer dimension
        self.patch_projection = nn.Linear(cell_embed_dim, embed_dim)

        # 2D learnable positional embeddings for patch grid
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches_per_dim, num_patches_per_dim, embed_dim) * 0.02
        )

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

    def _embed_and_aggregate_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed each cell and aggregate within patches using LayerNorm + Attention + Residual.

        Args:
            patches: [batch, 8, 8, 64] - Integer cell values 0-15

        Returns:
            patch_embeddings: [batch, 8, 8, cell_embed_dim]
        """
        # Embed each cell: [batch, 8, 8, 64, cell_embed_dim]
        # Ensure integer type for embedding lookup
        cell_embeddings = self.cell_embedding(patches.long())

        # LayerNorm for stable attention computation
        # Shape: [batch, 8, 8, 64, cell_embed_dim]
        normed_embeddings = self.cell_norm(cell_embeddings)

        # Compute attention scores on normalized features
        # Shape: [batch, 8, 8, 64, 1]
        attention_scores = self.attention_head(normed_embeddings)

        # Softmax over cells within each patch (dim=3)
        # Shape: [batch, 8, 8, 64, 1]
        attention_weights = F.softmax(attention_scores, dim=3)

        # Weighted sum using original (unnormalized) embeddings
        # Shape: [batch, 8, 8, cell_embed_dim]
        attended = (attention_weights * cell_embeddings).sum(dim=3)

        # Mean pooling (baseline/residual path)
        # Shape: [batch, 8, 8, cell_embed_dim]
        mean_pooled = cell_embeddings.mean(dim=3)

        # Per-patch learnable combination with sigmoid to bound alpha ∈ [0,1]
        # alpha shape: [8, 8] → broadcast to [1, 8, 8, 1]
        alpha = torch.sigmoid(self.alpha).unsqueeze(0).unsqueeze(-1)

        # Residual combination: (1-alpha)*mean + alpha*attended
        # Shape: [batch, 8, 8, cell_embed_dim]
        patch_embeddings = (1 - alpha) * mean_pooled + alpha * attended

        return patch_embeddings

    def forward(self, grid_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_states: [batch, 64, 64] - Integer grid with values 0-15

        Returns:
            state_repr: [batch, embed_dim] - State representations
        """
        batch_size = grid_states.shape[0]

        # Extract patches: [batch, 8, 8, 64]
        patches = self._extract_patches(grid_states)

        # Embed cells and aggregate: [batch, 8, 8, cell_embed_dim]
        patch_repr = self._embed_and_aggregate_patches(patches)

        # Project to transformer dimension: [batch, 8, 8, embed_dim]
        x = self.patch_projection(patch_repr)

        # Add 2D positional embeddings
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
    """Action embedding for 4101 action vocabulary: ACTION1-5 + coordinates."""

    def __init__(self, embed_dim=256):
        super().__init__()
        # 4101 actions: ACTION1-5 (indices 0-4) + coordinates (indices 5-4100)
        self.action_embedding = nn.Embedding(4101, embed_dim)

    def forward(self, action_indices):
        """
        Args:
            action_indices: [batch, seq_len] with values in [0, 4100]
        Returns:
            action_embeddings: [batch, seq_len, embed_dim]
        """
        return self.action_embedding(action_indices)

class DecisionTransformer(nn.Module):
    """End-to-end transformer for ARC-AGI action prediction using state-action sequences."""

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_context_len=20,
        # ViT encoder parameters
        vit_cell_embed_dim=64,
        vit_patch_size=8,
        vit_num_layers=4,
        vit_num_heads=8,
        vit_dropout=0.1,
        vit_use_cls_token=True,
    ):
        super().__init__()

        # Component modules - Use ViT State Encoder with learned cell embeddings
        self.state_encoder = ViTStateEncoder(
            num_colors=16,
            embed_dim=embed_dim,
            cell_embed_dim=vit_cell_embed_dim,
            patch_size=vit_patch_size,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            dropout=vit_dropout,
            use_cls_token=vit_use_cls_token,
        )
        self.action_embedding = ActionEmbedding(embed_dim=embed_dim)

        # Positional encoding for temporal context
        # Sequence: state0, action0, state1, action1, ..., state_k (final state)
        # Total positions: max_context_len * 2 + 1
        self.pos_embedding = nn.Parameter(
            torch.randn(max_context_len * 2 + 1, embed_dim) * 0.02
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

        # Action head for predicting changes caused by discrete actions (ACTION1-5)
        self.change_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5),  # ACTION1-5
        )

        # Coordinate head for predicting changes caused by spatial actions (64x64 coordinates)
        self.change_coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096),  # 64x64 coordinates
        )

        # Action head for predicting level completion caused by discrete actions (ACTION1-5)
        self.completion_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5),
        )

        # Coordinate head for predicting level completion caused by spatial actions (64x64 coordinates)
        self.completion_coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4096),
        )

        # Action head for predicting GAME_OVER caused by discrete actions (ACTION1-5)
        self.gameover_action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 5),  # ACTION1-5
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

    def forward(self, states, actions) -> dict[str, torch.Tensor]:
        """
        Args:
            states: [batch, seq_len+1, 64, 64] - k+1 integer grids with cell values 0-15 (past + current)
            actions: [batch, seq_len] - k past actions (0-4100)

        Returns:
            action_logits: [batch, 4101] - Logits over full action space for next action
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

        # Extract final representation (current state)
        final_repr = transformer_output[:, -1]  # [batch, embed_dim]

        # Multi-head prediction
        change_action_logits = self.change_action_head(
            final_repr
        )  # [batch, 5] - ACTION1-5
        completion_action_logits = self.completion_action_head(
            final_repr
        )  # [batch, 5] - ACTION1-5
        gameover_action_logits = self.gameover_action_head(
            final_repr
        )  # [batch, 5] - ACTION1-5
        change_coord_logits = self.change_coord_head(
            final_repr
        )  # [batch, 4096] - coordinates
        completion_coord_logits = self.completion_coord_head(
            final_repr
        )  # [batch, 4096] - coordinates
        gameover_coord_logits = self.gameover_coord_head(
            final_repr
        )  # [batch, 4096] - coordinates

        # Concatenate for compatibility with existing interface
        change_logits = torch.cat(
            [change_action_logits, change_coord_logits], dim=1
        )  # [batch, 4101]
        completion_logits = torch.cat(
            [completion_action_logits, completion_coord_logits], dim=1
        )  # [batch, 4101]
        gameover_logits = torch.cat(
            [gameover_action_logits, gameover_coord_logits], dim=1
        )  # [batch, 4101]

        return {
            "change_logits": change_logits,
            "completion_logits": completion_logits,
            "gameover_logits": gameover_logits,
        }