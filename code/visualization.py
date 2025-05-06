import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_weights(source_points, target_points, attention_weights, top_k=10):
    source_np = source_points.detach().cpu().numpy()
    target_np = target_points.detach().cpu().numpy()
    attn_np = attention_weights.mean(dim=0).detach().cpu().numpy()
    flat_indices = np.argsort(attn_np.flatten())[-top_k:]
    source_indices, target_indices = np.unravel_index(flat_indices, attn_np.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], c='r', label='Source', alpha=0.3, s=10)
    ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], c='b', label='Target', alpha=0.3, s=10)
    ax.scatter(source_np[source_indices, 0], source_np[source_indices, 1], source_np[source_indices, 2],
               c='darkred', label='High Attention Source', s=40)
    ax.scatter(target_np[target_indices, 0], target_np[target_indices, 1], target_np[target_indices, 2],
               c='darkblue', label='High Attention Target', s=40)
    for i in range(top_k):
        src_idx = source_indices[i]
        tgt_idx = target_indices[i]
        xs = [source_np[src_idx, 0], target_np[tgt_idx, 0]]
        ys = [source_np[src_idx, 1], target_np[tgt_idx, 1]]
        zs = [source_np[src_idx, 2], target_np[tgt_idx, 2]]
        attn_value = attn_np[src_idx, tgt_idx]
        ax.plot(xs, ys, zs, c='g', alpha=attn_value / attn_np.max(),
                linewidth=1 + 2 * attn_value / attn_np.max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Registration with Attention Visualization')
    ax.legend()

    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    plt.close()