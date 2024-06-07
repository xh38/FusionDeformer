import torch
from utilities.laplacian import find_edges
def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei]
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences[counts.cpu() == 2].to(device=indices.device)
def normal_consistency_loss(face_normals, connected_faces):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """

    loss = 1 - torch.cosine_similarity(face_normals[connected_faces[:, 0]], face_normals[connected_faces[:, 1]], dim=1)
    return (loss**2).mean()

def compute_normals(vertices, indices):
     # Compute the face normals
     a = vertices[indices][:, 0, :]
     b = vertices[indices][:, 1, :]
     c = vertices[indices][:, 2, :]
     face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)
     return face_normals