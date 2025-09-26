import torch
import torch.nn.functional as F

def rank_by_cosine(ref_vec, cand_vecs, candidate_texts):
    """
    Rank candidate vectors by their cosine similarity to the reference vector.
    
    Args:
        ref_vec (list or np.array): The reference embedding vector.
        cand_vecs (list of list or np.array): List of candidate embedding vectors.
        cand_texts (list of str): List of candidate texts corresponding to cand_vecs.
        
    Returns:
        ranked_texts (list of str): Candidate texts ranked by similarity to ref_vec.
    """
    ref_tensor = torch.tensor(ref_vec).unsqueeze(0)  # Shape: (1, dim)
    candidate_tensors = torch.tensor(cand_vecs)           # Shape: (N, dim)
    
    # Compute cosine similarities
    similarities = F.cosine_similarity(ref_tensor, candidate_tensors)  # Shape: (N,)
    
	# Create result items with similarity scores
    items = []
    for i, (sim, text) in enumerate(zip(similarities.tolist(), candidate_texts)):
        items.append({
			"index": i,
			"message": text,
			"cosine_similarity": sim,
		})
	
	# Sort by similarity in descending order
    items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    # print('##### CONFIRM AS A LIST - Ranked Messages by Cosine Similarity: ', type(items))
    return items