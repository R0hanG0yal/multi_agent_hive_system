def fuse(responses):
    if not responses: return "No response"
    # return the highest score
    best = max(responses, key=lambda r: r.get('score', 0))
    return best['content']
