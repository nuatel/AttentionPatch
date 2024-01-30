
class AttentionCore():
    def __init__(self, features, attentions, config):
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))

        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))



