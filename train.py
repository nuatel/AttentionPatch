

def Trainer(model, config):

    image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category), transform=self.data_transforms,
                                  gt_transform=self.gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)


    for batch in enumerate(dataloader):

        model = feature_extractor()


        self.model.eval()  # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []

        x, _, _, _, _ = batch
        features = model(x)


        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        embedding_list.extend(reshape_embedding(np.array(embedding)))

        total_embeddings = np.array(self.embedding_list)

        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto',
                                                      eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]

        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        # faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset)
        faiss.write_index(self.index, os.path.join(self.embedding_dir_path, 'index.faiss'))




