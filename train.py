from dataset import *
from networks import get_model


def Trainer(model, config):

    image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category), transform=self.data_transforms,
                                  gt_transform=self.gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)


    for im_id, inp in enumerate(train_loader):
        img = inp[0]
        img = img.cuda(non_blocking=True)
        # feature extraction
        with torch.no_grad():

            if "vit" in config.pretrained_model:
                # Lưu đầu ra của qkv layers từ attention layer cuối cùng
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qlv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
                # forward pass in the model
                attentions = model.get_last_selfaatention(img[None, :, :, :])

                scales = [config.patch_size, config.patchsize]
                # dimension
                nb_im = attentions.shape[0]
                nh = attentions.shape[1]
                nb_tokens = attentions.shape[2]

                qkv = (
                    feat_out["qkv"]
                    .reshape(nb_im, nb_tokens, 3, n, -1 // nh)
                    .permute(2, 0, 3, 1, 4)
                )

                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                # atten = Q*K.T
                v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            else:

                raise ValueError("Unknown model.")
            features = feature_slection(q, k, v)

            ####
            embeddings = []
            for feature in features:
                embeddings.append(feature)

            #embedding = embedding_concat(embeddings[0], embedding[1])
            embedding_list.extend(reshape_embedding(np.array(embedding)))
            total_embeddings = np.array(embedding_list)

            ## random projection
            randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
            randomprojector.fit(total_embeddings)

            # Coreset Subsampling
            selector = kCenterGreedy(total_embeddings, 0, 0)
            selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                                 N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
            embedding_coreset = total_embeddings[selected_idx]

            print('initial embedding size : ', total_embeddings.shape)
            print('final embedding size : ', embedding_coreset.shape)

            # faiss
            self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
            self.index.add(self.embedding_coreset)
            # luu faiss

            faiss.write_index(self.index, os.path.join(self.embedding_dir_path, 'index.faiss'))


