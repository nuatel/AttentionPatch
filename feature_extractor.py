with torch.no_grad():
    # Forward pass
    if config.model.feature_extractor == 'vit':
        #store the output of qkv layer from the last attention layer
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][-1]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        # Forward pass in the model
        attentions = model.get_last_selfattention(img[None, :, :, :,])
        # Scaling factor: patch_size là số patch cắt tấm ảnh ra
        scales = [config.patch_size, config.parch_size]

        # Dimensions
        nb_im = attentions.shape[0]  # batch size
        nh = attentions.shape[1] # number of heads
        nb_tokens = attentions.shape[2] # number of tokens
        if args.dinoseg:
            #pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
            pred = np.asarray(pred)
        else:
            # extract the qkv of the last attentions layer
            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            # modality selection
            if args.which_features == "k"
                feats = k
            elif args.which_features == "q":
                feats = q
            elif args.which_features == "v":
                feats = v

            if args.save_feat_dir is not None:
                np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
