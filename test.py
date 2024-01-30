
def Tester():
    self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
    self.index = faiss.read_index(os.path.join(self.embedding_dir_path, 'index.faiss'))
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    self.init_results_list()



    x, gt, label, file_name, x_type = batch
    # extract embedding
    features = self(x)
    embeddings = []
    for feature in features:
        m = torch.nn.AvgPool2d(3, 1, 1)
        embeddings.append(m(feature))
    embedding_ = embedding_concat(embeddings[0], embeddings[1])
    embedding_test = np.array(reshape_embedding(np.array(embedding_)))
    score_patches, _ = self.index.search(embedding_test, k=args.n_neighbors)
    anomaly_map = score_patches[:, 0].reshape((28, 28))
    N_b = score_patches[np.argmax(score_patches[:, 0])]
    w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
    score = w * max(score_patches[:, 0])  # Image-level score
    gt_np = gt.cpu().numpy()[0, 0].astype(int)
    anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
    anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
    self.gt_list_px_lvl.extend(gt_np.ravel())
    self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
    self.gt_list_img_lvl.append(label.cpu().numpy()[0])
    self.pred_list_img_lvl.append(score)
    self.img_path_list.extend(file_name)
    # save images
    x = self.inv_normalize(x)
    input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
    self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np * 255, file_name[0], x_type[0])

    print("Total pixel-level auc-roc score :")
    pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
    print(pixel_auc)
    print("Total image-level auc-roc score :")
    img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
    print(img_auc)
    print('test_epoch_end')
    values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
    self.log_dict(values)